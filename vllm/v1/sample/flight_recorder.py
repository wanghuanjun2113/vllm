# SPDX-License-Identifier: Apache-2.0
"""Bounded, lossless flight recorder transport and standalone durable writer.

Only trusted local IPC uses pickle. Persisted arrays never use pickle.
"""
import atexit
import hashlib
import io
import json
import os
import pickle
import queue
import sqlite3
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

_ENABLED = bool(os.getenv("VLLM_FLIGHT_RECORDER_DIR"))
_transport = None


def enabled():
    return _ENABLED


def _read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        part = stream.read(size - len(data))
        if not part:
            raise EOFError("truncated recorder IPC")
        data.extend(part)
    return bytes(data)


class Transport:
    def __init__(self):
        self.root = Path(os.environ["VLLM_FLIGHT_RECORDER_DIR"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.queue = queue.Queue(maxsize=int(os.getenv("VLLM_FLIGHT_RECORDER_QUEUE", "8")))
        self.error = None
        self.closed = False
        self.proc = subprocess.Popen(
            [sys.executable, __file__, "--writer", str(self.root), str(os.getpid())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        )
        self.thread = threading.Thread(target=self._loop, daemon=True, name="trace-transfer")
        self.thread.start()
        atexit.register(self.close)

    def check(self):
        if self.error:
            raise RuntimeError("flight recorder failed; trace is incomplete") from self.error

    def submit(self, meta, arrays=None, event=None, on_done=None):
        self.check()
        while True:
            try:
                self.queue.put((meta, arrays or {}, event, on_done), timeout=0.2)
                self.check()
                return
            except queue.Full:
                self.check()

    def _loop(self):
        try:
            while True:
                item = self.queue.get()
                try:
                    if item is None:
                        return
                    meta, arrays, event, on_done = item
                    if event is not None:
                        event.synchronize()
                    # CPU tensors and their pinned storage stay alive until IPC completes.
                    arrays = {k: v.numpy() if hasattr(v, "numpy") else v
                              for k, v in arrays.items()}
                    payload = pickle.dumps((meta, arrays), protocol=5)
                    self.proc.stdin.write(struct.pack("<Q", len(payload)))
                    self.proc.stdin.write(payload)
                    self.proc.stdin.flush()
                    if self.proc.stdout.read(1) != b"1":
                        raise IOError("writer died before durable acknowledgement")
                    if on_done is not None:
                        on_done()
                finally:
                    self.queue.task_done()
        except BaseException as exc:
            self.error = exc
            (self.root / ("FAILED-" + str(os.getpid()))).write_text(repr(exc))

    def flush(self):
        while self.queue.unfinished_tasks:
            self.check()
            time.sleep(0.01)
        self.check()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.flush()
        self.queue.put(None)
        self.thread.join()
        self.proc.stdin.close()
        if self.proc.wait(timeout=30) != 0:
            raise RuntimeError("flight recorder writer exited unsuccessfully")


def transport():
    global _transport
    if _transport is None:
        _transport = Transport()
    return _transport


def record_event(kind, request_id, **values):
    if enabled():
        transport().submit(dict(kind=kind, request_id=request_id, **values))


def request_start(request):
    if not enabled():
        return
    import msgspec
    params = request.sampling_params
    record_event("request", request.request_id,
                 external_req_id=request.external_req_id,
                 prompt_token_ids=request.prompt_token_ids,
                 sampling_params=msgspec.to_builtins({name: getattr(params, name)
                     for name in params.__struct_fields__}),
                 sampling_params_complete=True, timestamp=time.time())



def summarize_shared_snapshot(meta, arrays, mappings, executor):
    """Read a pool slot only while its producer retains ownership (until ACK)."""
    import mmap
    import numpy as np
    started = time.perf_counter_ns()
    path = meta["shared_path"]
    if path not in mappings:
        with open(path, "rb") as source:
            mappings[path] = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
    capacity, vocab, slot = meta["capacity_rows"], meta["vocab"], meta["shared_slot"]
    plane_bytes = capacity * vocab * 4
    raw = np.ndarray((capacity, vocab), np.float32, buffer=mappings[path],
                     offset=slot * 2 * plane_bytes)
    final = np.ndarray((capacity, vocab), np.float32, buffer=mappings[path],
                       offset=slot * 2 * plane_bytes + plane_bytes)
    selected = arrays["selected"].reshape(-1)
    width = meta["width"]
    drafts = meta["num_draft_tokens"]
    starts = np.cumsum([0] + [n + 1 for n in drafts[:-1]]) if drafts else np.arange(len(meta["req_ids"]))
    ignored = set(meta["discard_rows"])
    tasks = []
    for flat, token in enumerate(selected):
        req, column = divmod(flat, width)
        if token < 0 or req in ignored:
            continue
        row = int(starts[req] + min(column, drafts[req])) if drafts else req
        if not arrays["row_filled"][row]:
            raise ValueError("missing processed snapshot row")
        tasks.append((flat, row, int(token)))
    k = meta["top_k"]
    probe_ids = meta["probe_ids"]

    def statistics(row, token):
        ids = np.argpartition(row, vocab-k)[-k:]
        vals = row[ids]
        order = np.lexsort((ids, -vals))
        maximum = float(np.max(row))
        if np.isfinite(maximum):
            shifted = row.astype(np.float64) - maximum
            np.exp(shifted, out=shifted)
            lse = maximum + np.log(shifted.sum())
        else:
            lse = maximum  # Preserve NaN/Inf evidence; no invented distribution.
        probes = row[[token] + probe_ids]
        return {
            "ids": ids[order].astype(np.int32), "values": vals[order],
            "lse": lse,
            "counts": np.array([np.isfinite(row).sum(), np.isnan(row).sum(),
                                np.isposinf(row).sum(), np.isneginf(row).sum()],np.int32),
            "probe_values": probes,
            "probe_ranks": np.array([np.count_nonzero(row > value)+1 for value in probes],np.int32),
        }

    def calculate(task):
        flat, row, token = task
        raw_stats = statistics(raw[row], token)
        # Exact snapshot equality permits reusing statistics, including counts/ranks.
        final_stats = raw_stats if np.array_equal(raw[row], final[row]) else statistics(final[row], token)
        return flat, raw_stats, final_stats

    result = {"selected": selected.astype(np.int32), "final_seen": np.zeros(len(selected),bool)}
    for stage in ("raw","final"):
        for key, shape, dtype in (
            ("ids",(len(selected),k),np.int32),("values",(len(selected),k),np.float32),
            ("lse",(len(selected),),np.float64),("counts",(len(selected),4),np.int32),
            ("probe_values",(len(selected),len(probe_ids)+1),np.float32),
            ("probe_ranks",(len(selected),len(probe_ids)+1),np.int32)):
            result[stage+"_"+key] = np.zeros(shape,dtype)
    for flat, raw_stats, final_stats in executor.map(calculate, tasks):
        result["final_seen"][flat] = True
        for stage, stats in (("raw",raw_stats),("final",final_stats)):
            for key,value in stats.items():
                result[stage+"_"+key][flat] = value
    if "draft_ids" in arrays:
        result["draft_ids"] = arrays["draft_ids"]
    meta = {key:value for key,value in meta.items()
            if key not in ("shared_path","capacity_rows","vocab","shared_slot")}
    meta["kind"] = "batch"
    meta["summary_backend"] = "cpu_pool"
    meta["cpu_statistics_ns"] = time.perf_counter_ns() - started
    return meta,result


def writer(root, shard):
    import numpy as np
    root = Path(root)
    db = sqlite3.connect(root / (shard + ".sqlite"))
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS chunks "
               "(id INTEGER PRIMARY KEY, offset INTEGER, size INTEGER, sha256 TEXT, meta TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS tokens "
               "(request_id TEXT, position INTEGER, token_id INTEGER, chunk INTEGER, row INTEGER, "
               "PRIMARY KEY(request_id,position))")
    db.execute("CREATE TABLE IF NOT EXISTS events "
               "(id INTEGER PRIMARY KEY, request_id TEXT, kind TEXT, data TEXT)")
    positions = {}
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="trace-cpu")
    mappings = {}
    with (root / (shard + ".bin")).open("ab") as f:
        while True:
            header = sys.stdin.buffer.read(8)
            if not header:
                break
            if len(header) != 8:
                raise EOFError("partial IPC header")
            meta, arrays = pickle.loads(_read_exact(sys.stdin.buffer, struct.unpack("<Q", header)[0]))
            if meta["kind"] == "snapshot":
                meta, arrays = summarize_shared_snapshot(meta, arrays, mappings, executor)
            if meta["kind"] == "batch":
                selected = arrays["selected"].reshape(-1)
                width = meta["width"]
                valid = selected >= 0
                for row in meta["discard_rows"]:
                    valid[row * width:(row + 1) * width] = False
                flat_slots = np.flatnonzero(valid)
                out = {k: a[valid] for k, a in arrays.items()
                       if k not in ("draft_ids",)}
                out["flat_slots"] = flat_slots.astype(np.int32)
                if "draft_ids" in arrays:
                    meta["draft_ids"] = arrays["draft_ids"].tolist()
                stream = io.BytesIO()
                np.savez_compressed(stream, **out)
                data = stream.getvalue()
                offset = f.tell()
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
                chunk = db.execute("INSERT INTO chunks(offset,size,sha256,meta) VALUES(?,?,?,?)",
                                   (offset, len(data), hashlib.sha256(data).hexdigest(),
                                    json.dumps(meta, ensure_ascii=False))).lastrowid
                for n, slot in enumerate(flat_slots):
                    rid = meta["req_ids"][int(slot) // width]
                    pos = positions.get(rid, 0)
                    db.execute("INSERT INTO tokens VALUES(?,?,?,?,?)",
                               (rid, pos, int(selected[slot]), chunk, n))
                    positions[rid] = pos + 1
            else:
                db.execute("INSERT INTO events(request_id,kind,data) VALUES(?,?,?)",
                           (meta["request_id"], meta["kind"], json.dumps(meta, ensure_ascii=False)))
            db.commit()
            sys.stdout.buffer.write(b"1")
            sys.stdout.buffer.flush()
    db.close()
    executor.shutdown()
    for mapping in mappings.values():
        mapping.close()
    (root / (shard + ".closed")).write_text("durably closed\n")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--writer":
        writer(sys.argv[2], sys.argv[3])

# Registered only by the backend recorder; no backend dependency on normal runs.
capture_callback = None


def capture_processed(logits, metadata):
    if capture_callback is not None:
        capture_callback(logits, metadata)
