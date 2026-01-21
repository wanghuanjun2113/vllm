"""
Chunk Metadata for Position-Agnostic Chunk Cache

This module defines the data structures for chunk metadata used in
the position-agnostic KV cache system.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks


@dataclass
class ChunkMetadata:
    """
    Metadata for a single chunk in the chunk cache system.

    Attributes:
        chunk_id: Unique identifier for this chunk
        token_ids: Token IDs for this chunk (system_prompt + chunk content)
        chunk_hash: Content-based hash (position-agnostic)
        position_offset: Starting position offset for this chunk
        dependencies: List of chunk IDs this chunk depends on
        kv_cache_blocks: Cached KV cache blocks (if available)
    """
    chunk_id: str
    token_ids: List[int]
    chunk_hash: str = ""
    position_offset: int = 0
    dependencies: List[str] = field(default_factory=list)
    kv_cache_blocks: Optional["KVCacheBlocks"] = None

    def __post_init__(self):
        """Compute hash if not provided."""
        if not self.chunk_hash and self.token_ids:
            from vllm_ascend.chunk.hash_utils import compute_chunk_hash
            self.chunk_hash = compute_chunk_hash(self.token_ids)

    def get_num_tokens(self) -> int:
        """Get the number of tokens in this chunk."""
        return len(self.token_ids)

    def compute_hash(self) -> str:
        """
        Compute position-agnostic content hash for this chunk.

        Returns:
            The hash string for this chunk's content.
        """
        from vllm_ascend.chunk.hash_utils import compute_chunk_hash
        self.chunk_hash = compute_chunk_hash(self.token_ids)
        return self.chunk_hash

    def has_cached_kv(self) -> bool:
        """Check if this chunk has cached KV cache."""
        return self.kv_cache_blocks is not None


@dataclass
class ChunkParseResult:
    """
    Result of parsing a prompt with chunk delimiters.

    Attributes:
        system_prompt: The system prompt (before first delimiter)
        chunks: List of chunk strings (between delimiters)
        user_question: The user question (after last delimiter)
        delimiters: Positions of delimiters in the original prompt
        has_chunks: Whether the prompt contains chunks
    """
    system_prompt: str
    chunks: List[str]
    user_question: str
    delimiters: List[int] = field(default_factory=list)
    has_chunks: bool = True

    def get_num_chunks(self) -> int:
        """Get the number of chunks in this result."""
        return len(self.chunks)

    def is_empty(self) -> bool:
        """Check if the parse result is empty (no chunks)."""
        return not self.has_chunks or self.get_num_chunks() == 0


@dataclass
class ChunkMatchResult:
    """
    Result of matching chunks against the cache.

    Attributes:
        matched_chunks: Dictionary of chunk_index -> ChunkKVCache for hits
        missing_chunks: List of chunk indices that missed the cache
        hit_ratio: Ratio of chunks that hit the cache (0.0 - 1.0)
    """
    matched_chunks: dict = field(default_factory=dict)
    missing_chunks: List[int] = field(default_factory=list)
    hit_ratio: float = 0.0

    def get_hit_count(self) -> int:
        """Get the number of chunks that hit the cache."""
        return len(self.matched_chunks)

    def get_miss_count(self) -> int:
        """Get the number of chunks that missed the cache."""
        return len(self.missing_chunks)

    def get_total_count(self) -> int:
        """Get the total number of chunks."""
        return self.get_hit_count() + self.get_miss_count()


@dataclass
class ChunkKVCache:
    """
    KV cache storage for a single chunk.

    Attributes:
        chunk_hash: Hash of the chunk content
        key_cache: Key cache tensor
        value_cache: Value cache tensor
        num_tokens: Number of tokens in the chunk
        last_accessed: Timestamp of last access (for LRU)
        ref_count: Reference count for shared chunks
    """
    chunk_hash: str
    key_cache: "torch.Tensor"
    value_cache: "torch.Tensor"
    num_tokens: int
    last_accessed: float = 0.0
    ref_count: int = 1

    def __post_init__(self):
        """Initialize last_accessed on creation."""
        import time
        if self.last_accessed == 0.0:
            self.last_accessed = time.time()

    def touch(self):
        """Update last_accessed timestamp."""
        import time
        self.last_accessed = time.time()

    def increment_ref(self):
        """Increment reference count."""
        self.ref_count += 1

    def decrement_ref(self):
        """Decrement reference count."""
        if self.ref_count > 0:
            self.ref_count -= 1

    def is_shared(self) -> bool:
        """Check if this cache is shared by multiple requests."""
        return self.ref_count > 1

    def get_size_bytes(self) -> int:
        """
        Get the size of this KV cache in bytes.

        Returns:
            Size in bytes (key_cache + value_cache).
        """
        return self.key_cache.element_size() * self.key_cache.nelement() + \
               self.value_cache.element_size() * self.value_cache.nelement()


@dataclass
class CacheStats:
    """
    Statistics for the chunk cache pool.

    Attributes:
        total_size_gb: Total cache pool size in GB
        used_size_gb: Currently used size in GB
        hit_count: Number of cache hits
        miss_count: Number of cache misses
        total_chunks: Total number of chunks in cache
        shared_chunks: Number of chunks with ref_count > 1
    """
    total_size_gb: float = 0.0
    used_size_gb: float = 0.0
    hit_count: int = 0
    miss_count: int = 0
    total_chunks: int = 0
    shared_chunks: int = 0

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hit_count + self.miss_count
        if total == 0:
            return 0.0
        return self.hit_count / total

    @property
    def usage_ratio(self) -> float:
        """Get cache usage ratio."""
        if self.total_size_gb == 0:
            return 0.0
        return self.used_size_gb / self.total_size_gb

    def __str__(self) -> str:
        """String representation of cache stats."""
        return (
            f"CacheStats(hit_rate={self.hit_rate:.2%}, "
            f"usage={self.usage_ratio:.2%}, "
            f"hits={self.hit_count}, "
            f"misses={self.miss_count}, "
            f"chunks={self.total_chunks}, "
            f"shared={self.shared_chunks})"
        )
