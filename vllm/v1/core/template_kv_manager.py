# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Template KV Cache Manager for API-based prefix caching.

This module manages the storage and retrieval of KV cache for API templates.
It handles template registration, cache eviction, and provides metrics for monitoring.
"""

import threading
import time
from typing import TYPE_CHECKING, Optional

from vllm.logger import init_logger
from vllm.utils import get_dtype_size

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.template_registry import APITemplate
    from vllm.v1.kv_cache_interface import KVCacheBlock

logger = init_logger(__name__)


class TemplateKVCacheManager:
    """Manages KV cache storage and retrieval for API templates.

    This class is responsible for:
    1. Computing and caching KV cache for templates (one-time prefill)
    2. Retrieving cached blocks for template requests
    3. LRU eviction when cache is full
    4. Providing metrics for monitoring

    Template caching happens once per template when the first request arrives.
    The cached KV blocks are then reused for all subsequent requests with that template,
    regardless of which APIs are selected (different masks are applied per request).

    Example workflow:
        1. First request with template "api_v1" arrives
        2. Manager detects template not cached
        3. Triggers registration: runs ~20K token prefill
        4. Stores KV blocks in template_blocks dict
        5. Subsequent requests retrieve blocks instantly
    """

    def __init__(
        self,
        block_pool: "BlockPool",
        registry,
        max_cached_templates: int = 5,
    ):
        """Initialize the template KV cache manager.

        Args:
            block_pool: The BlockPool for allocating KV cache blocks
            registry: The TemplateRegistry for accessing template metadata
            max_cached_templates: Maximum number of templates to cache simultaneously
        """
        self.block_pool = block_pool
        self.registry = registry
        self.max_cached_templates = max_cached_templates

        # template_id -> list of KVCacheBlock
        self.template_blocks: dict[str, list["KVCacheBlock"]] = {}

        # LRU tracking: template_id -> last access timestamp
        self.last_access: dict[str, float] = {}

        # Registration lock to prevent concurrent registration
        self.registration_locks: dict[str, threading.Lock] = {}
        self.registration_lock = threading.Lock()

        # Pending templates waiting for registration
        self.pending_registrations: dict[str, "APITemplate"] = {}
        self.pending_lock = threading.Lock()

        # Metrics tracking
        self.total_registrations = 0
        self.total_evictions = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def get_template_blocks(
        self,
        template_id: str,
    ) -> Optional[list["KVCacheBlock"]]:
        """Retrieve cached blocks for a template.

        This is the main method called by the KV cache manager during request processing.
        Returns the pre-computed KV blocks if cached, None otherwise.

        Args:
            template_id: The template identifier

        Returns:
            List of KVCacheBlocks if cached, None if not cached
        """
        blocks = self.template_blocks.get(template_id)

        if blocks is not None:
            # Cache hit - update LRU timestamp
            self.last_access[template_id] = time.time()
            self.cache_hits += 1
            logger.debug(f"Template cache HIT for '{template_id}'")
        else:
            # Cache miss
            self.cache_misses += 1
            logger.debug(f"Template cache MISS for '{template_id}'")

        return blocks

    def cache_template(
        self,
        template: "APITemplate",
        model_executor: "ModelRunner",
        blocking: bool = True,
    ) -> bool:
        """Compute and cache KV cache for a template.

        This method performs a "cold prefill" of the full template prompt.
        For a 50-API template with ~20K tokens, this takes several seconds but is
        a one-time cost that is amortized over all subsequent requests.

        Args:
            template: The APITemplate to cache
            model_executor: Model executor for running prefill
            blocking: If True, block until complete. If False, run in background.

        Returns:
            True if caching succeeded or was already cached, False on error

        Thread Safety:
            Multiple threads can call this for the same template safely.
            The first thread performs registration, others wait for completion.
        """
        # Check if already cached
        if template.is_cached:
            logger.info(f"Template '{template.template_id}' already cached")
            return True

        # Get or create per-template lock
        with self.registration_lock:
            if template.template_id not in self.registration_locks:
                self.registration_locks[template.template_id] = threading.Lock()
            template_lock = self.registration_locks[template.template_id]

        # Acquire template-specific lock
        with template_lock:
            # Double-check after acquiring lock
            if template.is_cached:
                return True

            # Check if we need to evict first
            if len(self.template_blocks) >= self.max_cached_templates:
                if template.template_id not in self.template_blocks:
                    evicted_id = self.evict_lru_template()
                    if evicted_id is None:
                        logger.error(
                            f"Failed to evict template, cannot cache '{template.template_id}'"
                        )
                        return False

            logger.info(
                f"Caching template '{template.template_id}' "
                f"(~{len(template.prompt_token_ids)} tokens)"
            )

            try:
                # Allocate blocks
                num_blocks = self._calculate_num_blocks(template)
                blocks = self.block_pool.get_new_blocks(num_blocks)

                if blocks is None or len(blocks) < num_blocks:
                    logger.error(
                        f"Failed to allocate {num_blocks} blocks for template "
                        f"'{template.template_id}'"
                    )
                    return False

                # Run prefill to compute KV cache
                start_time = time.time()

                # Call model executor to compute KV cache
                # (This is a placeholder - actual integration depends on ModelExecutor interface)
                kv_cache = self._compute_template_kv(
                    template,
                    blocks,
                    model_executor,
                )

                elapsed = time.time() - start_time

                # Store blocks
                self.template_blocks[template.template_id] = blocks
                template.is_cached = True
                self.last_access[template.template_id] = time.time()
                self.total_registrations += 1

                logger.info(
                    f"Template '{template.template_id}' cached successfully "
                    f"in {elapsed:.2f}s ({num_blocks} blocks, {len(template.prompt_token_ids)} tokens)"
                )

                return True

            except Exception as e:
                logger.error(f"Failed to cache template '{template.template_id}': {e}")
                return False

    def _calculate_num_blocks(self, template: "APITemplate") -> int:
        """Calculate number of blocks needed for a template.

        Args:
            template: The APITemplate

        Returns:
            Number of blocks required
        """
        block_size = self.block_pool.block_size
        num_tokens = len(template.prompt_token_ids)
        return (num_tokens + block_size - 1) // block_size

    def _compute_template_kv(
        self,
        template: "APITemplate",
        blocks: list["KVCacheBlock"],
        model_executor: "ModelRunner",
    ):
        """Compute KV cache for a template by running model prefill.

        This is a placeholder method that needs to be integrated with the actual
        ModelExecutor. The implementation will depend on the vLLM version and
        the ModelExecutor interface.

        Args:
            template: The APITemplate with prompt tokens
            blocks: Allocated KV cache blocks
            model_executor: Model executor instance

        Returns:
            KV cache data (implementation-specific)
        """
        # TODO: Integrate with actual ModelExecutor
        # This will involve calling something like:
        # model_executor.execute_model(
        #     prompt_token_ids=template.prompt_token_ids,
        #     kv_cache_blocks=blocks,
        # )
        logger.warning(
            "Template KV computation not yet integrated with ModelExecutor. "
            "This is a placeholder implementation."
        )
        return None

    def mark_for_registration(self, template: "APITemplate") -> None:
        """Mark a template for lazy registration.

        This adds the template to a pending queue. Registration should be
        triggered externally (e.g., by LLMEngine during initialization
        or by a background process).

        Args:
            template: The APITemplate to register
        """
        with self.pending_lock:
            if template.template_id not in self.template_blocks:
                self.pending_registrations[template.template_id] = template
                logger.info(
                    f"Template '{template.template_id}' marked for pending registration "
                    f"({len(template.prompt_token_ids)} tokens, {len(template.api_descriptions)} APIs)"
                )

    def get_pending_registrations(self) -> list["APITemplate"]:
        """Get all templates waiting for registration.

        Returns:
            List of templates pending registration
        """
        with self.pending_lock:
            return list(self.pending_registrations.values())

    def clear_pending_registration(self, template_id: str) -> None:
        """Remove a template from the pending queue.

        Args:
            template_id: The template to remove from pending queue
        """
        with self.pending_lock:
            self.pending_registrations.pop(template_id, None)

    def manual_register_template(
        self,
        template_id: str,
        prompt_tokens: list[int],
        model_executor: "ModelRunner" = None,
    ) -> bool:
        """Manually trigger template registration.

        This allows external systems (like LLMEngine) to register templates
        when they have access to the model executor.

        Args:
            template_id: ID of the template to register
            prompt_tokens: Token IDs for the full template prompt
            model_executor: Model executor instance (optional)

        Returns:
            True if registration succeeded, False otherwise
        """
        template = self.registry.get_template(template_id)
        if template is None:
            logger.error(f"Cannot register unknown template '{template_id}'")
            return False

        if template.is_cached:
            logger.info(f"Template '{template_id}' already cached")
            return True

        if model_executor is None:
            logger.error(
                f"Model executor is required for template registration. "
                f"Please provide model_executor or call this during initialization."
            )
            return False

        return self.cache_template(template, model_executor)

    def evict_lru_template(self) -> Optional[str]:
        """Evict the least recently used template from cache.

        LRU (Least Recently Used) eviction is triggered when:
        - Cache is at capacity (max_cached_templates)
        - A new template needs to be cached

        Args:
            None

        Returns:
            The template_id that was evicted, or None if nothing to evict
        """
        if not self.template_blocks:
            return None

        # Find LRU template
        lru_id = min(self.last_access, key=self.last_access.get)

        # Get blocks to free
        blocks = self.template_blocks.pop(lru_id, None)
        self.last_access.pop(lru_id, None)
        self.registration_locks.pop(lru_id, None)

        if blocks is None:
            return None

        # Free blocks in reverse order (LIFO)
        for block in reversed(blocks):
            self.block_pool.free_block(block)

        self.total_evictions += 1

        logger.info(f"Evicted LRU template '{lru_id}' ({len(blocks)} blocks)")

        return lru_id

    def remove_template(self, template_id: str) -> bool:
        """Manually remove a template from cache.

        This can be used for cache invalidation or manual management.

        Args:
            template_id: The template to remove

        Returns:
            True if template was removed, False if it didn't exist
        """
        blocks = self.template_blocks.pop(template_id, None)
        self.last_access.pop(template_id, None)
        self.registration_locks.pop(template_id, None)

        if blocks is None:
            return False

        # Free blocks
        for block in reversed(blocks):
            self.block_pool.free_block(block)

        logger.info(f"Manually removed template '{template_id}' from cache")

        return True

    def get_cache_stats(self) -> dict:
        """Get cache statistics for monitoring.

        Returns:
            Dictionary with cache metrics:
            - num_cached_templates: Number of templates currently cached
            - max_cached_templates: Maximum cache capacity
            - cached_template_ids: List of cached template IDs
            - cache_hit_rate: Ratio of hits to total lookups
            - total_registrations: Total number of template registrations
            - total_evictions: Total number of template evictions
            - memory_usage_mb: Estimated memory usage in MB
        """
        total_lookups = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_lookups if total_lookups > 0 else 0.0

        # Estimate memory usage
        memory_mb = 0.0
        for template_id, blocks in self.template_blocks.items():
            template = self.registry.get_template(template_id)
            if template:
                # Rough estimate: 2 blocks per token (K+V) * block_size * dtype_size * num_heads * head_dim
                # This is a simplified estimate
                num_tokens = len(template.prompt_token_ids)
                memory_mb += num_tokens * 2 * get_dtype_size() * 4 / (1024 * 1024)  # Rough estimate

        return {
            "num_cached_templates": len(self.template_blocks),
            "max_cached_templates": self.max_cached_templates,
            "cached_template_ids": list(self.template_blocks.keys()),
            "cache_hit_rate": hit_rate,
            "total_registrations": self.total_registrations,
            "total_evictions": self.total_evictions,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "memory_usage_mb": memory_mb,
        }

    def clear_cache(self) -> None:
        """Clear all cached templates.

        This is useful for testing or manual cache management.
        """
        template_ids = list(self.template_blocks.keys())

        for template_id in template_ids:
            self.remove_template(template_id)

        logger.info(f"Cleared all template caches ({len(template_ids)} templates)")

    def get_template_info(self, template_id: str) -> Optional[dict]:
        """Get detailed information about a cached template.

        Args:
            template_id: The template ID

        Returns:
            Dictionary with template info or None if not found
        """
        blocks = self.template_blocks.get(template_id)
        if blocks is None:
            return None

        template = self.registry.get_template(template_id)
        if template is None:
            return None

        return {
            "template_id": template_id,
            "num_blocks": len(blocks),
            "num_tokens": len(template.prompt_token_ids),
            "num_apis": len(template.api_descriptions),
            "is_cached": template.is_cached,
            "last_access": self.last_access.get(template_id),
            "version": template.version,
        }
