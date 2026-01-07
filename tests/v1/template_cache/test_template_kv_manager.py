# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TemplateKVCacheManager."""
import time
from unittest.mock import MagicMock, Mock

import pytest

from vllm.v1.core.template_kv_manager import TemplateKVCacheManager
from vllm.v1.core.template_registry import APITemplate, TemplateRegistry

pytestmark = pytest.mark.cpu_test


# ------------------ Fixtures ------------------ #
@pytest.fixture
def mock_block_pool():
    """Mock BlockPool for testing."""
    block_pool = MagicMock()
    block_pool.block_size = 16

    # Mock get_new_blocks to return fake blocks
    block_pool.get_new_blocks = Mock(return_value=[Mock() for _ in range(10)])

    # Mock free_block
    block_pool.free_block = Mock()

    return block_pool


@pytest.fixture
def mock_registry():
    """Mock TemplateRegistry for testing."""
    registry = MagicMock(spec=TemplateRegistry)

    # Create mock template
    mock_template = Mock(spec=APITemplate)
    mock_template.template_id = "test_template"
    mock_template.prompt_token_ids = list(range(100))  # 100 tokens
    mock_template.api_descriptions = []
    mock_template.is_cached = False
    mock_template.version = 0

    # Mock get_template to return our mock template
    registry.get_template = Mock(return_value=mock_template)

    return registry


@pytest.fixture
def sample_template(mock_registry):
    """Create a real sample template for testing."""
    template = APITemplate(
        template_id="sample_template",
        prompt_token_ids=list(range(100)),
        api_descriptions=[],
        block_hashes=[],
        version=0,
        is_cached=False,
    )
    return template


@pytest.fixture
def template_kv_manager(mock_block_pool, mock_registry):
    """TemplateKVCacheManager instance for testing."""
    return TemplateKVCacheManager(
        block_pool=mock_block_pool, registry=mock_registry, max_cached_templates=3
    )


# ------------------ Unit Tests ------------------ #
def test_template_kv_manager_init(template_kv_manager):
    """Test TemplateKVCacheManager initialization."""
    assert template_kv_manager.block_pool is not None
    assert template_kv_manager.registry is not None
    assert template_kv_manager.max_cached_templates == 3
    assert template_kv_manager.template_blocks == {}
    assert template_kv_manager.last_access == {}
    assert template_kv_manager.total_registrations == 0
    assert template_kv_manager.total_evictions == 0
    assert template_kv_manager.cache_hits == 0
    assert template_kv_manager.cache_misses == 0


def test_template_kv_manager_custom_max_cached(mock_block_pool, mock_registry):
    """Test TemplateKVCacheManager with custom max_cached_templates."""
    manager = TemplateKVCacheManager(
        block_pool=mock_block_pool, registry=mock_registry, max_cached_templates=10
    )

    assert manager.max_cached_templates == 10


def test_get_template_blocks_miss(template_kv_manager):
    """Test get_template_blocks when template not cached."""
    blocks = template_kv_manager.get_template_blocks("nonexistent_template")

    assert blocks is None
    assert template_kv_manager.cache_misses == 1
    assert template_kv_manager.cache_hits == 0


def test_get_template_blocks_hit(template_kv_manager):
    """Test get_template_blocks when template is cached."""
    # Manually add a cached template
    mock_blocks = [Mock(), Mock(), Mock()]
    template_kv_manager.template_blocks["cached_template"] = mock_blocks

    blocks = template_kv_manager.get_template_blocks("cached_template")

    assert blocks is not None
    assert blocks == mock_blocks
    assert template_kv_manager.cache_hits == 1
    assert template_kv_manager.cache_misses == 0


def test_get_template_blocks_updates_lru(template_kv_manager):
    """Test that get_template_blocks updates LRU timestamp."""
    # Add cached template
    mock_blocks = [Mock()]
    template_kv_manager.template_blocks["lru_test"] = mock_blocks
    old_timestamp = template_kv_manager.last_access.get("lru_test", 0)

    # Wait a bit to ensure timestamp difference
    time.sleep(0.01)

    # Get blocks
    template_kv_manager.get_template_blocks("lru_test")

    # Check timestamp was updated
    new_timestamp = template_kv_manager.last_access["lru_test"]
    assert new_timestamp > old_timestamp


def test_mark_for_registration(template_kv_manager, sample_template):
    """Test marking template for pending registration."""
    template_kv_manager.mark_for_registration(sample_template)

    assert sample_template.template_id in template_kv_manager.pending_registrations
    assert template_kv_manager.pending_registrations[sample_template.template_id] == sample_template


def test_get_pending_registrations(template_kv_manager, sample_template):
    """Test getting pending registrations."""
    # Mark two templates
    template_kv_manager.mark_for_registration(sample_template)

    template2 = Mock(spec=APITemplate)
    template2.template_id = "template2"
    template_kv_manager.mark_for_registration(template2)

    # Get pending
    pending = template_kv_manager.get_pending_registrations()

    assert len(pending) == 2
    assert sample_template in pending
    assert template2 in pending


def test_clear_pending_registration(template_kv_manager, sample_template):
    """Test clearing pending registration."""
    template_kv_manager.mark_for_registration(sample_template)

    assert sample_template.template_id in template_kv_manager.pending_registrations

    # Clear it
    template_kv_manager.clear_pending_registration(sample_template.template_id)

    assert sample_template.template_id not in template_kv_manager.pending_registrations


def test_manual_register_template_not_cached(template_kv_manager, mock_registry):
    """Test manual template registration when not cached."""
    # Our mock template has is_cached=False
    mock_template = mock_registry.get_template("test_template")

    result = template_kv_manager.manual_register_template(
        template_id="test_template", prompt_token_ids=list(range(50))
    )

    # Should return False because model_executor is None
    assert result is False


def test_evict_lru_template(template_kv_manager):
    """Test LRU eviction of templates."""
    # Add two templates with different timestamps
    blocks1 = [Mock()]
    blocks2 = [Mock()]

    template_kv_manager.template_blocks["template1"] = blocks1
    template_kv_manager.template_blocks["template2"] = blocks2

    # Set different timestamps
    old_time = time.time() - 10
    template_kv_manager.last_access["template1"] = old_time
    template_kv_manager.last_access["template2"] = time.time()

    # Evict LRU (should be template1)
    evicted_id = template_kv_manager.evict_lru_template()

    assert evicted_id == "template1"
    assert "template1" not in template_kv_manager.template_blocks
    assert "template2" in template_kv_manager.template_blocks
    assert template_kv_manager.total_evictions == 1
    assert template_kv_manager.block_pool.free_block.called


def test_evict_lru_template_empty_cache(template_kv_manager):
    """Test eviction when cache is empty."""
    evicted_id = template_kv_manager.evict_lru_template()

    assert evicted_id is None


def test_remove_template(template_kv_manager):
    """Test manual template removal."""
    # Add a template
    blocks = [Mock(), Mock()]
    template_kv_manager.template_blocks["to_remove"] = blocks
    template_kv_manager.last_access["to_remove"] = time.time()

    # Remove it
    result = template_kv_manager.remove_template("to_remove")

    assert result is True
    assert "to_remove" not in template_kv_manager.template_blocks
    assert "to_remove" not in template_kv_manager.last_access
    assert template_kv_manager.block_pool.free_block.called


def test_remove_nonexistent_template(template_kv_manager):
    """Test removing a template that doesn't exist."""
    result = template_kv_manager.remove_template("nonexistent")

    assert result is False


def test_get_cache_stats_empty(template_kv_manager):
    """Test getting cache stats when empty."""
    stats = template_kv_manager.get_cache_stats()

    assert stats["num_cached_templates"] == 0
    assert stats["max_cached_templates"] == 3
    assert stats["cached_template_ids"] == []
    assert stats["cache_hit_rate"] == 0.0
    assert stats["total_registrations"] == 0
    assert stats["total_evictions"] == 0
    assert stats["cache_hits"] == 0
    assert stats["cache_misses"] == 0


def test_get_cache_stats_with_templates(template_kv_manager):
    """Test getting cache stats with cached templates."""
    # Add some templates
    template_kv_manager.template_blocks["t1"] = [Mock()]
    template_kv_manager.template_blocks["t2"] = [Mock()]

    # Generate some cache activity
    template_kv_manager.get_template_blocks("t1")  # hit
    template_kv_manager.get_template_blocks("t3")  # miss
    template_kv_manager.total_registrations = 2
    template_kv_manager.total_evictions = 1

    stats = template_kv_manager.get_cache_stats()

    assert stats["num_cached_templates"] == 2
    assert stats["max_cached_templates"] == 3
    assert set(stats["cached_template_ids"]) == {"t1", "t2"}
    assert stats["cache_hit_rate"] == 0.5  # 1 hit / 2 total
    assert stats["total_registrations"] == 2
    assert stats["total_evictions"] == 1
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 1


def test_clear_cache(template_kv_manager):
    """Test clearing all cached templates."""
    # Add multiple templates
    template_kv_manager.template_blocks["t1"] = [Mock()]
    template_kv_manager.template_blocks["t2"] = [Mock()]
    template_kv_manager.template_blocks["t3"] = [Mock()]

    # Clear all
    template_kv_manager.clear_cache()

    assert len(template_kv_manager.template_blocks) == 0
    assert len(template_kv_manager.last_access) == 0


def test_get_template_info(template_kv_manager, mock_registry):
    """Test getting info about a specific template."""
    # Add a template to cache
    blocks = [Mock(), Mock(), Mock()]
    template_kv_manager.template_blocks["test_template"] = blocks
    template_kv_manager.last_access["test_template"] = time.time()

    # Get info
    info = template_kv_manager.get_template_info("test_template")

    assert info is not None
    assert info["template_id"] == "test_template"
    assert info["num_blocks"] == 3
    assert info["is_cached"] is False  # From our mock
    assert info["version"] == 0
    assert "last_access" in info


def test_get_template_info_nonexistent(template_kv_manager):
    """Test getting info for non-existent template."""
    info = template_kv_manager.get_template_info("nonexistent")

    assert info is None


def test_cache_at_max_capacity(template_kv_manager):
    """Test behavior when cache reaches max capacity."""
    # Fill cache to max capacity
    for i in range(3):
        template_kv_manager.template_blocks[f"template_{i}"] = [Mock()]
        template_kv_manager.last_access[f"template_{i}"] = time.time()

    # Cache should be at max capacity
    assert len(template_kv_manager.template_blocks) == template_kv_manager.max_cached_templates


def test_calculate_num_blocks(template_kv_manager):
    """Test block calculation for templates."""
    template = Mock(spec=APITemplate)
    template.prompt_token_ids = list(range(100))  # 100 tokens
    template_kv_manager.block_pool.block_size = 16

    num_blocks = template_kv_manager._calculate_num_blocks(template)

    # 100 tokens / 16 block_size = 6.25 -> 7 blocks (ceiling division)
    assert num_blocks == 7


def test_thread_safe_operations(template_kv_manager, sample_template):
    """Test that operations are thread-safe."""
    import threading

    results = []
    exceptions = []

    def access_template(template_id):
        try:
            blocks = template_kv_manager.get_template_blocks(template_id)
            results.append(blocks)
        except Exception as e:
            exceptions.append(e)

    # Create multiple threads accessing the cache
    threads = []
    for i in range(5):
        t = threading.Thread(target=access_template, args=(f"template_{i}",))
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    # Should not have any exceptions
    assert len(exceptions) == 0
    assert len(results) == 5


def test_cache_hit_rate_calculation(template_kv_manager):
    """Test cache hit rate is calculated correctly."""
    # Add one template
    template_kv_manager.template_blocks["test"] = [Mock()]

    # Generate hits and misses
    template_kv_manager.get_template_blocks("test")  # hit
    template_kv_manager.get_template_blocks("test")  # hit
    template_kv_manager.get_template_blocks("missing")  # miss

    stats = template_kv_manager.get_cache_stats()

    # 2 hits / 3 total = 0.666...
    assert abs(stats["cache_hit_rate"] - 2 / 3) < 0.001
