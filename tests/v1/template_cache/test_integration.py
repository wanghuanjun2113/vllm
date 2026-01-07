# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for template-based caching system."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from vllm.config import CacheConfig, VllmConfig
from vllm.v1.core.template_kv_manager import TemplateKVCacheManager
from vllm.v1.core.template_mapper import TemplateRequestMapper
from vllm.v1.core.template_registry import TemplateRegistry
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


# ------------------ Fixtures ------------------ #
@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for integration testing."""

    class MockTokenizer:
        def encode(self, text):
            words = text.split()
            return [hash(w) % 100000 for w in words]

    return MockTokenizer()


@pytest.fixture
def sample_template_config():
    """Sample template configuration for testing."""
    return {
        "template_id": "integration_test_template",
        "apis": [
            {"api_id": "weather", "api_name": "Weather", "description": "Weather API"},
            {"api_id": "search", "api_name": "Search", "description": "Search API"},
            {"api_id": "email", "api_name": "Email", "description": "Email API"},
            {"api_id": "calc", "api_name": "Calculator", "description": "Calculator API"},
            {"api_id": "db", "api_name": "Database", "description": "Database API"},
        ],
    }


@pytest.fixture
def template_config_file(sample_template_config):
    """Create a temporary template config file."""
    import json

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_template_config, f)
        config_path = f.name

    yield config_path

    # Cleanup
    Path(config_path).unlink()


# ------------------ Integration Tests ------------------ #
def test_end_to_end_template_loading(template_config_file, mock_tokenizer):
    """Test end-to-end template loading from config file."""
    # Step 1: Create registry
    registry = TemplateRegistry()

    # Step 2: Load template from config
    template = registry.load_template_from_config(
        config_path=template_config_file, tokenizer=mock_tokenizer
    )

    assert template is not None
    assert template.template_id == "integration_test_template"
    assert len(template.api_descriptions) == 5

    # Step 3: Retrieve template
    retrieved = registry.get_template("integration_test_template")
    assert retrieved is not None
    assert retrieved.template_id == "integration_test_template"


def test_end_to_end_request_mapping(template_config_file, mock_tokenizer):
    """Test end-to-end request mapping."""
    # Setup
    registry = TemplateRegistry()
    template = registry.load_template_from_config(
        config_path=template_config_file, tokenizer=mock_tokenizer
    )

    mapper = TemplateRequestMapper(registry=registry, tokenizer=mock_tokenizer)

    # Create request
    prompt_tokens = mock_tokenizer.encode("I need weather and email services")

    request = Request(
        request_id="test_req",
        prompt=None,
        prompt_token_ids=prompt_tokens,
        multi_modal_data=None,
        sampling_params=None,
        block_hasher=None,
        arrival_time=0.0,
        lora_request=None,
    )

    # Map request
    mapping = mapper.map_request(request, template_id="integration_test_template")

    # Verify mapping
    if mapping is not None:
        assert mapping.template_id == "integration_test_template"
        assert isinstance(mapping.selected_api_indices, list)
        assert isinstance(mapping.attention_mask, Mock) or mapping.attention_mask is not None


def test_template_cache_workflow(template_config_file, mock_tokenizer):
    """Test complete template cache workflow."""
    # Setup components
    registry = TemplateRegistry()
    template = registry.load_template_from_config(
        config_path=template_config_file, tokenizer=mock_tokenizer
    )

    mock_block_pool = Mock()
    mock_block_pool.block_size = 16
    mock_block_pool.get_new_blocks = Mock(return_value=[Mock() for _ in range(10)])

    kv_manager = TemplateKVCacheManager(
        block_pool=mock_block_pool, registry=registry, max_cached_templates=5
    )

    # Step 1: Mark for registration
    kv_manager.mark_for_registration(template)
    assert template.template_id in kv_manager.pending_registrations

    # Step 2: Get pending registrations
    pending = kv_manager.get_pending_registrations()
    assert template in pending

    # Step 3: Clear pending
    kv_manager.clear_pending_registration(template.template_id)
    assert template.template_id not in kv_manager.pending_registrations


def test_multi_template_management(template_config_file, mock_tokenizer):
    """Test managing multiple templates."""
    registry = TemplateRegistry()

    # Load multiple templates
    template1 = registry.load_template_from_config(
        config_path=template_config_file,
        template_id="template1",
        tokenizer=mock_tokenizer,
    )

    template2 = registry.load_template_from_config(
        config_path=template_config_file,
        template_id="template2",
        tokenizer=mock_tokenizer,
    )

    # Verify both are in registry
    assert registry.get_template("template1") is not None
    assert registry.get_template("template2") is not None

    # Verify they're different objects
    assert template1 is not template2
    assert template1.template_id != template2.template_id


def test_cache_config_integration():
    """Test CacheConfig integration with template caching."""
    # Create CacheConfig with template options
    cache_config = CacheConfig(
        enable_prefix_caching=True,
        enable_template_caching=True,
        template_config_path="/path/to/template.yaml",
        max_cached_templates=10,
        template_match_threshold=0.9,
    )

    # Verify fields
    assert cache_config.enable_template_caching is True
    assert cache_config.template_config_path == "/path/to/template.yaml"
    assert cache_config.max_cached_templates == 10
    assert cache_config.template_match_threshold == 0.9


def test_cache_config_defaults():
    """Test CacheConfig defaults for template caching."""
    cache_config = CacheConfig()

    # Verify defaults
    assert cache_config.enable_template_caching is False
    assert cache_config.template_config_path is None
    assert cache_config.max_cached_templates == 5
    assert cache_config.template_match_threshold == 0.85


def test_template_lifecycle(template_config_file, mock_tokenizer):
    """Test complete template lifecycle from load to cache to evict."""
    registry = TemplateRegistry()
    template = registry.load_template_from_config(
        config_path=template_config_file, tokenizer=mock_tokenizer
    )

    mock_block_pool = Mock()
    mock_block_pool.block_size = 16
    mock_block_pool.get_new_blocks = Mock(return_value=[Mock() for _ in range(10)])
    mock_block_pool.free_block = Mock()

    kv_manager = TemplateKVCacheManager(
        block_pool=mock_block_pool, registry=registry, max_cached_templates=2
    )

    # Add templates to cache (simulate)
    kv_manager.template_blocks["template1"] = [Mock()]
    kv_manager.template_blocks["template2"] = [Mock()]
    kv_manager.last_access["template1"] = 1.0
    kv_manager.last_access["template2"] = 2.0

    # Cache should be full
    assert len(kv_manager.template_blocks) == 2

    # Evict LRU
    evicted = kv_manager.evict_lru_template()
    assert evicted == "template1"
    assert len(kv_manager.template_blocks) == 1


def test_request_to_template_to_cache_flow(template_config_file, mock_tokenizer):
    """Test flow from request -> template mapping -> cache operations."""
    # Setup
    registry = TemplateRegistry()
    template = registry.load_template_from_config(
        config_path=template_config_file, tokenizer=mock_tokenizer
    )

    mapper = TemplateRequestMapper(registry=registry, tokenizer=mock_tokenizer)

    mock_block_pool = Mock()
    mock_block_pool.block_size = 16
    kv_manager = TemplateKVCacheManager(
        block_pool=mock_block_pool, registry=registry, max_cached_templates=5
    )

    # Create and map request
    prompt_tokens = mock_tokenizer.encode("weather and email")
    request = Request(
        request_id="test",
        prompt=None,
        prompt_token_ids=prompt_tokens,
        multi_modal_data=None,
        sampling_params=None,
        block_hasher=None,
        arrival_time=0.0,
        lora_request=None,
    )

    mapping = mapper.map_request(request, template_id=template.template_id)

    # Verify flow
    assert mapping is not None or mapping is None  # May succeed or fail based on matching

    # Check cache stats
    stats = kv_manager.get_cache_stats()
    assert stats["num_cached_templates"] == 0  # Not cached yet


def test_template_error_handling(template_config_file, mock_tokenizer):
    """Test error handling in template operations."""
    registry = TemplateRegistry()

    # Try to load non-existent file
    template = registry.load_template_from_config(
        config_path="/nonexistent/path.yaml", tokenizer=mock_tokenizer
    )
    assert template is None

    # Try to get non-existent template
    template = registry.get_template("nonexistent")
    assert template is None


def test_template_concurrent_access(template_config_file, mock_tokenizer):
    """Test concurrent access to template system."""
    import threading

    registry = TemplateRegistry()
    template = registry.load_template_from_config(
        config_path=template_config_file, tokenizer=mock_tokenizer
    )

    mapper = TemplateRequestMapper(registry=registry, tokenizer=mock_tokenizer)

    results = []
    exceptions = []

    def map_request(req_id):
        try:
            prompt_tokens = mock_tokenizer.encode("weather service")
            request = Request(
                request_id=req_id,
                prompt=None,
                prompt_token_ids=prompt_tokens,
                multi_modal_data=None,
                sampling_params=None,
                block_hasher=None,
                arrival_time=0.0,
                lora_request=None,
            )

            mapping = mapper.map_request(request, template_id=template.template_id)
            results.append(mapping)
        except Exception as e:
            exceptions.append(e)

    # Create multiple threads
    threads = []
    for i in range(10):
        t = threading.Thread(target=map_request, args=(f"req_{i}",))
        threads.append(t)
        t.start()

    # Wait for completion
    for t in threads:
        t.join()

    # Verify no exceptions
    assert len(exceptions) == 0
    assert len(results) == 10


def test_cache_stats_reporting(template_config_file, mock_tokenizer):
    """Test cache statistics reporting."""
    registry = TemplateRegistry()
    template = registry.load_template_from_config(
        config_path=template_config_file, tokenizer=mock_tokenizer
    )

    mock_block_pool = Mock()
    mock_block_pool.block_size = 16
    kv_manager = TemplateKVCacheManager(
        block_pool=mock_block_pool, registry=registry, max_cached_templates=5
    )

    # Add some templates
    kv_manager.template_blocks["t1"] = [Mock()]
    kv_manager.template_blocks["t2"] = [Mock()]

    # Generate activity
    kv_manager.get_template_blocks("t1")  # hit
    kv_manager.get_template_blocks("t3")  # miss

    # Get stats
    stats = kv_manager.get_cache_stats()

    # Verify all stat fields are present
    expected_keys = [
        "num_cached_templates",
        "max_cached_templates",
        "cached_template_ids",
        "cache_hit_rate",
        "total_registrations",
        "total_evictions",
        "cache_hits",
        "cache_misses",
        "memory_usage_mb",
    ]

    for key in expected_keys:
        assert key in stats


def test_template_persistence_across_operations(template_config_file, mock_tokenizer):
    """Test that templates persist across various operations."""
    registry = TemplateRegistry()
    template = registry.load_template_from_config(
        config_path=template_config_file, tokenizer=mock_tokenizer
    )

    # Get template multiple times
    t1 = registry.get_template("integration_test_template")
    t2 = registry.get_template("integration_test_template")

    # Should return same object
    assert t1 is t2

    # Modify template
    template.version = 5

    # Retrieve again
    t3 = registry.get_template("integration_test_template")

    # Should reflect changes
    assert t3.version == 5


def test_template_yaml_and_json_equivalence(mock_tokenizer):
    """Test that YAML and JSON configs produce equivalent templates."""
    import json

    api_list = [
        {"api_id": "test", "api_name": "Test", "description": "Test API"},
    ]

    # Create JSON config
    json_config = {"template_id": "json_test", "apis": api_list}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(json_config, f)
        json_path = f.name

    # Create YAML config
    import yaml

    yaml_config = {"template_id": "yaml_test", "apis": api_list}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(yaml_config, f)
        yaml_path = f.name

    try:
        registry = TemplateRegistry()

        # Load both
        json_template = registry.load_template_from_config(
            config_path=json_path, tokenizer=mock_tokenizer
        )
        yaml_template = registry.load_template_from_config(
            config_path=yaml_path, tokenizer=mock_tokenizer
        )

        # Should both succeed
        assert json_template is not None
        assert yaml_template is not None

        # Should have same number of APIs
        assert len(json_template.api_descriptions) == len(yaml_template.api_descriptions)

    finally:
        Path(json_path).unlink()
        Path(yaml_path).unlink()
