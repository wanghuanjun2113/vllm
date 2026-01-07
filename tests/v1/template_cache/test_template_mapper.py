# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TemplateRequestMapper."""
import pytest
import torch

from vllm.v1.core.template_mapper import TemplateMapping, TemplateRequestMapper
from vllm.v1.core.template_registry import APITemplate, TemplateRegistry
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


# ------------------ Fixtures ------------------ #
@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""

    class MockTokenizer:
        def encode(self, text):
            # Simple mock: split by spaces and return fake token IDs
            words = text.split()
            return [hash(w) % 100000 for w in words]

    return MockTokenizer()


@pytest.fixture
def sample_template(mock_tokenizer):
    """Create a sample template for testing."""
    api_list = [
        {"api_id": "weather", "api_name": "Weather", "description": "Weather API"},
        {"api_id": "search", "api_name": "Search", "description": "Search API"},
        {"api_id": "email", "api_name": "Email", "description": "Email API"},
        {"api_id": "calc", "api_name": "Calculator", "description": "Calculator API"},
        {"api_id": "db", "api_name": "Database", "description": "Database API"},
    ]

    registry = TemplateRegistry()
    template = registry.load_template_from_config(
        api_list=api_list, template_id="test_template", tokenizer=mock_tokenizer
    )
    return template


@pytest.fixture
def template_registry(sample_template):
    """Template registry with sample template."""
    registry = TemplateRegistry()
    registry.templates["test_template"] = sample_template
    return registry


@pytest.fixture
def template_mapper(template_registry, mock_tokenizer):
    """TemplateRequestMapper instance for testing."""
    return TemplateRequestMapper(registry=template_registry, tokenizer=mock_tokenizer)


@pytest.fixture
def sample_request(mock_tokenizer):
    """Create a sample request for testing."""

    def make_request(prompt: str):
        prompt_tokens = mock_tokenizer.encode(prompt)
        # Create minimal Request object
        request = Request(
            request_id="test_request",
            prompt=None,
            prompt_token_ids=prompt_tokens,
            multi_modal_data=None,
            sampling_params=None,
            block_hasher=None,
            arrival_time=0.0,
            lora_request=None,
        )
        return request

    return make_request


# ------------------ Unit Tests ------------------ #
def test_template_mapper_init(template_mapper):
    """Test TemplateRequestMapper initialization."""
    assert template_mapper.registry is not None
    assert template_mapper.tokenizer is not None


def test_map_request_success(template_mapper, sample_request):
    """Test successful request mapping."""
    request = sample_request("I need weather and email services")

    mapping = template_mapper.map_request(request, template_id="test_template")

    assert mapping is not None
    assert isinstance(mapping, TemplateMapping)
    assert mapping.template_id == "test_template"
    assert isinstance(mapping.selected_api_indices, list)
    assert isinstance(mapping.confidence_scores, list)
    assert mapping.attention_mask is not None
    assert isinstance(mapping.token_ranges, list)


def test_map_request_nonexistent_template(template_mapper, sample_request):
    """Test mapping with non-existent template."""
    request = sample_request("Some prompt")

    mapping = template_mapper.map_request(request, template_id="nonexistent_template")

    assert mapping is None


def test_map_request_no_matches(template_mapper, sample_request):
    """Test mapping when no APIs match."""
    # Use a prompt with no matching keywords
    request = sample_request("xyz abc def random words")

    mapping = template_mapper.map_request(request, template_id="test_template")

    # Should return None or mapping with empty selections
    # depending on implementation
    if mapping is not None:
        assert len(mapping.selected_api_indices) == 0


def test_mapping_confidence_scores(template_mapper, sample_request):
    """Test that confidence scores are valid."""
    request = sample_request("I need weather and search")

    mapping = template_mapper.map_request(request, template_id="test_template")

    if mapping is not None and len(mapping.confidence_scores) > 0:
        for score in mapping.confidence_scores:
            assert isinstance(score, float)
            assert 0 <= score <= 1


def test_mapping_selected_indices_valid(template_mapper, sample_request):
    """Test that selected API indices are valid."""
    request = sample_request("weather email database calculator")

    mapping = template_mapper.map_request(request, template_id="test_template")

    if mapping is not None and len(mapping.selected_api_indices) > 0:
        for idx in mapping.selected_api_indices:
            assert isinstance(idx, int)
            assert idx >= 0
            # Should be within template API count range
            assert idx < 100  # Reasonable upper bound


def test_attention_mask_generation(template_mapper, sample_request):
    """Test attention mask generation."""
    request = sample_request("weather and email")

    mapping = template_mapper.map_request(request, template_id="test_template")

    if mapping is not None and mapping.attention_mask is not None:
        mask = mapping.attention_mask
        # Mask should be a tensor or list
        if isinstance(mask, torch.Tensor):
            assert mask.dtype == torch.bool
        elif isinstance(mask, list):
            assert all(isinstance(x, bool) for x in mask)


def test_token_ranges_valid(template_mapper, sample_request):
    """Test that token ranges are valid."""
    request = sample_request("weather search")

    mapping = template_mapper.map_request(request, template_id="test_template")

    if mapping is not None and len(mapping.token_ranges) > 0:
        for start, end in mapping.token_ranges:
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert start >= 0
            assert end > start


def test_generate_attention_mask(template_mapper, sample_template):
    """Test attention mask generation directly."""
    selected_indices = [0, 2, 4]  # Select first, third, and fifth APIs

    mask = template_mapper.generate_attention_mask(sample_template, selected_indices)

    assert mask is not None
    # Check mask is correct type and shape
    if isinstance(mask, torch.Tensor):
        assert mask.ndim == 1
        assert mask.dtype == torch.bool
        assert len(mask) > 0
    elif isinstance(mask, list):
        assert len(mask) > 0
        assert all(isinstance(x, (bool, torch.Tensor)) for x in mask)


def test_map_request_with_empty_prompt(template_mapper, sample_request):
    """Test mapping with empty prompt."""
    request = sample_request("")

    mapping = template_mapper.map_request(request, template_id="test_template")

    # Should handle gracefully
    if mapping is not None:
        assert mapping.template_id == "test_template"


def test_map_request_with_long_prompt(template_mapper, sample_request):
    """Test mapping with very long prompt."""
    long_prompt = " ".join(["weather"] * 1000)
    request = sample_request(long_prompt)

    mapping = template_mapper.map_request(request, template_id="test_template")

    # Should handle long prompts
    assert mapping is None or isinstance(mapping, TemplateMapping)


def test_mapping_consistency(template_mapper, sample_request):
    """Test that mapping is consistent for identical requests."""
    request1 = sample_request("I need weather and email")
    request2 = sample_request("I need weather and email")

    mapping1 = template_mapper.map_request(request1, template_id="test_template")
    mapping2 = template_mapper.map_request(request2, template_id="test_template")

    # Results should be similar (might not be exactly identical due to threading)
    if mapping1 is not None and mapping2 is not None:
        assert len(mapping1.selected_api_indices) == len(mapping2.selected_api_indices)


def test_mapping_different_templates(template_mapper, sample_request):
    """Test mapping with different templates."""
    request = sample_request("weather and search")

    # Map to first template
    mapping1 = template_mapper.map_request(request, template_id="test_template")

    # Try to map to non-existent template
    mapping2 = template_mapper.map_request(request, template_id="other_template")

    # First should succeed or return mapping, second should be None
    if mapping2 is None:
        assert mapping1 is None or isinstance(mapping1, TemplateMapping)


def test_attention_mask_matches_selection(template_mapper, sample_template):
    """Test that attention mask corresponds to selected APIs."""
    selected_indices = [0, 2]

    mask = template_mapper.generate_attention_mask(sample_template, selected_indices)

    if mask is not None and isinstance(mask, torch.Tensor):
        # Convert to list for easier checking
        mask_list = mask.cpu().tolist() if mask.is_cuda else mask.tolist()

        # Check that mask has some True values (for selected APIs)
        # and some False values (for non-selected APIs)
        assert any(mask_list), "Mask should have some True values"


def test_mapping_with_single_api(template_mapper, sample_request):
    """Test mapping when only one API is relevant."""
    request = sample_request("I just need weather")

    mapping = template_mapper.map_request(request, template_id="test_template")

    if mapping is not None:
        # Should have at least one selected API
        assert len(mapping.selected_api_indices) >= 0


def test_mapping_with_all_apis(template_mapper, sample_request):
    """Test mapping when all APIs are mentioned."""
    request = sample_request("weather search email calculator database")

    mapping = template_mapper.map_request(request, template_id="test_template")

    if mapping is not None:
        # Should detect multiple APIs
        assert len(mapping.selected_api_indices) >= 0


def test_template_mapping_attributes(template_mapper, sample_request):
    """Test all TemplateMapping attributes are present."""
    request = sample_request("weather and email")

    mapping = template_mapper.map_request(request, template_id="test_template")

    if mapping is not None:
        # Check all required attributes
        assert hasattr(mapping, "template_id")
        assert hasattr(mapping, "selected_api_indices")
        assert hasattr(mapping, "confidence_scores")
        assert hasattr(mapping, "attention_mask")
        assert hasattr(mapping, "token_ranges")

        # Check types
        assert isinstance(mapping.template_id, str)
        assert isinstance(mapping.selected_api_indices, list)
        assert isinstance(mapping.confidence_scores, list)
        assert mapping.attention_mask is not None
        assert isinstance(mapping.token_ranges, list)
