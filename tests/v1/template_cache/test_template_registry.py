# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TemplateRegistry."""
import tempfile
from pathlib import Path

import pytest

from vllm.v1.core.template_registry import APIDescription, APITemplate, TemplateRegistry

pytestmark = pytest.mark.cpu_test


# ------------------ Fixtures ------------------ #
@pytest.fixture
def sample_api_list():
    """Sample API list for testing."""
    return [
        {
            "api_id": "weather",
            "api_name": "Weather Service",
            "description": "Provides current weather data for any location worldwide.",
        },
        {
            "api_id": "search",
            "api_name": "Search Service",
            "description": "Searches company database for records and documents.",
        },
        {
            "api_id": "calculator",
            "api_name": "Calculator Service",
            "description": "Performs mathematical calculations and conversions.",
        },
        {
            "api_id": "email",
            "api_name": "Email Service",
            "description": "Sends and manages email communications.",
        },
        {
            "api_id": "database",
            "api_name": "Database Service",
            "description": "Manages database connections and queries.",
        },
    ]


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
def template_registry():
    """TemplateRegistry instance for testing."""
    return TemplateRegistry()


# ------------------ Unit Tests ------------------ #
def test_template_registry_init(template_registry):
    """Test TemplateRegistry initialization."""
    assert template_registry.templates == {}
    assert template_registry.lock is not None
    assert template_registry._default_template_source is None


def test_load_template_from_api_list(template_registry, sample_api_list, mock_tokenizer):
    """Test loading template from API list."""
    template = template_registry.load_template_from_config(
        api_list=sample_api_list,
        template_id="test_template",
        tokenizer=mock_tokenizer,
    )

    assert template is not None
    assert template.template_id == "test_template"
    assert len(template.api_descriptions) == 5
    assert template.prompt_token_ids is not None
    assert len(template.prompt_token_ids) > 0
    assert template.is_cached is False
    assert template.version == 0


def test_load_template_from_yaml(template_registry, mock_tokenizer):
    """Test loading template from YAML file."""
    yaml_content = """
template_id: yaml_test_template
apis:
  - api_id: api1
    api_name: API One
    description: First API description
  - api_id: api2
    api_name: API Two
    description: Second API description
  - api_id: api3
    api_name: API Three
    description: Third API description
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name

    try:
        template = template_registry.load_template_from_config(
            config_path=yaml_path,
            tokenizer=mock_tokenizer,
        )

        assert template is not None
        assert template.template_id == "yaml_test_template"
        assert len(template.api_descriptions) == 3
        assert template.api_descriptions[0].api_id == "api1"
        assert template.api_descriptions[1].api_id == "api2"
        assert template.api_descriptions[2].api_id == "api3"
    finally:
        Path(yaml_path).unlink()


def test_load_template_from_json(template_registry, mock_tokenizer):
    """Test loading template from JSON file."""
    import json

    json_content = {
        "template_id": "json_test_template",
        "apis": [
            {
                "api_id": "json_api1",
                "api_name": "JSON API One",
                "description": "First JSON API",
            },
            {
                "api_id": "json_api2",
                "api_name": "JSON API Two",
                "description": "Second JSON API",
            },
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(json_content, f)
        json_path = f.name

    try:
        template = template_registry.load_template_from_config(
            config_path=json_path,
            tokenizer=mock_tokenizer,
        )

        assert template is not None
        assert template.template_id == "json_test_template"
        assert len(template.api_descriptions) == 2
    finally:
        Path(json_path).unlink()


def test_get_template(template_registry, sample_api_list, mock_tokenizer):
    """Test retrieving a template from registry."""
    # Load template
    template = template_registry.load_template_from_config(
        api_list=sample_api_list,
        template_id="get_test",
        tokenizer=mock_tokenizer,
    )

    # Retrieve it
    retrieved = template_registry.get_template("get_test")

    assert retrieved is not None
    assert retrieved.template_id == "get_test"
    assert retrieved is template  # Should be same object


def test_get_nonexistent_template(template_registry):
    """Test retrieving a non-existent template."""
    retrieved = template_registry.get_template("nonexistent")
    assert retrieved is None


def test_match_apis_from_prompt(template_registry, sample_api_list, mock_tokenizer):
    """Test API matching from prompt."""
    # Load template
    template = template_registry.load_template_from_config(
        api_list=sample_api_list,
        template_id="match_test",
        tokenizer=mock_tokenizer,
    )

    # Create a prompt that mentions some APIs
    prompt_tokens = mock_tokenizer.encode(
        "I need to check the weather and search the database"
    )

    # Match APIs
    matches = template_registry.match_apis_from_prompt(
        prompt_token_ids=prompt_tokens,
        template=template,
        threshold=0.5,
        max_matches=3,
    )

    # Should return some matches
    assert isinstance(matches, list)
    assert len(matches) <= 3

    # Each match should be a tuple of (api_index, confidence)
    if len(matches) > 0:
        api_idx, confidence = matches[0]
        assert isinstance(api_idx, int)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
        assert 0 <= api_idx < len(template.api_descriptions)


def test_match_apis_with_high_threshold(template_registry, sample_api_list, mock_tokenizer):
    """Test API matching with high threshold returns fewer matches."""
    template = template_registry.load_template_from_config(
        api_list=sample_api_list,
        template_id="high_threshold_test",
        tokenizer=mock_tokenizer,
    )

    prompt_tokens = mock_tokenizer.encode("random unrelated text")

    # Use very high threshold
    matches = template_registry.match_apis_from_prompt(
        prompt_token_ids=prompt_tokens,
        template=template,
        threshold=0.99,
        max_matches=5,
    )

    # Should return very few or no matches
    assert len(matches) <= 1


def test_api_description_structure(template_registry, sample_api_list, mock_tokenizer):
    """Test APIDescription structure is correct."""
    template = template_registry.load_template_from_config(
        api_list=sample_api_list,
        template_id="structure_test",
        tokenizer=mock_tokenizer,
    )

    # Check first API description
    api = template.api_descriptions[0]
    assert isinstance(api, APIDescription)
    assert api.api_id == "weather"
    assert api.api_name == "Weather Service"
    assert len(api.description) > 0
    assert api.token_start >= 0
    assert api.token_end > api.token_start
    assert len(api.signature_tokens) > 0


def test_template_version_increment(template_registry, sample_api_list, mock_tokenizer):
    """Test template version handling."""
    # Load template
    template = template_registry.load_template_from_config(
        api_list=sample_api_list,
        template_id="version_test",
        tokenizer=mock_tokenizer,
    )

    assert template.version == 0

    # Increment version
    template.version += 1
    assert template.version == 1


def test_concurrent_template_loading(template_registry, sample_api_list, mock_tokenizer):
    """Test thread-safe template loading."""
    import threading

    results = []
    exceptions = []

    def load_template(template_id):
        try:
            template = template_registry.load_template_from_config(
                api_list=sample_api_list,
                template_id=template_id,
                tokenizer=mock_tokenizer,
            )
            results.append(template)
        except Exception as e:
            exceptions.append(e)

    # Create multiple threads
    threads = []
    for i in range(5):
        t = threading.Thread(target=load_template, args=(f"concurrent_{i}",))
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    # Check results
    assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"
    assert len(results) == 5

    # Verify all templates are in registry
    for i in range(5):
        template = template_registry.get_template(f"concurrent_{i}")
        assert template is not None
        assert template.template_id == f"concurrent_{i}"


def test_match_apis_respects_max_matches(template_registry, sample_api_list, mock_tokenizer):
    """Test that match_apis_from_prompt respects max_matches parameter."""
    template = template_registry.load_template_from_config(
        api_list=sample_api_list,
        template_id="max_matches_test",
        tokenizer=mock_tokenizer,
    )

    # Create prompt with keywords for multiple APIs
    prompt_tokens = mock_tokenizer.encode(
        "weather search calculator email database weather search"
    )

    # Request only 2 matches
    matches = template_registry.match_apis_from_prompt(
        prompt_token_ids=prompt_tokens,
        template=template,
        threshold=0.3,
        max_matches=2,
    )

    assert len(matches) <= 2


def test_template_prompt_construction(template_registry, sample_api_list, mock_tokenizer):
    """Test that template prompt is constructed correctly."""
    template = template_registry.load_template_from_config(
        api_list=sample_api_list,
        template_id="prompt_test",
        tokenizer=mock_tokenizer,
    )

    # Prompt should be constructed from API descriptions
    assert template.prompt_token_ids is not None
    assert len(template.prompt_token_ids) > 0

    # Token ranges should be non-overlapping
    for i, api in enumerate(template.api_descriptions):
        for j, other_api in enumerate(template.api_descriptions):
            if i != j:
                # Check no overlap
                assert not (
                    api.token_start < other_api.token_end
                    and api.token_end > other_api.token_start
                ), f"Token ranges overlap for {api.api_id} and {other_api.api_id}"
