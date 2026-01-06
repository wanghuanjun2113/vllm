# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Template Registry for API-based prefix caching.

This module provides a registry system for managing API templates that can be
used for dynamic prefix caching with attention masks.
"""

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import BlockHash

logger = init_logger(__name__)


@dataclass
class APIDescription:
    """Metadata for a single API in the template.

    Attributes:
        api_id: Unique identifier for this API (e.g., "weather_api")
        api_name: Human-readable display name (e.g., "Weather Service")
        description: Full text description of the API
        token_start: Start position (inclusive) in the template token sequence
        token_end: End position (exclusive) in the template token sequence
        signature_tokens: First N tokens for fast matching (default: 50 tokens)
        embedding: Optional embedding vector for semantic matching
    """
    api_id: str
    api_name: str
    description: str
    token_start: int
    token_end: int
    signature_tokens: list[int] = field(default_factory=list)
    embedding: Optional[torch.Tensor] = None

    def __post_init__(self):
        """Validate API description after initialization."""
        if self.token_start < 0:
            raise ValueError(f"token_start must be >= 0, got {self.token_start}")
        if self.token_end <= self.token_start:
            raise ValueError(
                f"token_end ({self.token_end}) must be > token_start ({self.token_start})"
            )
        if not self.api_id:
            raise ValueError("api_id cannot be empty")


@dataclass
class APITemplate:
    """Represents a registered multi-API template.

    A template contains a full prompt with multiple API descriptions.
    For example, a template might contain 50 API descriptions totaling ~20K tokens.

    Attributes:
        template_id: Unique identifier for this template (e.g., "company_apis_v1")
        prompt_token_ids: Full token sequence for the template (~20K tokens for 50 APIs)
        api_descriptions: List of API metadata, one per API in the template
        block_hashes: Pre-computed cache keys for each block (computed after caching)
        version: Version number for cache invalidation
        created_at: Timestamp when template was created
        is_cached: Whether KV cache has been computed for this template
    """
    template_id: str
    prompt_token_ids: list[int]
    api_descriptions: list[APIDescription]
    block_hashes: list["BlockHash"] = field(default_factory=list)
    version: int = 1
    created_at: float = field(default_factory=time.time)
    is_cached: bool = False

    def __post_init__(self):
        """Validate template after initialization."""
        if not self.template_id:
            raise ValueError("template_id cannot be empty")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids cannot be empty")
        if not self.api_descriptions:
            raise ValueError("api_descriptions cannot be empty")

        # Validate token ranges
        template_len = len(self.prompt_token_ids)
        for api in self.api_descriptions:
            if api.token_end > template_len:
                raise ValueError(
                    f"API {api.api_id} token range [{api.token_start}, {api.token_end}) "
                    f"exceeds template length {template_len}"
                )

    def get_api_by_id(self, api_id: str) -> Optional[APIDescription]:
        """Find an API by its ID."""
        for api in self.api_descriptions:
            if api.api_id == api_id:
                return api
        return None

    def get_api_by_index(self, index: int) -> Optional[APIDescription]:
        """Get an API by its index in the descriptions list."""
        if 0 <= index < len(self.api_descriptions):
            return self.api_descriptions[index]
        return None


class TemplateRegistry:
    """Global registry for API templates.

    This class manages the registration, storage, and retrieval of API templates.
    It is thread-safe and supports loading templates from configuration files.

    The registry maintains a dictionary of templates indexed by template_id.
    Templates can be loaded from YAML/JSON configuration files or provided
    programmatically as a list of API descriptions.

    Example usage:
        ```python
        registry = TemplateRegistry()
        template = registry.load_template_from_config(
            config_path="api_template.yaml",
            tokenizer=tokenizer
        )
        ```
    """

    def __init__(self):
        """Initialize an empty template registry."""
        self.templates: dict[str, APITemplate] = {}
        self.lock = threading.RLock()
        self._default_template_id: Optional[str] = None

    def load_template_from_config(
        self,
        config_path: Optional[str] = None,
        api_list: Optional[list[dict]] = None,
        template_id: str = "default",
        tokenizer: Optional[Any] = None,
    ) -> APITemplate:
        """Load a template from a configuration file or API list.

        Args:
            config_path: Path to YAML/JSON configuration file. If None, api_list must be provided.
            api_list: List of API descriptions. If None, config_path must be provided.
            template_id: Unique identifier for the template
            tokenizer: Tokenizer to encode API descriptions into tokens

        Returns:
            The loaded APITemplate object

        Raises:
            ValueError: If neither config_path nor api_list is provided
            FileNotFoundError: If config_path doesn't exist
            ValueError: If tokenizer is not provided when needed

        Config file format (YAML):
            ```yaml
            template_id: "company_apis_v1"
            apis:
              - api_id: "weather"
                api_name: "Weather Service"
                description: "Provides current weather data for any location..."
              - api_id: "search"
                api_name: "Search Service"
                description: "Searches the company database for records..."
            ```

        Config file format (JSON):
            ```json
            {
                "template_id": "company_apis_v1",
                "apis": [
                    {
                        "api_id": "weather",
                        "api_name": "Weather Service",
                        "description": "Provides current weather data..."
                    },
                    {
                        "api_id": "search",
                        "api_name": "Search Service",
                        "description": "Searches the company database..."
                    }
                ]
            }
            ```
        """
        if config_path is None and api_list is None:
            raise ValueError("Either config_path or api_list must be provided")

        if config_path is not None:
            # Load from file
            config_path_obj = Path(config_path)
            if not config_path_obj.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            if config_path_obj.suffix in ['.yaml', '.yml']:
                import yaml
                with open(config_path_obj, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            elif config_path_obj.suffix == '.json':
                with open(config_path_obj, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported config file format: {config_path_obj.suffix}. "
                    "Supported formats: .yaml, .yml, .json"
                )

            api_list = config.get('apis', [])
            template_id = config.get('template_id', template_id)

        if not api_list:
            raise ValueError("API list is empty")

        if tokenizer is None:
            raise ValueError("Tokenizer is required to encode API descriptions")

        # Build the full prompt
        full_prompt = self._build_full_prompt(api_list)

        # Encode to tokens
        prompt_token_ids = tokenizer.encode(full_prompt, add_special_tokens=False)

        # Extract API metadata with token ranges
        api_descriptions = self._extract_api_metadata(
            api_list, full_prompt, tokenizer, prompt_token_ids
        )

        # Create template
        template = APITemplate(
            template_id=template_id,
            prompt_token_ids=prompt_token_ids,
            api_descriptions=api_descriptions,
        )

        # Register
        with self.lock:
            self.templates[template_id] = template
            if self._default_template_id is None:
                self._default_template_id = template_id

        logger.info(
            f"Registered template '{template_id}' with {len(api_descriptions)} APIs "
            f"({len(prompt_token_ids)} tokens)"
        )

        return template

    def _build_full_prompt(self, api_list: list[dict]) -> str:
        """Build the full prompt by concatenating all API descriptions.

        Args:
            api_list: List of API description dictionaries

        Returns:
            Concatenated prompt string with all API descriptions
        """
        prompt_parts = []
        for i, api_desc in enumerate(api_list):
            api_id = api_desc.get('api_id', f'api_{i}')
            api_name = api_desc.get('api_name', f'API {i}')
            description = api_desc.get('description', '')

            # Format: "API {i}: {api_name}\n{description}\n\n"
            part = f"API {i}: {api_name}\n{description}\n\n"
            prompt_parts.append(part)

        return ''.join(prompt_parts)

    def _extract_api_metadata(
        self,
        api_list: list[dict],
        full_prompt: str,
        tokenizer: Any,
        prompt_token_ids: list[int],
        signature_length: int = 50,
    ) -> list[APIDescription]:
        """Extract metadata for each API description in the template.

        This method determines the token range for each API and extracts
        signature tokens for fast matching.

        Args:
            api_list: List of API description dictionaries
            full_prompt: The full concatenated prompt
            tokenizer: Tokenizer for encoding
            prompt_token_ids: Token IDs for the full prompt
            signature_length: Number of tokens to extract as signature

        Returns:
            List of APIDescription objects with token ranges and signatures
        """
        api_descriptions = []
        current_pos = 0

        for i, api_desc in enumerate(api_list):
            api_id = api_desc.get('api_id', f'api_{i}')
            api_name = api_desc.get('api_name', f'API {i}')
            description = api_desc.get('description', '')

            # Build the API-specific prompt segment
            api_prompt = f"API {i}: {api_name}\n{description}\n\n"

            # Find the position of this API's description in the full prompt
            start_pos = full_prompt.find(api_prompt, current_pos)
            if start_pos == -1:
                logger.warning(f"Could not find exact position for API {api_id}")
                start_pos = current_pos

            end_pos = start_pos + len(api_prompt)

            # Tokenize this API's description to find token range
            api_tokens = tokenizer.encode(api_prompt, add_special_tokens=False)
            token_start = len(tokenizer.encode(
                full_prompt[:start_pos], add_special_tokens=False
            ))
            token_end = token_start + len(api_tokens)

            # Extract signature tokens (first N tokens of this API)
            signature_tokens = api_tokens[:signature_length]

            api_metadata = APIDescription(
                api_id=api_id,
                api_name=api_name,
                description=description,
                token_start=token_start,
                token_end=token_end,
                signature_tokens=signature_tokens,
                embedding=None,  # Can be computed later if needed
            )
            api_descriptions.append(api_metadata)

            current_pos = end_pos

        return api_descriptions

    def get_template(self, template_id: str) -> Optional[APITemplate]:
        """Retrieve a registered template by ID.

        Args:
            template_id: The template identifier

        Returns:
            The APITemplate if found, None otherwise
        """
        with self.lock:
            return self.templates.get(template_id)

    def get_default_template(self) -> Optional[APITemplate]:
        """Get the default template (first registered template).

        Returns:
            The default APITemplate if any templates are registered, None otherwise
        """
        with self.lock:
            if self._default_template_id:
                return self.templates.get(self._default_template_id)
            return None

    def list_templates(self) -> list[str]:
        """List all registered template IDs.

        Returns:
            List of template IDs
        """
        with self.lock:
            return list(self.templates.keys())

    def match_apis_from_prompt(
        self,
        prompt_token_ids: list[int],
        template: APITemplate,
        threshold: float = 0.85,
        max_matches: int = 5,
    ) -> list[tuple[int, float]]:
        """Match APIs from a request prompt to the template APIs.

        This method analyzes the request prompt to identify which APIs from the
        template are being used. It uses token-level matching with confidence scoring.

        Args:
            prompt_token_ids: Token IDs from the request prompt
            template: The APITemplate to match against
            threshold: Minimum confidence score (0.0-1.0) for a match
            max_matches: Maximum number of APIs to return

        Returns:
            List of (api_index, confidence) tuples, sorted by confidence descending

        Algorithm:
            1. For each API in the template, compute signature match score
            2. Score = (matching signature tokens / total signature tokens)
            3. Return top N matches above threshold
        """
        matches = []

        for api_idx, api_desc in enumerate(template.api_descriptions):
            # Compute match score
            score = self._compute_signature_match_score(
                prompt_token_ids,
                api_desc.signature_tokens,
            )

            if score >= threshold:
                matches.append((api_idx, score))

        # Sort by confidence descending
        matches.sort(key=lambda x: x[1], reverse=True)

        # Return top N matches
        return matches[:max_matches]

    def _compute_signature_match_score(
        self,
        prompt_tokens: list[int],
        signature_tokens: list[int],
    ) -> float:
        """Compute the match score between prompt and API signature.

        Args:
            prompt_tokens: Token IDs from the request prompt
            signature_tokens: Signature tokens from an API description

        Returns:
            Match score between 0.0 and 1.0
        """
        if not signature_tokens:
            return 0.0

        # Count how many signature tokens appear in the prompt
        # (allowing for non-contiguous matches)
        prompt_set = set(prompt_tokens)
        signature_set = set(signature_tokens)

        intersection = len(prompt_set & signature_set)
        score = intersection / len(signature_set)

        return score

    def register_template(self, template: APITemplate) -> None:
        """Manually register a pre-built template.

        Args:
            template: The APITemplate to register
        """
        with self.lock:
            self.templates[template.template_id] = template
            if self._default_template_id is None:
                self._default_template_id = template.template_id

        logger.info(f"Manually registered template '{template.template_id}'")

    def remove_template(self, template_id: str) -> bool:
        """Remove a template from the registry.

        Args:
            template_id: The template to remove

        Returns:
            True if template was removed, False if it didn't exist
        """
        with self.lock:
            if template_id in self.templates:
                del self.templates[template_id]
                if self._default_template_id == template_id:
                    self._default_template_id = next(
                        iter(self.templates.keys()), None
                    )
                logger.info(f"Removed template '{template_id}'")
                return True
            return False
