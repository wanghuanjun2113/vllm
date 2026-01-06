# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Template Request Mapper for API-based prefix caching.

This module provides the mapping logic between user requests and cached templates.
It handles API matching, validation, and attention mask generation.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.template_registry import APITemplate
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class TemplateMapping:
    """Result of mapping a request to a template.

    Attributes:
        template_id: The template ID this request maps to
        selected_api_indices: Indices of selected APIs in the template (e.g., [5, 12, 23, 34, 45])
        confidence_scores: Match confidence for each selected API (0.0-1.0)
        attention_mask: Binary mask tensor of shape (template_length,)
                        True = attend to this token, False = ignore
        token_ranges: List of (token_start, token_end) tuples for each selected API
        num_selected_apis: Number of APIs selected (should be 5 for the target use case)
    """
    template_id: str
    selected_api_indices: list[int] = field(default_factory=list)
    confidence_scores: list[float] = field(default_factory=list)
    attention_mask: Optional[torch.Tensor] = None
    token_ranges: list[tuple[int, int]] = field(default_factory=list)

    @property
    def num_selected_apis(self) -> int:
        """Number of selected APIs."""
        return len(self.selected_api_indices)

    def validate(self) -> bool:
        """Validate the mapping.

        Returns:
            True if the mapping is valid, False otherwise
        """
        if not self.template_id:
            logger.error("Template ID is empty")
            return False

        if len(self.selected_api_indices) != len(self.confidence_scores):
            logger.error(
                f"Mismatch between selected APIs ({len(self.selected_api_indices)}) "
                f"and confidence scores ({len(self.confidence_scores)})"
            )
            return False

        if len(self.selected_api_indices) != len(self.token_ranges):
            logger.error(
                f"Mismatch between selected APIs ({len(self.selected_api_indices)}) "
                f"and token ranges ({len(self.token_ranges)})"
            )
            return False

        if self.attention_mask is None:
            logger.error("Attention mask is None")
            return False

        return True


class TemplateRequestMapper:
    """Maps incoming user requests to cached templates.

    This class is responsible for:
    1. Matching APIs from the request prompt to template APIs
    2. Validating match quality
    3. Generating attention masks for selected APIs
    4. Handling errors and fallback scenarios

    The mapper uses a two-stage matching process:
    - Fast token-based matching (signature tokens)
    - Optional semantic matching (embeddings) for improved accuracy
    """

    def __init__(
        self,
        registry: "TemplateRegistry",
        tokenizer,
        match_threshold: float = 0.85,
        min_required_apis: int = 5,
        max_allowed_apis: int = 10,
    ):
        """Initialize the template request mapper.

        Args:
            registry: The TemplateRegistry containing registered templates
            tokenizer: Tokenizer for encoding/decoding
            match_threshold: Minimum confidence score for API matching (0.0-1.0)
            min_required_apis: Minimum number of APIs that must match
            max_allowed_apis: Maximum number of APIs allowed per request
        """
        self.registry = registry
        self.tokenizer = tokenizer
        self.match_threshold = match_threshold
        self.min_required_apis = min_required_apis
        self.max_allowed_apis = max_allowed_apis

    def map_request(
        self,
        request: "Request",
        template_id: Optional[str] = None,
    ) -> Optional[TemplateMapping]:
        """Map a user request to a template.

        This is the main entry point for request-to-template mapping.
        It performs the following steps:
        1. Retrieve the template (use default if template_id is None)
        2. Match APIs from the request prompt
        3. Validate match quality
        4. Generate attention mask
        5. Return mapping or None if mapping fails

        Args:
            request: The user request
            template_id: Template ID to use (None for default template)

        Returns:
            TemplateMapping if successful, None if mapping fails
        """
        # Get template
        if template_id is None:
            template = self.registry.get_default_template()
            if template is None:
                logger.error("No default template found in registry")
                return None
        else:
            template = self.registry.get_template(template_id)
            if template is None:
                logger.error(f"Template '{template_id}' not found in registry")
                return None

        # Match APIs from request
        matches = self.registry.match_apis_from_prompt(
            prompt_token_ids=request.prompt_token_ids,
            template=template,
            threshold=self.match_threshold,
            max_matches=self.max_allowed_apis,
        )

        if not matches:
            logger.warning(
                f"No APIs matched from request '{request.request_id}' "
                f"(threshold={self.match_threshold})"
            )
            return None

        # Validate number of matches
        num_matched = len(matches)
        if num_matched < self.min_required_apis:
            logger.warning(
                f"Only {num_matched} APIs matched from request '{request.request_id}' "
                f"(minimum required: {self.min_required_apis}), "
                "falling back to regular processing"
            )
            return None

        # Extract selected API indices and confidence scores
        selected_indices = [idx for idx, _ in matches]
        confidence_scores = [conf for _, conf in matches]

        # Check confidence scores
        low_confidence_apis = [
            (idx, conf) for idx, conf in matches
            if conf < self.match_threshold
        ]
        if low_confidence_apis:
            logger.warning(
                f"Some APIs in request '{request.request_id}' have low confidence: "
                f"{low_confidence_apis}, falling back to regular processing"
            )
            return None

        # Generate attention mask
        try:
            attention_mask = self.generate_attention_mask(template, selected_indices)
        except Exception as e:
            logger.error(f"Failed to generate attention mask: {e}")
            return None

        # Extract token ranges
        token_ranges = []
        for idx in selected_indices:
            api = template.api_descriptions[idx]
            token_ranges.append((api.token_start, api.token_end))

        # Create mapping
        mapping = TemplateMapping(
            template_id=template.template_id,
            selected_api_indices=selected_indices,
            confidence_scores=confidence_scores,
            attention_mask=attention_mask,
            token_ranges=token_ranges,
        )

        # Validate
        if not mapping.validate():
            logger.error(f"Template mapping validation failed for request '{request.request_id}'")
            return None

        logger.info(
            f"Successfully mapped request '{request.request_id}' to template "
            f"'{template.template_id}' with {num_matched} APIs "
            f"(avg confidence: {sum(confidence_scores) / len(confidence_scores):.3f})"
        )

        return mapping

    def generate_attention_mask(
        self,
        template: "APITemplate",
        selected_indices: list[int],
    ) -> torch.Tensor:
        """Generate binary attention mask for selected APIs.

        The mask has the same length as the template prompt token IDs.
        Tokens belonging to selected APIs are marked True (visible),
        all other tokens are marked False (masked).

        Args:
            template: The APITemplate containing all API descriptions
            selected_indices: Indices of selected APIs in the template

        Returns:
            Boolean tensor of shape (template_length,)
            True = attend to this token, False = ignore this token

        Example:
            For a 50-API template with APIs [5, 12, 23] selected:
            - Mask positions for APIs 5, 12, 23 will be True
            - All other positions will be False
        """
        template_length = len(template.prompt_token_ids)
        mask = torch.zeros(template_length, dtype=torch.bool)

        for idx in selected_indices:
            if idx < 0 or idx >= len(template.api_descriptions):
                logger.warning(f"Invalid API index {idx}, skipping")
                continue

            api = template.api_descriptions[idx]
            # Mark tokens for this API as visible
            mask[api.token_start:api.token_end] = True

        return mask

    def generate_sparse_mask(
        self,
        template: "APITemplate",
        selected_indices: list[int],
    ) -> torch.Tensor:
        """Generate a sparse mask alternative for memory efficiency.

        Instead of a full binary mask, this returns a list of (start, end) ranges
        that can be used for block-wise masking in the attention kernel.

        Args:
            template: The APITemplate
            selected_indices: Indices of selected APIs

        Returns:
            List of (token_start, token_end) tuples for contiguous selected ranges
        """
        # Sort selected indices
        sorted_indices = sorted(selected_indices)

        # Merge overlapping or adjacent ranges
        merged_ranges = []
        for idx in sorted_indices:
            api = template.api_descriptions[idx]
            if not merged_ranges:
                merged_ranges.append([api.token_start, api.token_end])
            else:
                last_start, last_end = merged_ranges[-1]
                # Merge if adjacent or overlapping
                if api.token_start <= last_end:
                    merged_ranges[-1][1] = max(last_end, api.token_end)
                else:
                    merged_ranges.append([api.token_start, api.token_end])

        return [(start, end) for start, end in merged_ranges]

    def get_api_descriptions_for_request(
        self,
        mapping: TemplateMapping,
        template: "APITemplate",
    ) -> list[str]:
        """Get human-readable API descriptions for a mapping.

        Args:
            mapping: The TemplateMapping from a previous map_request call
            template: The APITemplate used for mapping

        Returns:
            List of API descriptions (one per selected API)
        """
        descriptions = []
        for idx in mapping.selected_api_indices:
            api = template.get_api_by_index(idx)
            if api:
                descriptions.append(f"{api.api_name}: {api.description}")
        return descriptions

    def estimate_match_quality(
        self,
        mapping: TemplateMapping,
    ) -> dict[str, float]:
        """Estimate the quality of a template mapping.

        Args:
            mapping: The TemplateMapping to evaluate

        Returns:
            Dictionary with quality metrics:
            - avg_confidence: Average confidence score
            - min_confidence: Minimum confidence score
            - num_apis: Number of matched APIs
            - coverage_ratio: Ratio of selected tokens to total tokens
        """
        if not mapping.confidence_scores:
            return {
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "num_apis": 0,
                "coverage_ratio": 0.0,
            }

        avg_confidence = sum(mapping.confidence_scores) / len(mapping.confidence_scores)
        min_confidence = min(mapping.confidence_scores)

        # Calculate token coverage
        if mapping.attention_mask is not None:
            num_selected_tokens = mapping.attention_mask.sum().item()
            total_tokens = mapping.attention_mask.numel()
            coverage_ratio = num_selected_tokens / total_tokens if total_tokens > 0 else 0.0
        else:
            coverage_ratio = 0.0

        return {
            "avg_confidence": avg_confidence,
            "min_confidence": min_confidence,
            "num_apis": len(mapping.selected_api_indices),
            "coverage_ratio": coverage_ratio,
        }
