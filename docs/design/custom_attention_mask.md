# Custom Attention Mask Design

## Overview

This document describes how to implement prompt-specific custom attention masks in vLLM using the FlexAttention backend's `logical_mask_mod` interface. This feature allows developers to customize attention patterns based on prompt characteristics, enabling use cases such as:

- **RAG (Retrieval-Augmented Generation)**: Bidirectional attention for retrieved document sections
- **Long-context processing**: Sliding window attention for improved efficiency
- **Prompt-specific patterns**: Different attention strategies for different prompt types
- **Multi-modal scenarios**: Special attention patterns for vision-language models

## Background: Attention Mask in vLLM

### FlexAttention Backend

vLLM's FlexAttention backend (`vllm/v1/attention/backends/flex_attention.py`) provides a flexible interface for customizing attention computation through mask modification functions.

### Key Components

1. **FlexAttentionMetadata**: Contains attention computation metadata including `logical_mask_mod` field
2. **FlexAttentionMetadataBuilder**: Builds metadata from common attention metadata
3. **mask_mod function**: The core interface for customizing attention masks

## Architecture

### Mask Modification Function Signature

```python
def mask_mod(
    b: torch.Tensor,      # batch index
    h: torch.Tensor,      # head index
    q_idx: torch.Tensor,  # query token logical position
    kv_idx: torch.Tensor  # key-value token logical position
) -> torch.Tensor:        # returns boolean mask (True=attend, False=mask)
```

**Important Notes**:
- `q_idx` and `kv_idx` are **logical indices** - vLLM automatically handles physical-to-logical conversion
- Physical KV cache layout is abstracted away
- The function is vectorized and operates on all query-key pairs simultaneously

### Mask Composition Pipeline

The mask composition follows a two-stage process:

```
Stage 1: Base Mask (Causal/Bidirectional)
    ↓
Stage 2: Custom Masks (Sliding Window, Prefix LM, User-defined)
    ↓
Final Mask: and_masks() / or_masks() combination
```

## Implementation Methods

### Method 1: Global Custom Mask (Recommended for Fixed Patterns)

**Location**: `vllm/v1/attention/backends/flex_attention.py`

**Use Case**: Apply a custom mask pattern to all requests

#### Step 1: Define Custom Mask Function

Add your custom mask function in the FlexAttention backend file:

```python
def custom_prompt_based_mask_mod(
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor
) -> torch.Tensor:
    """
    Custom attention mask based on prompt characteristics.

    Args:
        b: Batch index (can encode request type information)
        h: Head index
        q_idx: Query token logical position
        kv_idx: Key-value token logical position

    Returns:
        Boolean mask tensor where True allows attention
    """
    # Base causal mask
    mask = q_idx >= kv_idx

    # === Add Your Custom Logic ===

    # Example 1: Bidirectional attention for specific prompt types
    # Encode prompt type in batch index (lower 4 bits)
    prompt_type = b & 0xF
    is_special_prompt = prompt_type == 1  # Type 1 prompts
    if is_special_prompt.any():
        # Allow bidirectional attention in special region
        in_special_region = (q_idx >= 100) & (q_idx < 200)
        mask = torch.where(
            in_special_region & is_special_prompt,
            True,   # Allow bidirectional
            mask    # Otherwise keep original mask
        )

    # Example 2: Different sliding window sizes per prompt type
    window_size = torch.where(
        prompt_type == 2, 64,   # Type 2: 64-token window
        torch.where(
            prompt_type == 3, 32,   # Type 3: 32-token window
            float('inf')             # Others: full causal
        )
    )
    in_window = (q_idx - kv_idx) < window_size
    mask = mask & in_window

    # Example 3: Template-based masking
    # Prompt structure: [template (0-49)] + [user input (50+)]
    is_template_kv = kv_idx < 50
    is_user_query = q_idx >= 50
    # User query can only attend to first 10 template tokens
    mask = torch.where(
        is_user_query & is_template_kv,
        kv_idx < 10,
        mask
    )

    return mask
```

#### Step 2: Modify FlexAttentionMetadataBuilder

Update the `build()` method to use your custom mask:

```python
class FlexAttentionMetadataBuilder(AttentionMetadataBuilder[FlexAttentionMetadata]):
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlexAttentionMetadata:
        # ... existing code ...

        out = FlexAttentionMetadata(
            # ... other parameters ...
            logical_mask_mod=custom_prompt_based_mask_mod,  # Set custom mask
            # ... other parameters ...
        )
        return out
```

### Method 2: Request-Specific Custom Masks

**Location**: `vllm/v1/attention/backends/utils.py` and `flex_attention.py`

**Use Case**: Different masks for different requests based on prompt features

#### Step 1: Extend CommonAttentionMetadata

Add prompt-specific fields to `CommonAttentionMetadata`:

```python
@dataclass
class CommonAttentionMetadata:
    # ... existing fields ...

    # Add prompt feature fields
    prompt_types: torch.Tensor | None = None  # shape: (num_reqs,)
    special_token_ranges: list[tuple[int, int]] | None = None
    custom_mask_params: dict[str, Any] | None = None
```

#### Step 2: Create Prompt-Aware Mask Factory

```python
def make_prompt_aware_mask_mod(
    prompt_types: torch.Tensor,
    query_start_loc: torch.Tensor,
    custom_params: dict[str, Any] | None = None,
):
    """
    Factory function to create a prompt-aware mask modification function.

    Args:
        prompt_types: Tensor of shape (num_reqs,) with prompt type per request
        query_start_loc: Query start locations to map tokens to requests
        custom_params: Optional dictionary of custom parameters
    """

    def mask_mod(b: torch.Tensor, h: torch.Tensor,
                 q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        # Base causal mask
        mask = q_idx >= kv_idx

        # Map query position to request ID
        request_id = torch.searchsorted(query_start_loc[1:], q_idx, right=True)
        prompt_type = prompt_types[request_id]

        # Apply different logic based on prompt type
        # Note: This is simplified; actual implementation needs proper broadcasting
        if prompt_type == 0:
            # Standard causal attention
            return mask
        elif prompt_type == 1:
            # Prefix LM: bidirectional in prefix region
            prefix_len = 64
            in_prefix = (kv_idx < prefix_len) & (q_idx < prefix_len)
            return mask | in_prefix
        elif prompt_type == 2:
            # Sliding window attention
            window_size = custom_params.get('window_size', 128) if custom_params else 128
            return mask & ((q_idx - kv_idx) < window_size)
        else:
            return mask

    return mask_mod
```

#### Step 3: Use in FlexAttentionMetadataBuilder

```python
class FlexAttentionMetadataBuilder(AttentionMetadataBuilder[FlexAttentionMetadata]):
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        # ... existing code ...

        # Create prompt-aware mask_mod if prompt types are available
        if common_attn_metadata.prompt_types is not None:
            custom_mask = make_prompt_aware_mask_mod(
                common_attn_metadata.prompt_types,
                common_attn_metadata.query_start_loc,
                common_attn_metadata.custom_mask_params
            )
        else:
            custom_mask = causal_mask_mod  # Default to standard causal

        out = FlexAttentionMetadata(
            # ... other parameters ...
            logical_mask_mod=custom_mask,
            # ... other parameters ...
        )
        return out
```

### Method 3: Runtime Dynamic Modification

**Use Case**: Dynamically change mask behavior during model execution

This approach allows modifying the mask at runtime without recompiling:

```python
from vllm import LLM

# Initialize model
llm = LLM(model="your-model", attention_backend="FLEX_ATTENTION")

# Access the attention implementation
# Note: This requires internal access to the model structure
# and may need helper methods to be exposed

def runtime_custom_mask(b, h, q_idx, kv_idx):
    base_mask = q_idx >= kv_idx
    # Apply runtime-specific logic
    return base_mask & ((q_idx - kv_idx) < 100)

# Set the custom mask (requires extending FlexAttentionImpl API)
# llm.set_custom_mask_mod(runtime_custom_mask)
```

## Usage Examples

### Example 1: RAG (Retrieval-Augmented Generation)

**Scenario**: Retrieved documents should support bidirectional attention within themselves, while user queries remain causal.

```python
def rag_aware_mask_mod(b, h, q_idx, kv_idx):
    """
    RAG-aware attention mask.

    Prompt structure:
    - [0, doc_start): System prompt
    - [doc_start, doc_end): Retrieved documents (bidirectional)
    - [doc_end, ): User query and conversation (causal)
    """
    # Base causal mask
    mask = q_idx >= kv_idx

    # Document boundaries (can be made configurable)
    doc_start = 256
    doc_end = 1024

    # Identify document region
    in_doc_kv = (kv_idx >= doc_start) & (kv_idx < doc_end)
    in_doc_query = (q_idx >= doc_start) & (q_idx < doc_end)

    # Bidirectional within documents
    bidirectional = in_doc_kv & in_doc_query
    mask = mask | bidirectional

    # Post-document tokens can attend to documents
    post_doc_query = q_idx >= doc_end
    can_attend_doc = kv_idx < doc_end
    mask = mask | (post_doc_query & can_attend_doc)

    return mask
```

### Example 2: Long-Context Sliding Window

**Scenario**: Apply sliding window attention to long sequences while maintaining full attention for recent tokens.

```python
def sliding_window_with_full_recent(b, h, q_idx, kv_idx):
    """
    Sliding window with full attention for recent tokens.

    Pattern:
    - Recent 128 tokens: full causal attention
    - Older tokens: sliding window of 256 tokens
    """
    # Base causal mask
    mask = q_idx >= kv_idx

    # Define thresholds
    full_attention_threshold = 128
    window_size = 256

    # Check if in recent region
    is_recent = kv_idx >= (q_idx - full_attention_threshold)

    # Sliding window for older tokens
    in_window = (q_idx - kv_idx) < window_size

    # Combine: full attention for recent, window for older
    mask = mask & (is_recent | in_window)

    return mask
```

### Example 3: Prefix Language Model

**Scenario**: Bidirectional attention for prefix (e.g., prompts, instructions), causal for generation.

```python
def prefix_lm_mask_mod(prefix_len=512):
    """
    Creates a prefix LM mask modification function.

    Args:
        prefix_len: Length of the prefix region

    Returns:
        mask_mod function
    """
    def mask_mod(b, h, q_idx, kv_idx):
        # Base causal mask
        mask = q_idx >= kv_idx

        # Both query and kv in prefix: bidirectional
        in_prefix = (q_idx < prefix_len) & (kv_idx < prefix_len)

        # Query in prefix can attend to all prefix tokens
        # Query outside prefix can only attend causally
        mask = mask | in_prefix

        return mask

    return mask_mod

# Usage in FlexAttentionMetadataBuilder:
# logical_mask_mod=prefix_lm_mask_mod(prefix_len=512)
```

### Example 4: Multi-Modal Prefix LM

**Scenario**: Vision-language models where image tokens require bidirectional attention.

```python
def multimodal_prefix_lm_mask_mod(vision_ranges: list[tuple[int, int]]):
    """
    Multi-modal prefix LM with special handling for image tokens.

    Args:
        vision_ranges: List of (start, end) tuples for vision token ranges

    Returns:
        mask_mod function
    """
    vision_ranges_tensor = torch.tensor(vision_ranges)

    def mask_mod(b, h, q_idx, kv_idx):
        # Base causal mask
        mask = q_idx >= kv_idx

        # Check if both query and kv are in the same vision region
        for start, end in vision_ranges:
            in_vision_q = (q_idx >= start) & (q_idx < end)
            in_vision_kv = (kv_idx >= start) & (kv_idx < end)
            same_vision = in_vision_q & in_vision_kv
            mask = mask | same_vision

        # Text tokens can attend to all vision tokens (if before them)
        # This is already handled by causal mask

        return mask

    return mask_mod
```

## Integration Points

### Scheduler Integration

To pass prompt-specific information to the attention layer, you need to extend the scheduler:

```python
# In scheduler output
class SchedulerOutput:
    # ... existing fields ...

    # Add prompt metadata
    prompt_types: np.ndarray | None = None
    special_token_ranges: dict[int, tuple[int, int]] | None = None
```

### Configuration

Add configuration options to enable custom masks:

```python
# In vllm/config.py
class CacheConfig:
    # ... existing fields ...

    # Custom mask configuration
    enable_custom_attention_mask: bool = False
    custom_mask_type: str | None = None  # "rag", "sliding_window", etc.
    custom_mask_params: dict[str, Any] = field(default_factory=dict)
```

## Performance Considerations

### Block Mask Efficiency

FlexAttention uses block-wise mask computation for efficiency. Custom masks should be designed to work well with this:

1. **Block-aligned patterns**: Masks that align with block boundaries (default 16 tokens) are more efficient
2. **Simple boolean operations**: Complex nested conditions can impact performance
3. **Vectorization**: The mask_mod function should be vectorized, avoiding Python loops

### Compilation

The `create_block_mask` function is compiled with `torch.compile` for efficiency. Custom mask functions that are:
- **Static** (same computation graph) benefit most from compilation
- **Dynamic** (different logic per request) may trigger recompilation

### Direct Build Path

FlexAttention has a "direct build" path for standard causal attention that's more efficient. Complex custom masks may fall back to the generic path, which has some performance overhead.

```python
# In FlexAttentionMetadata
direct_build: bool = True  # Set to False for complex custom masks
```

## Testing

### Unit Tests

Create tests for your custom mask logic:

```python
# tests/v1/attention/test_custom_masks.py

import torch
from vllm.v1.attention.backends.flex_attention import custom_prompt_based_mask_mod

def test_custom_mask_basic():
    """Test basic custom mask functionality."""
    b = torch.tensor([0, 0, 0])  # 3 queries in batch 0
    h = torch.tensor([0, 0, 0])  # All in head 0
    q_idx = torch.tensor([10, 20, 30])
    kv_idx = torch.tensor([5, 15, 25])

    mask = custom_prompt_based_mask_mod(b, h, q_idx, kv_idx)

    # Assertions
    assert mask.shape == (3,)
    assert mask[0] == True   # q=10 can attend to kv=5
    assert mask[1] == True   # q=20 can attend to kv=15
    assert mask[2] == True   # q=30 can attend to kv=25

def test_sliding_window_mask():
    """Test sliding window mask."""
    # ... implementation ...

def test_rag_mask():
    """Test RAG-aware mask."""
    # ... implementation ...
```

### Integration Tests

Test end-to-end with model inference:

```python
def test_custom_mask_inference():
    """Test custom mask with actual model inference."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="facebook/opt-125m",
        attention_backend="FLEX_ATTENTION",
        enable_custom_attention_mask=True,
        custom_mask_type="rag",
        custom_mask_params={"doc_start": 256, "doc_end": 1024}
    )

    prompts = [
        "System prompt here. " + "Document content. " * 50 + "User question?",
    ]

    outputs = llm.generate(prompts)
    assert len(outputs) == 1
    # Verify output quality and correctness
```

## Debugging

### Visualizing Masks

To debug custom masks, you can visualize them:

```python
def visualize_mask(mask_mod_fn, seq_len=20, batch_size=1):
    """Visualize an attention mask."""
    b = torch.zeros(seq_len, dtype=torch.long)
    h = torch.zeros(seq_len, dtype=torch.long)
    q_idx = torch.arange(seq_len)
    kv_idx = torch.arange(seq_len).unsqueeze(1).expand(seq_len, seq_len)

    mask = mask_mod_fn(b.flatten(), h.flatten(),
                       q_idx.flatten(), kv_idx.flatten())
    mask_matrix = mask.reshape(seq_len, seq_len)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.imshow(mask_matrix.cpu().numpy(), cmap='viridis')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Custom Attention Mask')
    plt.colorbar()
    plt.show()

# Usage
# visualize_mask(custom_prompt_based_mask_mod)
```

### Logging

Add logging to track mask application:

```python
import logging
logger = logging.getLogger(__name__)

def custom_mask_with_logging(b, h, q_idx, kv_idx):
    mask = q_idx >= kv_idx  # Base causal

    # Log mask statistics
    logger.debug(f"Mask shape: {mask.shape}")
    logger.debug(f"Mask coverage: {mask.float().mean().item():.2%}")

    return mask
```

## Advanced Topics

### Combining Multiple Masks

FlexAttention supports combining multiple masks using `and_masks` and `or_masks`:

```python
from torch.nn.attention.flex_attention import and_masks, or_masks

# In FlexAttentionMetadata.get_mask_mod()
def get_mask_mod(self):
    # Start with base mask
    if self.causal:
        mask_mod = self.get_causal_mask_mod()
    else:
        mask_mod = self.get_bidirectional_mask_mod()

    # Add sliding window (AND operation)
    if self.sliding_window is not None:
        sliding_mask_mod = self.get_sliding_window_mask_mod()
        mask_mod = and_masks(mask_mod, sliding_mask_mod)

    # Add prefix LM (OR operation)
    if self.mm_prefix_range:
        prefix_mask_mod = self.get_prefix_lm_mask_mod()
        mask_mod = or_masks(mask_mod, prefix_mask_mod)

    # Add your custom mask
    custom_mask_mod = self.logical_mask_mod
    mask_mod = and_masks(mask_mod, custom_mask_mod)

    return mask_mod
```

### Score Modification

Besides masks, you can also modify attention scores:

```python
# In FlexAttentionMetadata
score_mod: _score_mod_signature | None = None

# Example: Apply ALiBi-style bias
def alibi_score_mod(
    score: torch.Tensor,
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    physical_q: torch.Tensor  # Physical query position
) -> torch.Tensor:
    # Apply distance-based bias
    slope = 1 / (2 ** (h.float() // 8))  # Different slope per head
    bias = (kv_idx - q_idx) * slope
    return score + bias
```

## Migration Guide

### From FlashAttention to FlexAttention

If you're currently using FlashAttention and want to migrate to FlexAttention for custom mask support:

1. **Change backend configuration**:
   ```python
   llm = LLM(model="...", attention_backend="FLEX_ATTENTION")
   ```

2. **Verify compatibility**:
   - FlexAttention requires PyTorch 2.5+
   - Check if your use case is supported (encoder-only, decoder-only)

3. **Performance comparison**:
   - Benchmark both backends
   - FlexAttention may have overhead for simple patterns

### From Custom Attention Implementation

If you have a custom attention implementation:

1. Extract the mask logic into a `mask_mod` function
2. Convert from physical to logical indices (let vLLM handle this)
3. Test correctness by comparing attention weights
4. Optimize for block-wise computation

## Limitations and Future Work

### Current Limitations

1. **Dynamic shapes**: Masks that change shape per request may trigger recompilation
2. **Complex patterns**: Highly non-local masks may be inefficient
3. **Encoder-decoder**: Cross-attention support is limited

### Planned Enhancements

1. **More efficient block mask compilation** for custom patterns
2. **Automatic pattern detection** to optimize common cases
3. **JIT compilation** of mask_mod functions
4. **API for runtime mask modification**

## References

- [PyTorch FlexAttention Documentation](https://pytorch.org/blog/flexattention/)
- vLLM FlexAttention implementation: `vllm/v1/attention/backends/flex_attention.py`
- vLLM attention architecture: `vllm/attention/backends/abstract.py`
- Template caching with masks: `docs/source/features/templating.md`

## Contributing

To contribute custom mask implementations:

1. Add your mask function to `flex_attention.py`
2. Write comprehensive tests
3. Add documentation with usage examples
4. Benchmark performance impact
5. Submit PR with clear use case description

## Appendix: Complete Example

Here's a complete example of implementing and using a custom attention mask:

```python
# File: vllm/v1/attention/backends/flex_attention.py

# ... existing imports ...

def my_custom_mask_mod(
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    """
    My custom attention mask implementation.

    This mask implements:
    1. Standard causal attention
    2. Bidirectional attention for prefix region (first 128 tokens)
    3. Sliding window of 256 for older tokens
    """
    # Base causal mask
    mask = q_idx >= kv_idx

    # Prefix bidirectional region
    prefix_len = 128
    in_prefix = (q_idx < prefix_len) & (kv_idx < prefix_len)
    mask = mask | in_prefix

    # Sliding window for tokens after prefix
    window_size = 256
    in_window = (q_idx - kv_idx) < window_size
    mask = mask & in_window

    return mask

# Modify FlexAttentionMetadataBuilder
class FlexAttentionMetadataBuilder(AttentionMetadataBuilder[FlexAttentionMetadata]):
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlexAttentionMetadata:
        # ... existing code ...

        out = FlexAttentionMetadata(
            causal=common_attn_metadata.causal,
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            block_size=block_size,
            max_possible_sequence_length=max_possible_seq_len,
            num_reqs=num_reqs,
            physical_to_logical=inverse_block_table,
            total_cache_tokens=total_cache_tokens,
            decode_offset=offset_tensor,
            num_blocks_per_seq=num_blocks_per_seq,
            direct_build=(self.direct_build and common_attn_metadata.causal),
            q_block_size=self.q_block_size,
            kv_block_size=self.kv_block_size,
            logical_mask_mod=my_custom_mask_mod,  # <-- Set custom mask here
        )
        return out

# Usage:
# from vllm import LLM
# llm = LLM(model="meta-llama/Llama-2-7b-hf",
#           attention_backend="FLEX_ATTENTION")
# outputs = llm.generate(["Your prompt here"])
```

---

**Document Version**: 1.0
**Last Updated**: 2025-01-09
**Status**: Design Document
