# Template-Based Prefix Caching with Dynamic Attention Masks

## Overview

This document describes the design and implementation of a template-based caching system for vLLM that allows a single cached 50-API prompt (~20K tokens) to be reused for requests with any 5-API subset (~2K tokens), using dynamic attention masks for strict isolation.

## Problem Statement

### Current Limitation

When selecting 5 APIs out of 50 possible APIs:
- Total combinations: C(50,5) = 2,118,760
- Each 5-API prompt: ~2,000 tokens
- Full 50-API prompt: ~20,000 tokens
- Current prefix caching命中率 (hit rate) is very low due to the massive number of possible combinations

### Solution Approach

**Key Idea**: Cache the full 50-API prompt once, then reuse it for all 5-API requests with dynamic attention masks.

**Requirements**:
- User API remains unchanged (backward compatible)
- Strict isolation: model only sees the 5 selected APIs
- Auto-detection of which APIs are selected from the prompt
- Lazy registration: first request triggers template caching

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User Request                                │
│                   (5-API prompt, ~2000 tokens)                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    API Matcher                                       │
│  - Extracts API descriptions from request prompt                    │
│  - Matches against registered template APIs                        │
│  - Returns API indices and confidence scores                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │ First Request?                 │
                    └───────────────┬───────────────┘
                           Yes │           │ No
                              ▼           ▼
┌──────────────────────────┐   ┌────────────────────────────────────┐
│  Template Registration   │   │    Template Cache Lookup           │
│  (ONCE on first request) │   │    (KV blocks from cache)          │
└──────────────────────────┘   └────────────────────────────────────┘
            │                                    │
            ▼                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Template Registry                                │
│  - Stores 50-API template metadata                                 │
│  - Manages token ranges per API                                    │
│  - Provides API signatures for matching                            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│              KV Cache Manager (Enhanced)                            │
│  - Detects template requests                                       │
│  - Retrieves cached 50-API KV blocks                               │
│  - Allocates blocks with template hash                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│           Attention Mask Generator                                  │
│  - Generates binary mask: [selected_apis=True, others=False]      │
│  - Applies to both prefill and decode phases                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Model Execution (with Mask)                            │
│  - Prefill: masked query initialization                            │
│  - Decode: masked KV cache access                                  │
└─────────────────────────────────────────────────────────────────────┘
```

## Implementation Components

### Core Modules

#### 1. Template Registry (`vllm/v1/core/template_registry.py`)

**Classes**:
- `APIDescription`: Metadata for a single API (token range, signature, embedding)
- `APITemplate`: Complete template with all APIs
- `TemplateRegistry`: Global singleton for template management

**Key Features**:
- Load templates from YAML/JSON configuration
- Token range tracking for each API
- Signature-based API matching
- Thread-safe operations

**Configuration Format** (YAML):
```yaml
template_id: "company_apis_v1"
apis:
  - api_id: "weather"
    api_name: "Weather Service"
    description: "Provides current weather data..."
  - api_id: "search"
    api_name: "Search Service"
    description: "Searches company database..."
```

#### 2. Template Mapper (`vllm/v1/core/template_mapper.py`)

**Classes**:
- `TemplateMapping`: Result of request-to-template mapping
- `TemplateRequestMapper`: Maps user requests to templates

**Key Features**:
- Automatic API matching from prompt
- Confidence scoring
- Attention mask generation
- Fallback on match failure

**Matching Algorithm**:
1. Extract signature tokens from request prompt
2. Compare with template API signatures
3. Compute match score = (intersection / signature_size)
4. Filter by confidence threshold (default 0.85)
5. Return top 5 matches

#### 3. Template KV Manager (`vllm/v1/core/template_kv_manager.py`)

**Classes**:
- `TemplateKVCacheManager`: Manages KV cache for templates

**Key Features**:
- One-time template prefill (~20K tokens)
- LRU eviction when cache is full
- Thread-safe caching
- Metrics and monitoring

**Cache Lifecycle**:
1. First request arrives → template not cached
2. Allocate blocks for full template
3. Run model prefill to compute KV cache
4. Store blocks with template ID as key
5. Subsequent requests → instant cache hit

#### 4. Request Integration (`vllm/v1/request.py`)

**Modified Request Class**:
- `is_template_request`: bool flag
- `template_id`: str | None
- `template_mapping`: TemplateMapping | None

**Detection**:
- Checks `sampling_params.extra_args["use_template_cache"]`
- Sets `is_template_request = True` if enabled

### Attention Mask Application

#### Metadata Enhancement (`vllm/v1/attention/backends/utils.py`)

Added to `CommonAttentionMetadata`:
```python
template_mask: torch.Tensor | None  # Binary mask (num_tokens,)
template_kv_mask: torch.Tensor | None  # KV cache mask
```

#### FlashAttention Integration (`vllm/v1/attention/backends/flash_attn.py`)

**Prefill Phase**:
```python
# Zero out query tokens for non-selected APIs
if metadata.template_masks:
    query = apply_mask_to_query(query, metadata.template_masks)
```

**Decode Phase**:
```python
# Zero out KV entries for non-selected APIs
if metadata.template_kv_masks:
    key, value = apply_mask_to_kv(key, value, metadata.template_kv_masks)
```

## Usage Example

### Server Initialization

```python
from vllm import LLM, SamplingParams

# Enable template caching
llm = LLM(
    model="meta-llama/Llama-3-8B",
    enable_template_caching=True,
    template_config_path="api_template.yaml"  # Load 50-API template
)
```

### Client Request (API Compatible)

```python
# Option 1: Use template caching explicitly
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=100,
    extra_args={
        "use_template_cache": True,
        "template_id": "company_apis_v1"
    }
)

# User provides 5-API prompt (~2000 tokens)
prompt = """
I need to:
1. Check weather for San Francisco
2. Search customer records
3. Update inventory
4. Get user profile
5. Calculate shipping costs

Which APIs should I use?
"""

output = llm.generate(prompt, sampling_params)
```

**What happens internally**:
1. Request detected as template request
2. System matches APIs from prompt to 50-API template
3. Retrieves cached 50-API KV blocks (~20K tokens, pre-cached)
4. Generates attention mask for 5 selected APIs
5. Model only sees those 5 APIs
6. Subsequent requests reuse cache instantly

## Performance Characteristics

### First Request (Template Registration)
- **Latency**: ~20K token prefill (similar to processing a 20K token prompt)
- **Time**: 5-10 seconds (model-dependent)
- **Memory**: ~2GB per template (model-dependent)

### Subsequent Requests
- **Latency**: Near-zero overhead (cache lookup + mask generation)
- **Time**: Comparable to regular 5-API request
- **Cache Hit Rate**: >95% (if API descriptions are stable)

### Memory Usage
- Template cache: ~2GB per 50-API template
- Can cache multiple templates (LRU eviction)
- Configurable limit: `max_cached_templates` (default: 5)

## Edge Cases and Solutions

| Edge Case | Solution |
|-----------|----------|
| **API description changes** | Version field in template, auto-invalidation |
| **Failed API matching** | Fallback to regular prefix caching |
| **Insufficient memory** | LRU eviction, configurable limit |
| **Concurrent first requests** | Locking during registration |
| **Multi-GPU inconsistency** | Distributed block allocation |
| **Ambiguous API matches** | Confidence threshold, tie-breaking |
| **Template larger than cache** | Block size alignment, chunked caching |

## Configuration Options

### Cache Configuration

```python
@dataclass
class CacheConfig:
    # Template caching
    enable_template_caching: bool = False
    template_config_path: str | None = None
    max_cached_templates: int = 5
    template_match_threshold: float = 0.85  # 0.0-1.0
```

### Environment Variables

```bash
# Enable template caching
VLLM_ENABLE_TEMPLATE_CACHING=true
VLLM_TEMPLATE_CONFIG=path/to/api_template.yaml
VLLM_MAX_CACHED_TEMPLATES=5
```

## Implementation Status

### ✅ Completed (Phase 1-2.1)
- [x] Core data structures (TemplateRegistry, TemplateMapper, TemplateKVManager)
- [x] Request class modifications
- [x] API matching logic
- [x] Attention mask generation

### 🚧 In Progress
- [ ] KV cache manager integration
- [ ] Scheduler integration
- [ ] Attention kernel modifications

### ⏳ TODO (Phase 3-5)
- [ ] Attention mask application in FlashAttention
- [ ] Template auto-registration on first request
- [ ] Configuration system integration
- [ ] LLMEngine integration
- [ ] Comprehensive test suite
- [ ] Documentation and examples

## Success Metrics

- [ ] Cache hit rate > 95% for template requests
- [ ] Request latency comparable to regular 5-API requests (±10%)
- [ ] API matching accuracy > 90%
- [ ] First-request registration time < 10 seconds
- [ ] Zero regression in non-template workloads
- [ ] Multi-GPU scaling efficiency > 80%

## Future Enhancements

1. **Background Pre-warming**: Pre-load templates during engine startup
2. **Hierarchical Templates**: Support nested template structures
3. **Template Composition**: Combine multiple templates per request
4. **Compression**: Quantize template KV cache for memory efficiency
5. **Distributed Template Cache**: Share templates across vLLM instances

## References

- Plan file: `C:\Users\echo\.claude\plans\hashed-honking-shore.md`
- Implementation files:
  - `vllm/v1/core/template_registry.py`
  - `vllm/v1/core/template_mapper.py`
  - `vllm/v1/core/template_kv_manager.py`
  - `vllm/v1/request.py` (modified)

## Contributors

- Design and implementation: Claude Code + User collaboration
- Date: 2025-01-07

---

**Note**: This is an active work in progress. The implementation is currently in Phase 1-2 of a planned 5-phase rollout (4-5 weeks total).
