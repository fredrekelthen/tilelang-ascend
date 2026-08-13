# T.tile.transpose

## 1. Description

Performs a 2D matrix block transposition: `dst[i][j] = src[j][i]` (i ∈ [0, W), j ∈ [0, H), where src shape is [H, W] and dst shape is [W, H]).

## 2. Function Prototype

### 2.1 Function Definition

```python
def transpose(
    dst: Buffer,
    src: Buffer,
)
```

### 2.2 Parameters

| Parameter | Direction | Description | Type | Required/Optional |
|-----------|-----------|-------------|------|-------------------|
| dst | Output | Stores the transposition result, shape `[W, H]` | tensor | Required |
| src | Input | The source matrix to transpose, shape `[H, W]` | tensor | Required |

> **Type notes**:
> - **tensor**: A buffer allocated via `T.alloc_ub`, `T.alloc_shared`, etc. This API only accepts whole Buffer objects, not BufferRegion slices

### 2.3 Parameter Specifications

#### 2.3.1 DataType Support

| Platform | dst | src |
|----------|:---:|:---:|
| Ascend A2 / A3 | float16, int16, uint16, float32, int32, uint32 | float16, int16, uint16, float32, int32, uint32 |

- dtypes other than those listed above (int8, bfloat16, int64, etc.) fall back to a scalar element-wise implementation. Results are correct but performance is lower. Among these, int64 is only supported on the ascendc backend (the pto backend's TTRANS instruction does not support 8-byte dtypes and will fail at compile time)

#### 2.3.2 Shape Support

- 2D only; `src` shape is `[H, W]`, `dst` shape is `[W, H]`
- H and W of `src` must be compile-time static values (dynamic dimensions cause a compile-time error)

#### 2.3.3 Implementation Path

The API automatically dispatches to different implementation paths based on dtype and shape:

| Condition | Implementation Path | Notes |
|-----------|---------------------|-------|
| H=16, W=16, B16 dtype (except bfloat16) | `AscendC::Transpose` hardware instruction | Single instruction, fastest path |
| H and W both multiples of 16, B16/B32 dtype (except bfloat16) | `TransDataTo5HD` block transpose | Transposes in 16×16 sub-blocks |
| Other (int8, bfloat16, or H/W not multiples of 16 but 32-byte aligned) | Scalar element-wise loop | Correct results, lower performance |

### 2.4 Constraints

1. dst and src must have the same dtype
2. dst shape must be `[W, H]`, i.e. the transpose of src shape `[H, W]`
3. H and W of src must satisfy 32-byte alignment: `H * sizeof(dtype)` and `W * sizeof(dtype)` must both be multiples of 32 (hardware constraint). Specifically: B16 dtypes (float16/int16/uint16/bfloat16) require H, W to be multiples of 16; B32 dtypes (float32/int32/uint32) require multiples of 8; int8 requires multiples of 32
4. H and W of src must be compile-time static values
5. src and dst addresses must not overlap (in-place transpose is not supported)
6. All operand addresses must be 32-byte aligned (hardware constraint)

## 3. Example Code

**Example 1: 16×16 transpose (hardware instruction path)**

```python
src = T.alloc_ub((16, 16), "float16")
dst = T.alloc_ub((16, 16), "float16")
T.tile.transpose(dst, src)
```

**Example 2: Non-square matrix transpose (block path)**

```python
src = T.alloc_ub((64, 32), "float32")
dst = T.alloc_ub((32, 64), "float32")
T.tile.transpose(dst, src)
```

**Example 3: int8 scalar fallback path**

```python
src = T.alloc_ub((32, 32), "int8")
dst = T.alloc_ub((32, 32), "int8")
T.tile.transpose(dst, src)
```
