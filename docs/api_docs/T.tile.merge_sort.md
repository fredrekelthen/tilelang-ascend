# T.tile.merge_sort

## 1. Description

Merges 2/3/4 descending-sorted queues into a single descending-sorted queue: `dst = MrgSort(src0, src1, [src2], [src3])`. Both input and output use the interleaved (value, index) pair format (each element occupies 2 buffer positions): `src = [val0, idx0, val1, idx1, ...]`, and `dst[i]` is the i-th largest (value, index) pair globally. The blockLen (valid element count) of each source is derived automatically from its buffer size.

## 2. Function Prototype

### 2.1 Function Definition

```python
def merge_sort(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    src1: Buffer | BufferRegion,
    src2: Buffer | BufferRegion | None = None,
    src3: Buffer | BufferRegion | None = None,
    *,
    tmp: Buffer | BufferRegion | None = None,
)
```

### 2.2 Parameters

| Parameter | Direction | Description | Type | Required/Optional |
|-----------|-----------|-------------|------|-------------------|
| dst | Output | Stores the merged (value, index) interleaved pairs. Its size must be at least the sum of all source buffer sizes | tensor | Required |
| src0 | Input | First source queue, sorted in descending order | tensor | Required |
| src1 | Input | Second source queue, sorted in descending order | tensor | Required |
| src2 | Input | Third source queue, sorted in descending order (for 3-way or 4-way merge) | tensor / None | Optional (default `None`) |
| src3 | Input | Fourth source queue, sorted in descending order (for 4-way merge only) | tensor / None | Optional (default `None`) |
| tmp | Input | Optional temporary buffer. The ascendc backend does not use it (an explicitly passed tmp is omitted from the emitted call); the pto backend needs a workspace — when omitted (None) it allocates a workspace of 1× the dst size automatically, and an explicitly passed tmp must be a non-empty buffer large enough to hold it (a zero-length tmp raises an error) | tensor / None | Optional (default `None`) |

> **Type notes**:
> - **tensor**: A buffer allocated via `T.alloc_ub`, `T.alloc_shared`, etc., or a slice (BufferRegion) of such a buffer

### 2.3 Parameter Specifications

#### 2.3.1 DataType Support

| Platform | dst | src0 | src1 | src2 | src3 |
|----------|:---:|:----:|:----:|:----:|:----:|
| Ascend A2 / A3 | float32 | float32 | float32 | float32 | float32 |

> **Note**: Only `float32` is supported. `float16` inputs produce incorrect results in the current implementation (the ascendc backend triggers an aicore exception). To sort half data, cast it to float32 with `T.cast` first.

#### 2.3.2 Shape Support

- Supports 1D buffers and slices of 1D buffers; the slice start must be 32-byte aligned
- Supports whole-row slices of 2D buffers (e.g. `buf[0, :]`, `buf[1, :]`)
- Column-offset slices of 2D buffers (e.g. `buf[0, 8:136]`) are NOT supported: codegen emits a wrong address and the result is unreliable; multi-row region slices (e.g. `buf[0:2, :]`) are supported on ascendc only (pto fails to compile)
- The merge width is determined automatically by whether src2 / src3 is None:

| src2 | src3 | Merge Width | Availability |
|------|------|-------------|--------------|
| None | None | 2-way | Supported on both ascendc and pto |
| Not None | None | 3-way | Supported on both ascendc and pto |
| Not None | Not None | 4-way | Supported on both ascendc and pto |

#### 2.3.3 blockLen Description

The blockLen (valid element count) of each source queue is derived from its buffer size: `blockLen = buffer_size // 2` (value-index pair format, each element occupies 2 buffer positions).

- **ascendc**: blockLen ∈ [1, 4095]; sources may have different blockLens (unequal-length merge)
- **pto**: blockLen ∈ [4, 4088], and all sources must have the same blockLen (equal-length merge); unequal lengths or out-of-range values fail at compile time or at runtime

### 2.4 Constraints

1. The number of source queues must be 2, 3 or 4 (determined by whether src2 / src3 is None); `num_ways < 2` or `> 4` raises a `ValueError`
2. All sources and dst must have the same dtype, float32; a dtype mismatch raises a compile error
3. dst size must be at least the sum of all source buffer sizes (2-way: `dst >= src0 + src1`; a dst larger than the sum also works, as verified). If dst is smaller than the sum, the kernel does not report an error but produces wrong results (out-of-bounds writes into adjacent UB memory)
4. Each source queue must already be sorted in descending order (typically the output of `T.tile.sort32` / `T.tile.sort`)
5. blockLen = src size / 2; the ascendc backend supports blockLen ∈ [1, 4095] (AscendC MrgSort elementLengths assertion), the pto backend supports blockLen ∈ [4, 4088] (hardware constraint). blockLen = 0 (an empty src buffer) raises a compile-time divide-by-zero error
6. dst and src addresses must be 32-byte aligned (hardware constraint; verified: 1D slice offsets of 4/8/16 bytes trigger aicore exceptions, 32 bytes and above work; whole buffers allocated with `T.alloc_ub` / `T.alloc_shared` satisfy this naturally)
7. Stable merge: when scores are equal, sources are consumed in src0 → src1 → src2 → src3 order, preserving the original order inside each source queue
8. The pto backend requires all sources to have the same size (equal-length merge); unequal-length merges are supported on the ascendc backend only (pto fails to compile with "no matching function")
9. NaN inputs have no deterministic ordering semantics; their output positions are not guaranteed (hardware behavior)
10. tmp parameter: the ascendc backend ignores it (an explicitly passed tmp is omitted from codegen); the pto backend needs a workspace — when omitted (None) it is allocated automatically, and an explicitly passed tmp must be a non-empty buffer large enough to hold the required size (a zero-length tmp raises an error)

## 3. Example Code

**Example 1: 2-way merge**

```python
src0 = T.alloc_ub((64,), "float32")
src1 = T.alloc_ub((64,), "float32")
dst  = T.alloc_ub((128,), "float32")
T.tile.merge_sort(dst, src0, src1)
```

**Example 2: 3-way merge**

```python
src0 = T.alloc_ub((64,), "float32")
src1 = T.alloc_ub((64,), "float32")
src2 = T.alloc_ub((64,), "float32")
dst  = T.alloc_ub((192,), "float32")
T.tile.merge_sort(dst, src0, src1, src2)
```