# T.tile.sin

## 1. 功能说明

对源操作数逐元素执行正弦运算：`dst[i] = sin(src[i])`

> sin 为独立实现，直接映射到 `tl.ascend_sin` intrinsic（Ascend C 高阶 API `Sin`，多项式近似实现）。Ascend C 后端内部需要临时空间（tmpBuffer）存储中间变量；**PTO 后端不支持 sin**（编译期报 `tl.ascend_sin is not supported by the PTO backend`，无原生 PTO-ISA 指令）。

## 2. 函数原型

### 2.1 函数定义

```python
def sin(
    dst: Buffer | BufferRegion,
    src: Buffer | BufferRegion,
    *,
    tmp: Buffer | BufferRegion | None = None,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放正弦运算结果 | 张量（tensor） | 必填 |
| src | 输入 | 源操作数 | 张量（tensor） | 必填 |
| tmp | 输入 | 显式指定的 UB 临时缓冲区（一般无需用户提供，省略时框架自动申请） | 张量（tensor） | 可选 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src |
|------|:---:|:---:|
| Ascend A2 / A3（Ascend C 后端） | float16, float32 | float16, float32 |

- 整数 dtype（int16/int32）与 bfloat16 编译失败（`src/transform/allocate_tmp_buffer.cc` 校验 `dtype.is_float() && bits ∈ {16, 32}`）
- PTO 后端：不支持（编译期报错）

#### 2.3.2 Shape 支持

- 支持 1D 和 2D
- 支持整行切片（如 `buf[0:32, :]`）；仅计算切片区域内元素，区域外内容未定义

### 2.4 约束条件

1. dst 与 src 的元素总数必须相同，否则前端 assert 失败（"size must be same"）
2. dst 与 src 的 dtype 必须一致（Ascend C 约束）
3. 操作数地址需 32 字节对齐（硬件约束）
4. **支持原地运算**（dst 与 src 为同一 buffer，Ascend C 后端真机实测正确）
5. 特殊值遵循 IEEE 语义（真机实测）：`sin(0)=0`、`sin(±inf)=nan`、`sin(nan)=nan`
6. 精度（真机实测，Ascend910B3）：float16 相对误差 ≤ 5e-4（多项式近似）、float32 绝对误差 ≤ 1.2e-7
7. 输入超出 [-65504.0, 65504.0] 时（float32 大值输入），多项式近似不做范围归约，结果与 IEEE 参考值可能不一致
8. 接口内部使用框架自动申请的临时缓冲区（fp16：`max(2S, 512)` 字节、fp32：`max(2S, 384)` 字节，S 为 src 字节数），无需用户手动分配；也可通过 `tmp` 参数显式提供

## 3. 示例代码

**示例 1：1D 正弦运算**

```python
src = T.alloc_ub((1024,), "float16")
dst = T.alloc_ub((1024,), "float16")
T.tile.sin(dst, src)
```

**示例 2：2D 正弦运算（整行切片）**

```python
src = T.alloc_ub((128, 64), "float16")
dst = T.alloc_ub((128, 64), "float16")
T.tile.sin(dst[0:64, :], src[0:64, :])
```