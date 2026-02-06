import torch
import triton
import triton.language as tl

def packed_bool_to_i8(bool_mask: torch.Tensor) -> torch.Tensor:
    orig_shape = bool_mask.shape
    # 步骤1：行优先展平为一维（8×8→64，保证原始布尔位的分组顺序）
    flat_mask = bool_mask.flatten()
    flat_len = flat_mask.numel()
    assert flat_len % 8 == 0, "输入总元素数需8对齐（最后一维8对齐则天然满足）"
    
    # 步骤2：8个连续布尔位为一组压缩，生成有效值（64→8）
    num_packed = flat_len // 8  # 压缩后有效元素数
    mask_8group = flat_mask.reshape(num_packed, 8)  # 按顺序分块
    # 核心位权：b0(第0位)→128(2^7)，b7(第7位)→1(2^0)
    weights = torch.tensor(
        [1 << i for i in range(0, 8)],
        dtype=torch.uint8,
        device=bool_mask.device
    )
    packed_vals = (mask_8group.to(torch.uint8) * weights).sum(dim=-1, dtype=torch.uint8)
    
    # 步骤3：补0至原展平长度（8→64），结果为[有效8位, 0,0,...0(56位)]，完全匹配你的预期
    padded_flat = torch.cat([
        packed_vals,  # 前8位：压缩的有效数据
        torch.zeros(flat_len - num_packed, dtype=torch.uint8, device=bool_mask.device)  # 后56位：pad 0
    ], dim=0)
    
    # 步骤4：reshape回原形状，因padded_flat前8位是有效数据，reshape后自然对应第一行，余行全0
    output = padded_flat.reshape(orig_shape)
    
    return output


@triton.jit
def kernel_mask(
    out:tl.tensor,
    mask:tl.tensor,
    STRIDE_R:tl.constexpr,
    STRIDE_C:tl.constexpr,
    BLOCK:tl.constexpr,
):
    msk = tl.load(
        mask 
        + tl.arange(0, BLOCK)[:, None] * STRIDE_R 
        + tl.arange(0, BLOCK)[None, :] * STRIDE_C
    )

    res = tl.where(msk, 100, 0)
    tl.compile_hint(res, "bitwise_mask")

    tl.store(
        out + tl.arange(0, BLOCK)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :], 
        res
    )


def mask_ref(A: torch.Tensor) -> torch.Tensor:
    B = torch.full_like(A, 100, dtype=torch.int16)
    # 掩码赋值：True的位置设为0
    B = torch.where(A, 0, 100)
    return B

if __name__ == "__main__":
    BLOCK_SIZE = 8
    idx = torch.arange(0, BLOCK_SIZE, dtype=torch.int)
    mask_bool = idx[:, None] == idx[None, :]
    print(f"{mask_bool=}")

    mask_i8 = packed_bool_to_i8(mask_bool).to('npu:0')
    print(f"{mask_i8=}")

    o = torch.ones((BLOCK_SIZE, BLOCK_SIZE), dtype=torch.int16).to('npu:0')
    kernel_mask[(1,)](o, mask_i8, mask_i8.stride(0), mask_i8.stride(1), BLOCK_SIZE)
    print(f"{o.cpu()=}")

    o_ref = mask_ref(mask_bool)
    print(o_ref)
    assert (o.cpu() == o_ref).all()