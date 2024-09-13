import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def naive_softmax(x: torch.tensor):
    x_max = x.max(dim=-1)[0]
    safe_x = x - x_max[:, None]
    num = torch.exp(safe_x)
    denum = num.sum(dim=-1)
    return num/denum[:, None]

@triton.jit
def _softmax_fwd_kernel(
    output_ptr,
    stride_output_row,
    input_ptr,
    stride_input_row,
    num_cols,
    block_size: tl.constexpr,
):

    # set up input ptr
    row_idx = tl.program_id(0)

    row_start_ptr = input_ptr + row_idx * stride_input_row
    col_offsets = tl.arange(0, block_size)
    input_pointers = row_start_ptr + col_offsets

    row_masks = col_offsets < num_cols
    # Move to SRAM
    row = tl.load(input_pointers, mask=row_masks, other=float("-inf"))

    # Start softmax computation
    safe_row = row - tl.max(row, axis=0)
    num = tl.exp(safe_row)
    denum = tl.sum(num, axis=0)
    out = num/denum

    # write to HBM
    output_row_ptr = output_ptr + row_idx * stride_output_row
    output_pointers = output_row_ptr + col_offsets

    tl.store(output_pointers, out, mask=row_masks)






def softmax(x: torch.tensor) -> torch.tensor:
    """Triton implementation of softmax. fwd pass"""
    rows, cols = x.shape
    # Parallize on the row.
    block_size = triton.next_power_of_2(cols)

    num_warps = 4 # num_warps * 32 = # threads
    while block_size > num_warps * 32:  # Each warp has 32 threads
        num_warps *= 2
    grid = (rows, )

    # allocate output buffer
    out = torch.empty_like(x)

    _softmax_fwd_kernel[grid](
        out, 
        out.stride(0),
        x,
        x.stride(0),
        cols,
        block_size=block_size,
        num_warps=num_warps # Only observed by compiler. Not needed in kernel
        )
    return out


sample = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=torch.float32, device='cuda')
ref_out = F.softmax(sample, dim=-1)
print(f'{ref_out=}')

naive_out = naive_softmax(sample)
print(f'{naive_out=}')
assert torch.allclose(ref_out, naive_out, rtol=1e-05, atol=1e-08), "Tensors are not nearly equal"

triton_out = softmax(sample)
print(f'{triton_out=}')
assert torch.allclose(ref_out, triton_out, rtol=1e-05, atol=1e-08), "Tensors are not nearly equal"
