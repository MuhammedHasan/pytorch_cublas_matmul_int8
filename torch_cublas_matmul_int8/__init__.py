import torch
import matmul_cublasLt


def matmul_int8(A, B):
    '''
    Perform matrix multiplication of two int8 matrices A and B: A @ B^T

    Args:
        A (torch.Tensor): 2D matrix of shape (m, k)
        B (torch.Tensor): 2D matrix of shape (n, k).
            Note that k must be a multiple of 32.

    Returns:
        torch.Tensor: 2D matrix of shape (m, n)
    '''
    assert A.dtype == torch.int8, f"Expected int8, got {A.dtype}"
    assert B.dtype == torch.int8,   f"Expected int8, got {B.dtype}"
    assert A.dim() == 2, f"Expected 2D, got {A.dim()}"
    assert B.dim() == 2, f"Expected 2D, got {B.dim()}"
    assert A.size(1) == B.size(1), \
        f"Expected A and B to have the same number of columns, got {A.size(1)} and {B.size(1)}"
    assert A.device == B.device, \
        f"Expected A and B to be on the same device, got {A.device} and {B.device}"
    assert A.device.type == 'cuda', \
        f"Expected A and B to be on the GPU, got {A.device.type}"
    assert B.size(1) % 32 == 0, \
        f"Expected B to have a multiple of 32 columns, got {B.size(1)}"

    A_col32 = matmul_cublasLt.transform_from_row_to_col32(A)
    B_ampere = matmul_cublasLt.transform_from_row_to_ampere(B)
    out_col32 = matmul_cublasLt.matmul_int8(A_col32, B_ampere)
    return matmul_cublasLt.transform_col32_to_row(out_col32)

