import pytest
import torch
from torch_cublas_matmul_int8 import matmul_int8

device = 0

def test_matmul_int8():

    A = torch.randint(-128, 127, size=(2048, 1184),
                      device=device).to(torch.int8)
    B = torch.randint(-128, 127, size=(512, 1184),
                      device=device).to(torch.int8)

    torch.testing.assert_close(
        matmul_int8(A, B).float(),
        torch.matmul(A.float(), B.float().t()),
    )

    with pytest.raises(AssertionError):
        A = torch.randint(-128, 127, size=(2048, 31),
                          device=device).to(torch.int8)
        B = torch.randint(-128, 127, size=(513, 31),
                          device=device).to(torch.int8)
        matmul_int8(A, B)
