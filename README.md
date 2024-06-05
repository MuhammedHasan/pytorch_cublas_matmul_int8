# torch_cublas_matmut_int8

pyTorch does not support int8 multiplication. This is a simple implementation of int8 multiplication using pyTorch and cuBLAS:

```python
import torch
from torch_cublas_matmul_int8 import matmul_int8

A = torch.randint(-128, 127, size=(2048, 1184),
                    device=0).to(torch.int8)
B = torch.randint(-128, 127, size=(512, 1184),
                    device=0).to(torch.int8)
matmul_int8(A, B)
```

pyTorch will throw error:

```python
torch.matmul(A, B)
# Traceback (most recent call last):
# File "<stdin>", line 1, in <module>
# RuntimeError: "addmm_cuda" not implemented for 'Char'
```

## Installation

```bash
# install pyTorch 
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia  
# install cuda-tools
conda install nvidia::cuda-toolkit
# install the package
pip install torch_cublas_matmul_int8
```

The current implementation only support ampere architecture.

