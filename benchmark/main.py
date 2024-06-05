import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.cpp_extension import load
import torch.utils.benchmark as benchmark
import bitsandbytes.functional as bnb_F


a = torch.randint(-128, 127, (2048, 4096), dtype=torch.int8, device=0)
b = torch.randint(-128, 127, (8192, 4096), dtype=torch.int8, device=0)

# ampere
ext = load(
    name='cublas_matmul_int8',
    sources=['csrc/matmul_ampere.cpp'],
    extra_include_paths=[f'{torch.utils.cpp_extension.CUDA_HOME}/include'],
)
def fn(a, b):
    a_col32 = ext.transform_from_row_to_col32(a)
    b_ampere = ext.transform_from_row_to_ampere(b)
    out_col32 = ext.matmul_int8(a_col32, b_ampere)

timer = benchmark.Timer(
    stmt='fn(a, b)',
    globals={'a': a, 'b': b, 'fn': fn},
    num_threads=torch.get_num_threads()
)
ampere_time = [timer.timeit(1000).mean for i in tqdm(range(10))]
print('Runtime ampere', np.mean(ampere_time))


# col
ext = load(
    name='cublas_matmul_int8_col',
    sources=['csrc/matmul.cpp'],
    extra_include_paths=[f'{torch.utils.cpp_extension.CUDA_HOME}/include'],
)
a = torch.randint(-128, 127, (2048, 4096), dtype=torch.int8, device=0)
b = torch.randint(-128, 127, (8192, 4096), dtype=torch.int8, device=0).t()


def fn(a, b):
    out = ext.matmul_int8(a, b)


timer = benchmark.Timer(
    stmt='fn(a, b)',
    globals={'a': a, 'b': b, 'fn': fn},
    num_threads=torch.get_num_threads()
)
time_col = [timer.timeit(1000).mean for i in tqdm(range(10))]
print('Runtime col', np.mean(time_col))


# pytorch
a = torch.randint(-128, 127, (2048, 4096), dtype=torch.int8, device=0)
b = torch.randint(-128, 127, (8192, 4096), dtype=torch.int8, device=0).t()


def fn(a, b):
    a.float() @ b.float()


timer = benchmark.Timer(
    stmt='fn(a, b)',
    globals={'a': a, 'b': b, 'fn': fn},
    num_threads=torch.get_num_threads()
)
torch_time = [timer.timeit(1000).mean for i in tqdm(range(10))]
print('Torch', np.mean(torch_time))


# bitsandbytes
a = torch.randint(-128, 127, (2048, 4096), dtype=torch.int8, device=0)
b = torch.randint(-128, 127, (8192, 4096), dtype=torch.int8, device=0)


def fn(a, b):
    cA, sA = bnb_F.transform(a, "col32")
    cB, sB = bnb_F.transform(b, "col_ampere")
    cC, sC = bnb_F.igemmlt(cA, cB, sA, sB)

timer = benchmark.Timer(
    stmt='fn(a, b)',
    globals={'a': a, 'b': b, 'fn': fn},
    num_threads=torch.get_num_threads()
)
bitsandbytes_time = [timer.timeit(1).mean for i in tqdm(range(10))]
print('bitsandbytes', np.mean(bitsandbytes_time))


df = pd.DataFrame({
    'torch': torch_time,
    'ampere': ampere_time,
    'col': time_col,
    'bitsandbytes': bitsandbytes_time,
})

plt.figure(figsize=(5, 5))
sns.boxplot(data=df.melt(), x='variable', y='value')
plt.xlabel('Implementation')
plt.ylabel('Runtime (s)')
plt.savefig('benchmark/runtime.png', dpi=300, bbox_inches='tight')
