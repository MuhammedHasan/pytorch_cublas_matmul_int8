from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


with open("README.md") as f:
    readme = f.read()

setup(
    name="torch_cublas_matmul_int8",
    version='0.0.1',
    description="Multiply two int8 matrices using cuBLAS",
    keywords=['LLM', 'Quantization'],
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/muhammedhasan",
    license="MIT",

    packages=find_packages(include=['torch_cublas_matmul_int8*']),
    include_package_data=True,
    zip_safe=True,

    install_requires=[
        'setuptools',
        'tqdm',
        'torch',
    ],

    test_suite='tests',
    tests_require=[
        'pytest',
        'pytest-runner',
    ],

    ext_modules=[
        CUDAExtension(
            name='matmul_cublasLt',
            sources=[
                'csrc/matmul_ampere.cpp',
            ],
            extra_compile_args={
                'nvcc': ['-lcublasLt']
            },
            libraries=['cublasLt']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

