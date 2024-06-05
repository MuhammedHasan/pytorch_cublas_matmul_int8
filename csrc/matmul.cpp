#include <utility>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <iostream>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>

// cuBLAS error checking
void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status)                        \
    {                                              \
        cublasCheck((status), __FILE__, __LINE__); \
    }

torch::Tensor int8_matmul_cublaslt(torch::Tensor a, torch::Tensor b)
{
    a = a.contiguous();
    b = b.contiguous();

    cublasLtHandle_t handle = reinterpret_cast<cublasLtHandle_t>(
        at::cuda::getCurrentCUDABlasHandle());

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(a.device());
    auto c = torch::zeros({a.size(0), b.size(1)}, options);

    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    cublasCheck(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8I, a.size(1), a.size(0), a.size(1)));
    cublasCheck(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8I, b.size(1), b.size(0), b.size(1)));
    cublasCheck(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32I, b.size(1), a.size(0), b.size(1)));

    cublasLtMatmulDesc_t operationDesc;
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));

    int8_t *a_data = (int8_t *)a.data_ptr();
    int8_t *b_data = (int8_t *)b.data_ptr();
    int32_t *c_data = (int32_t *)c.data_ptr();

    float alpha = 1.0f, beta = 0.0f;
    cublasCheck(cublasLtMatmul(
        handle,
        operationDesc,
        &alpha,
        b_data, layoutB, // a and b swapped
        a_data, layoutA,
        &beta,
        c_data, layoutC,
        c_data, layoutC,
        nullptr,
        nullptr, 0, 0));

    cublasCheck(cublasLtMatrixLayoutDestroy(layoutA));
    cublasCheck(cublasLtMatrixLayoutDestroy(layoutB));
    cublasCheck(cublasLtMatrixLayoutDestroy(layoutC));
    cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
    cublasCheck(cublasLtDestroy(handle));

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("matmul_int8", &int8_matmul_cublaslt, "int8 multiply");
}
