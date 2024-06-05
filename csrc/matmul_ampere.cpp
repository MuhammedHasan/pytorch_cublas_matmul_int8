#include <torch/extension.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <iostream>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

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

// CUDA error checking
#define cudaCheck(status)                                                                                                 \
    {                                                                                                                     \
        cudaError_t err = status;                                                                                         \
        if (err != cudaSuccess)                                                                                           \
        {                                                                                                                 \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                                                                           \
        }                                                                                                                 \
    }

#define SET_LAYOUT_ORDER(desc, order) cublasCheck(cublasLtMatrixLayoutSetAttribute(desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)))

torch::Tensor transform_col32_to_row(torch::Tensor A)
{

    A = A.contiguous();
    int n = A.size(0), m = A.size(1);
    cublasLtHandle_t handle = reinterpret_cast<cublasLtHandle_t>(
        at::cuda::getCurrentCUDABlasHandle());

    cublasLtMatrixLayout_t layoutA;
    int ld = n * 32;
    cublasCheck(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_32I, n, m, ld));
    cublasLtOrder_t col32 = CUBLASLT_ORDER_COL32;
    SET_LAYOUT_ORDER(layoutA, col32);

    cublasLtMatrixLayout_t outLayout;
    cublasCheck(cublasLtMatrixLayoutCreate(&outLayout, CUDA_R_32I, n, m, m));
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    SET_LAYOUT_ORDER(outLayout, order);

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(A.device());
    auto output = torch::empty({n, m}, options);

    cublasLtMatrixTransformDesc_t transformDesc;
    cublasCheck(cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F));

    float alpha = 1.0f, beta = 0.0f;
    cublasCheck(cublasLtMatrixTransform(
        handle,
        transformDesc,
        &alpha,
        A.data_ptr<int32_t>(), layoutA,
        &beta,
        nullptr, nullptr,
        output.data_ptr<int32_t>(), outLayout, 0));

    cublasCheck(cublasLtMatrixLayoutDestroy(outLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(layoutA));
    cublasCheck(cublasLtMatrixTransformDescDestroy(transformDesc));

    return output;
}

typedef enum Format_t
{
    COL32 = 0,
    AMPERE = 1,
} Transform_t;

template <int FORMAT>
torch::Tensor transform_from_row_to(torch::Tensor A)
{
    A = A.contiguous();
    int n = A.size(0), m = A.size(1);
    cublasLtHandle_t handle = reinterpret_cast<cublasLtHandle_t>(
        at::cuda::getCurrentCUDABlasHandle());

    cublasLtMatrixLayout_t Alayout;
    cublasCheck(cublasLtMatrixLayoutCreate(&Alayout, CUDA_R_8I, n, m, m));
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    SET_LAYOUT_ORDER(Alayout, order);

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(A.device());
    auto output = torch::empty({n, m}, options);

    cublasLtMatrixLayout_t outLayout;

    if (FORMAT == AMPERE)
    {
        int ld = (n + 31) / 32 * 32 * 32;
        cublasCheck(cublasLtMatrixLayoutCreate(&outLayout, CUDA_R_8I, n, m, ld));
        cublasLtOrder_t colAmpere = CUBLASLT_ORDER_COL32_2R_4R4;
        SET_LAYOUT_ORDER(outLayout, colAmpere);
    }
    else if (FORMAT == COL32)
    {
        int ld = n * 32;
        cublasCheck(cublasLtMatrixLayoutCreate(&outLayout, CUDA_R_8I, n, m, ld));
        cublasLtOrder_t col32 = CUBLASLT_ORDER_COL32;
        SET_LAYOUT_ORDER(outLayout, col32);
    }
    else
    {
        std::cerr << "Invalid format" << std::endl;
        exit(EXIT_FAILURE);
    }

    cublasLtMatrixTransformDesc_t transformDesc;
    cublasCheck(cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F));

    float alpha = 1.0f, beta = 0.0f;
    cublasCheck(cublasLtMatrixTransform(
        handle,
        transformDesc,
        &alpha,
        A.data_ptr<int8_t>(), Alayout,
        &beta,
        nullptr, nullptr,
        output.data_ptr<int8_t>(), outLayout, 0));

    cublasCheck(cublasLtMatrixLayoutDestroy(Alayout));
    cublasCheck(cublasLtMatrixTransformDescDestroy(transformDesc));

    return output;
}
template torch::Tensor transform_from_row_to<AMPERE>(torch::Tensor A);
template torch::Tensor transform_from_row_to<COL32>(torch::Tensor A);

torch::Tensor int8_matmul_cublaslt(torch::Tensor A, torch::Tensor B)
{
    A = A.contiguous();
    B = B.contiguous();
    int m = A.size(0), n = B.size(0), k = A.size(1);

    cublasLtHandle_t handle = reinterpret_cast<cublasLtHandle_t>(
        at::cuda::getCurrentCUDABlasHandle());

    int lda = m * 32;
    int ldb = (n + 31) / 32 * 32 * 32;
    int ldc = m * 32;

    cublasLtMatrixLayout_t Alayout, Blayout, Clayout;
    cublasCheck(cublasLtMatrixLayoutCreate(&Alayout, CUDA_R_8I, m, k, lda));
    cublasCheck(cublasLtMatrixLayoutCreate(&Blayout, CUDA_R_8I, n, k, ldb));
    cublasCheck(cublasLtMatrixLayoutCreate(&Clayout, CUDA_R_32I, m, n, ldc));

    cublasLtOrder_t col32 = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t colAmpere = CUBLASLT_ORDER_COL32_2R_4R4;

    SET_LAYOUT_ORDER(Alayout, col32);
    SET_LAYOUT_ORDER(Blayout, colAmpere);
    SET_LAYOUT_ORDER(Clayout, col32);

    cublasLtMatmulDesc_t matmulDesc;
    cublasOperation_t opT = CUBLAS_OP_T;
    cublasCheck(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    cublasCheck(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT)));

    auto optionsInt32 = torch::TensorOptions().dtype(torch::kInt32).device(A.device());
    auto output = torch::empty({m, n}, optionsInt32);

    float alpha = 1.0f, beta = 0.0f;
    cublasCheck(cublasLtMatmul(
        handle, matmulDesc,
        &alpha,
        A.data_ptr<int8_t>(), Alayout,
        B.data_ptr<int8_t>(), Blayout,
        &beta,
        output.data_ptr<int32_t>(), Clayout,
        output.data_ptr<int32_t>(), Clayout,
        nullptr, nullptr, 0, 0));

    cublasCheck(cublasLtMatmulDescDestroy(matmulDesc));
    cublasCheck(cublasLtMatrixLayoutDestroy(Alayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(Blayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(Clayout));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("transform_from_row_to_ampere", &transform_from_row_to<AMPERE>, "transform from row to ampere");
    m.def("transform_from_row_to_col32", &transform_from_row_to<COL32>, "transform from row to col32");
    m.def("transform_col32_to_row", &transform_col32_to_row, "transform from col32 to row");
    m.def("matmul_int8", &int8_matmul_cublaslt, "int8 multiply using cuBLASLt");
}