#include <torch/extension.h>

void run_int8_matmul_kernel(int8_t* a, int8_t* b, int32_t* c, int M, int N, int K);

torch::Tensor int8_matmul_wrapper(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "Input B must be a CUDA tensor");

    TORCH_CHECK(a.dtype() == torch::kInt8, "Input A must be Int8");
    TORCH_CHECK(b.dtype() == torch::kInt8, "Input B must be Int8");

    TORCH_CHECK(a.is_contiguous(), "Input A must be contiguous"); // because kernel does manual pointer arithmetic
    TORCH_CHECK(b.is_contiguous(), "Input B must be contiguous");

    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    TORCH_CHECK(b.size(0) == K, "Matrix dimensions of A and B are not compatible");

    auto c = torch::zeros({M, N}, torch::device(a.device()).dtype(torch::kInt32));

    int8_t* d_a = (int8_t*)a.data_ptr<int8_t>();
    int8_t* d_b = (int8_t*)b.data_ptr<int8_t>();
    int32_t* d_c = (int32_t*)c.data_ptr<int32_t>();

    run_int8_matmul_kernel(d_a, d_b, d_c, M, N, K);

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &int8_matmul_wrapper, "Int8 Tensor Core Matmul (Shared Mem Tiling)");
}