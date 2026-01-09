#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// dimensions of the Tensor Core operation
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// dimensions of tile calculated by one thread block
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 64

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 16
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

__global__ void wmma_int8_kernel(int8_t* A, int8_t* B, int32_t* C, int M, int N, int K) {
    /* CUDA Kernel to perform tiled Int8 GEMM */
    
    __shared__ int8_t smem_A[BLOCK_M * BLOCK_K];
    __shared__ int8_t smem_B[BLOCK_K * BLOCK_N];
    
    int blockRow = blockIdx.x * BLOCK_M; // starting row of 64x64 tile (global)
    int blockCol = blockIdx.y * BLOCK_N; // starting col of 64x64 tile (global)
    int warpId = threadIdx.x / WARP_SIZE; // local Warp ID within block, 0 -> 15

    // map Warp ID to 2D grid (4x4) inside block
    int warpRow = (warpId / 4) * WMMA_M;
    int warpCol = (warpId % 4) * WMMA_N;

    // declare registers for tensor core fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a_frag; // holds a fragment of the 16x16 chunk of matrix A
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> b_frag; // same for B
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag; // result

    wmma::fill_fragment(c_frag, 0);

    // loop through the tiles: width of A and height of B
    for (int k = 0; k < K; k += BLOCK_K) {
        // Cooperative Load from Global to Shared Memory

        int8_t* A_tile_ptr = A + (blockRow * K) + k; // top left corner of 64x64 tile in global mem
        
        // Copy this 64x64 tile of A from global mem to shared mem
        for (int i = threadIdx.x; i < BLOCK_M * BLOCK_K; i += THREADS_PER_BLOCK) {
            // convert linear index 'i' to (row, col) relative to 64x64 tile
            int local_row = i / BLOCK_K;
            int local_col = i % BLOCK_K;

            if ((blockRow + local_row) < M && (k + local_col) < K) {
                smem_A[i] = A_tile_ptr[local_row * K + local_col];
            } else {
                smem_A[i] = 0;
            }
        }

        // Same to copy 64x64 tile of B from global mem to shared mem
        int8_t* B_tile_ptr = B + (k * N) + blockCol;
        for (int i = threadIdx.x; i < BLOCK_K * BLOCK_N; i += THREADS_PER_BLOCK) {
            int local_row = i / BLOCK_N;
            int local_col = i % BLOCK_N;

            if ((k + local_row) < K && (blockCol + local_col) < N) {
                smem_B[i] = B_tile_ptr[local_row * N + local_col];
            } else {
                smem_B[i] = 0;
            }
        }

        __syncthreads();

        // Compute Loop, with A moving right and B moving down
        for (int k_sub = 0; k_sub < BLOCK_K; k_sub += WMMA_K) {
            int8_t* tile_a_ptr = &smem_A[warpRow * BLOCK_K + k_sub];
            wmma::load_matrix_sync(a_frag, tile_a_ptr, BLOCK_K);

            int8_t* tile_b_ptr = &smem_B[k_sub * BLOCK_N + warpCol];
            wmma::load_matrix_sync(b_frag, tile_b_ptr, BLOCK_N);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();
    }

    // where in C are we calculating for
    int cRow = blockRow + warpRow;
    int cCol = blockCol + warpCol;
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }

}

void run_int8_matmul_kernel(int8_t* a, int8_t* b, int32_t* c, int M, int N, int K) {
    dim3 blockDim(THREADS_PER_BLOCK);
    int grid_x = (M + BLOCK_M - 1) / BLOCK_M;
    int grid_y = (N + BLOCK_N - 1) / BLOCK_N;
    dim3 gridDim(grid_x, grid_y);

    wmma_int8_kernel<<<gridDim, blockDim>>>(a, b, c, M, N, K);
}