// How to compile (assume CUDA is installed at /usr/local/cuda/)
//   nvcc spmv_example.c -o spmv_example -L/usr/local/cuda/lib64 -lcusparse -lcudart
// or, for C compiler
//   cc -I/usr/local/cuda/include -c spmv_example.c -o spmv_example.o -std=c99
//   nvcc -lcusparse -lcudart spmv_example.o -o spmv_example
#include <cuda_runtime.h>  // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>      // cusparseSpMV
#include <stdio.h>         // printf
#include <stdlib.h>        // EXIT_FAILURE

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

int main() {
    // Host problem definition
    const int A_num_rows = 4;
    const int A_num_cols = 4;
    const int A_num_nnz  = 9;
    int   hA_csrOffsets[] = { 0, 3, 4, 7, 9 };
    int   hA_columns[]    = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    float hA_values[]     = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                              6.0f, 7.0f, 8.0f, 9.0f };
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_csrOffsets, *dA_columns, *dAt_coloffsets, * dAt_rows;
    float *dA_values,* dAt_values, *dX, *dY;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )

    CHECK_CUDA( cudaMalloc((void**) &dAt_values,  A_num_nnz * sizeof(float)) )		    
    CHECK_CUDA( cudaMalloc((void**) &dAt_coloffsets,  (A_num_cols+1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dAt_rows,  A_num_nnz * sizeof(int)) )

    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_num_nnz * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_values, A_num_nnz * sizeof(float)) )
			
    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_num_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values,
                           A_num_nnz * sizeof(float), cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = 0;
    cusparseSpMatDescr_t matA;
    void* dBuffer=NULL;
    size_t bufferSize = 10000;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_num_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    

    CHECK_CUSPARSE( cudaMalloc(&dBuffer, bufferSize) )

    CHECK_CUSPARSE( cusparseCsr2cscEx2_bufferSize(handle,A_num_rows, A_num_cols,A_num_nnz,
                   dA_values, dA_csrOffsets, dA_columns,
                   dAt_values, dAt_coloffsets, dAt_rows,
                   CUDA_R_32F,
		  CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, 
		   CUSPARSE_CSR2CSC_ALG1,
		   &bufferSize) )




    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
 	

    float dAt[A_num_nnz];	    
    CHECK_CUDA( cudaMemcpy(dAt,dA_values, A_num_nnz * sizeof(float),cudaMemcpyDeviceToHost )) 	    
    for(int i=0;i<A_num_nnz;i++){
    	printf("%f",dAt[i]);
    }
    
        printf("Transpose Passed\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    return EXIT_SUCCESS;
}
