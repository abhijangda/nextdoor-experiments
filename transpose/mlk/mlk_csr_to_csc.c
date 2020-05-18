#include "stdio.h"
#include "mkl.h"


int main()
{


// Spare Matrix definition
//https://scc.ustc.edu.cn/zlsc/tc4600/intel/2017.0.098/mkl/common/mklman_f/GUID-910E399F-11B5-46B8-9750-87EF52679AE8.htm
// Use three array indexing and attempt to transpose
sparse_index_base_t indexing = 0;
MKL_INT rows = 3;
MKL_INT rows_indx[] = {0,1,2,3,5};
MKL_INT col_indx[] = {1,1,1,1,2};
float values[] = {1,2,3,4,5};
	

// Begin Transpose
// https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c/top/blas-and-sparse-blas-routines/sparse-blas-level-2-and-level-3-routines/sparse-blas-level-2-and-level-3-routines-1/mkl-csrcsc.html
MKL_INT job[] = {0,0,0,0,0,1};
MKL_INT m = 4;
MKL_INT  ja1[5], ia1[5],info;
float acsc[5];
mkl_scsrcsc(job, &m, values, col_indx, rows_indx, acsc, ja1, ia1,&info); 

// Check all values are set
for(int i=0;i<5;i++){
	printf("% f",acsc[i]);
}
/*or sasum(&n,arr,&inc) ?? --- both give similar errors*/

printf("result");
return 0;
}
