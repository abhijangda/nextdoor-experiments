all: sparse.cu
	nvcc -I/usr/local/cuda/include -L/usr/local/cuda-10.1/targets/x86_64-linux/lib/ -I/mnt/homes/abhinav/cub-1.8.0 -lcusparse -lcudart sparse.cu -o sparse -Xptxas -O3
