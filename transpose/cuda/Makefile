

all: sparse.c
	cc -I/usr/local/cuda/include -c sparse.c -o sparse.o -std=c99
	nvcc -L/usr/local/cuda-10.1/targets/x86_64-linux/lib/ -lcusparse -lcudart sparse.o -o sparse
	./sparse
