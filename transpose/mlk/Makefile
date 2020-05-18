configure: 
	 (sh /opt/intel/compilers_and_libraries_2020.0.166/linux/mkl/bin/mklvars.sh intel64)
	 (export MKLROOT=/opt/intel/compilers_and_libraries_2020.0.166/linux/mkl)

all: compile

compile: configure
	gcc mlk_csr_to_csc.c  -DMKL_ILP64 -m64 -I${MKLROOT}/include  -L${MKLROOT}/lib/intel64/ -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

#Run the commands of make configure,all and followed by ./a.out
