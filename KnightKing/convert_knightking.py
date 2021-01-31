
import sys
import random
def processfile():
    if len(sys.argv) != 3:
        print("Incorrect number of arguments")
        exit()
    input_file = sys.argv[1]    
    output_file = sys.argv[2]
    infile = open(input_file,"r")
    outfile = open(output_file,"w")
    while True:
        line=infile.readline()
        if not line:
            break
        if(line.startswith('#')):
            continue
        a,b= line.strip().split()
        a = int(a)
        b = int(b)
        w = random.random()
        outfile.write("{}\t{}\t{}\n".format(a,b,w))
if __name__=="__main__":
    processfile()
