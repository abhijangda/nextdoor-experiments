
import numpy as np
import NextDoor
import ctypes

lib = ctypes.CDLL("./NextDoor.so")

class NextDoorSamplerFastGCN:

    def __init__(self, batch_size, graph, edges , nodes,layer_dims):
        self.edges = edges
        self.layer_dims = layer_dims
        self.nodes = nodes
        NextDoor.initSampling("/mnt/homes/abhinav/GPUesque-for-eval/input/reddit_sampled_matrix")
        self.samples = []
        self.batch_size = batch_size

    def sample(self,num_nodes):
        layers = []
        #print("nodes", len(self.nodes))
        if (self.samples == []):
            NextDoor.sample()
            length = NextDoor.finalSampleLength()
            print("length ", length)
            lib.finalSamplesArray.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape=(length))
            finalSamples = lib.finalSamplesArray()
            num_layers = len(self.layer_dims)
            newshape = (len(finalSamples)//(self.layer_dims[0]*num_layers), num_layers,self.layer_dims[0])
            print((num_layers, self.layer_dims[0]), newshape)
            self.samples = np.reshape(finalSamples, newshape)
            print(self.samples.shape)
            return self.samples
        else:
            return self.samples