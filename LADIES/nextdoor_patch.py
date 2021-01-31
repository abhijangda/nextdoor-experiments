
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
            lib.finalSamplesArray.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape=(NextDoor.finalSampleLength()))
            finalSamples = lib.finalSamplesArray()
            num_layers = 1 #len(self.layer_dims)
            newshape = (len(finalSamples)//(self.batch_size*num_layers), num_layers,self.batch_size)
            print((num_layers, self.batch_size), newshape)
            self.samples = np.reshape(finalSamples, newshape)
            print(self.samples.shape)
            return self.samples
        else:
            return self.samples