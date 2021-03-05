
import numpy as np, os
import FastGCNSamplingPy3, LADIESSamplingPy3
import ctypes

libFastGCNSampling = ctypes.CDLL("./FastGCNSamplingPy3.so")
libLADIESSampling = ctypes.CDLL("./LADIESSamplingPy3.so")

class NextDoorSamplerFastGCN:

    def __init__(self, dataset, batch_size, graph_dir, edges , nodes,layer_dims):
        self.edges = edges
        self.layer_dims = layer_dims
        self.nodes = nodes
        filename = os.path.join(graph_dir,dataset+".data")
        FastGCNSamplingPy3.initSampling(filename)
        self.samples = []
        self.batch_size = batch_size

    def freeDeviceMemory(self):
        FastGCNSamplingPy3.freeDeviceMemory()
    

    def sample(self,num_nodes):
        layers = []
        #print("nodes", len(self.nodes))
        if (self.samples == []):
            FastGCNSamplingPy3.sample()
            length = FastGCNSamplingPy3.finalSampleLength()
            print("length ", length)
            libFastGCNSampling.finalSamplesArray.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape=(length))
            finalSamples = libFastGCNSampling.finalSamplesArray()
            num_layers = len(self.layer_dims)
            newshape = (len(finalSamples)//(self.layer_dims[0]*num_layers), num_layers,self.layer_dims[0])
            print((num_layers, self.layer_dims[0]), newshape)
            self.samples = np.reshape(finalSamples, newshape)
            print(self.samples.shape)
            return self.samples
        else:
            return self.samples

class NextDoorSamplerLADIES:

    def __init__(self, dataset, batch_size, graph_dir, edges , nodes,layer_dims):
        self.edges = edges
        self.layer_dims = layer_dims
        self.nodes = nodes
        filename = os.path.join(graph_dir,dataset+".data")
        LADIESSamplingPy3.initSampling(filename)
        self.samples = []
        self.batch_size = batch_size

    def freeDeviceMemory(self):
        LADIESSamplingPy3.freeDeviceMemory()

    def sample(self,num_nodes):
        layers = []
        #print("nodes", len(self.nodes))
        if (self.samples == []):
            LADIESSamplingPy3.sample()
            length = LADIESSamplingPy3.finalSampleLength()
            print("length ", length)
            libLADIESSampling.finalSamplesArray.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape=(length))
            finalSamples = libLADIESSampling.finalSamplesArray()
            num_layers = len(self.layer_dims)
            newshape = (len(finalSamples)//(self.layer_dims[0]*num_layers), num_layers,self.layer_dims[0])
            print((num_layers, self.layer_dims[0]), newshape)
            self.samples = np.reshape(finalSamples, newshape)
            print(self.samples.shape)
            return self.samples
        else:
            return self.samples