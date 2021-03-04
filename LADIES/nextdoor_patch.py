
import numpy as np
import FastGCNSamplingPy3, LADIESSamplingPy3
import ctypes

libFastGCNSampling = ctypes.CDLL("./FastGCNSamplingPy3.so")
libLADIESSampling = ctypes.CDLL("./LADIESSamplingPy3.so")

class NextDoorSamplerFastGCN:

    def __init__(self, dataset, batch_size, graph, edges , nodes,layer_dims):
        self.edges = edges
        self.layer_dims = layer_dims
        self.nodes = nodes
        if (dataset == "reddit"):
            FastGCNSamplingPy3.initSampling("/mnt/homes/abhinav/GPUesque/input/reddit.data")
        elif (dataset == "ppi"):
            FastGCNSamplingPy3.initSampling("/mnt/homes/abhinav/GPUesque/input/ppi.data")
        elif (dataset == "orkut"):
            FastGCNSamplingPy3.initSampling("/mnt/homes/abhinav/GPUesque/input/orkut.data")
        elif (dataset == "livejournal"):
            FastGCNSamplingPy3.initSampling("/mnt/homes/abhinav/GPUesque/input/LJ1.data")
        elif (dataset == "patents"):
            FastGCNSamplingPy3.initSampling("/mnt/homes/abhinav/GPUesque/input/patents.data")

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

    def __init__(self, dataset, batch_size, graph, edges , nodes,layer_dims):
        self.edges = edges
        self.layer_dims = layer_dims
        self.nodes = nodes
        if (dataset == "reddit"):
            LADIESSamplingPy3.initSampling("/mnt/homes/abhinav/GPUesque/input/reddit.data")
        elif (dataset == "ppi"):
            LADIESSamplingPy3.initSampling("/mnt/homes/abhinav/GPUesque/input/ppi.data")
        elif (dataset == "orkut"):
            LADIESSamplingPy3.initSampling("/mnt/homes/abhinav/GPUesque/input/orkut.data")
        elif (dataset == "livejournal"):
            LADIESSamplingPy3.initSampling("/mnt/homes/abhinav/GPUesque/input/LJ1.data")
        elif (dataset == "patents"):
            LADIESSamplingPy3.initSampling("/mnt/homes/abhinav/GPUesque/input/patents.data")

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