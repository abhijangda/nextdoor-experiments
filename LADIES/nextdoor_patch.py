
import numpy as np
import NextDoor

class NextDoorSamplerFastGCN:

    def __init__(self, graph, edges , nodes,layer_dims):
        self.edges = edges
        self.layer_dims = layer_dims
        self.nodes = nodes
        NextDoor.initSampling("/mnt/homes/abhinav/GPUesque-for-eval/input/reddit_sampled_matrix")

    def sample(self,num_nodes):
        layers = []
        #print("nodes", len(self.nodes))
        for layer in self.layer_dims:
            layers += [ np.random.choice(num_nodes, layer,replace = True) ]
        return layers    
