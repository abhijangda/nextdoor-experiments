
import numpy as np

class NextDoorSamplerFastGCN:

    def __init__(self, edges , nodes,layer_dims):
        self.edges = edges
        self.layer_dims = layer_dims
        self.nodes = nodes

    def sample(self,batch_nodes):
        layers = []
        for layer in self.layer_dims:
            layers += [ np.random.choice(self.nodes, layer, replace = True) ]
        return layers    
