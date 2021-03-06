import numpy as np

class GraphKHopSampler():

    def __init__(self,G, layer_infos):
        self.G = G
        self.layer_infos = layer_infos
        self.support_sizes =  []
        p = 1
        for l in range(len(layer_infos)):
            # Reference: neigh_sampler.py on why this order of layer_infos is followed
            self.support_sizes.append(p * layer_infos[len(layer_infos)-1-l].num_samples)
            p = self.support_sizes[l]
        assert(self.support_sizes == [10,250])

    # Scratch function
    # Function has to be replaced with next door
    # nodes of batch_size
    # returns a dictionarky khop with {"hop1":10*len(nodes),"hop2":250*len(nodes)}
    def getKHopSampleSizes(self, nodes):
        target = [0 for i in range(len(self.layer_infos))]

        for i in range(len(self.layer_infos)):
            size = 0
            
            target[i] = self.support_sizes[i] * len(nodes)

        return target
        
    def getKHopSamples(self, nodes):
        khop = {}
        nodes = np.asarray(nodes)
        for i in range(len(self.layer_infos)):
            size = 0
            
            target = self.support_sizes[i] * len(nodes)
            output = np.ndarray((target,), dtype=np.int32)
            
            while(size < target):
                if(size+ len(nodes) <= target):
                    output[size:size+len(nodes)] = nodes
                    size = size +len(nodes)
                else:
                    output[size:target] = nodes[:(target-size)]
                    size = size + target - size
            khop[("hop{}".format(i+1))] = output
        return khop
