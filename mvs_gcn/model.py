from utils import *
from layers import GraphConvolution, GraphSageConvolution
import autograd_wl 

##########################################
##########################################
##########################################
class Net(nn.Module):
    def __init__(self, nfeat, nhid, num_classes, layers, dropout, multi_class):
        super(Net, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.multi_class = multi_class

        self.gcs = None
        self.gc_out = None
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        if multi_class:
            self.loss_f = nn.BCEWithLogitsLoss()
            self.loss_f_vec = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss_f = nn.CrossEntropyLoss()
            self.loss_f_vec = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, adjs):
        for ell in range(len(self.gcs)):
            x = self.gcs[ell](x, adjs[ell])
            x = self.relu(x)
            x = self.dropout(x)
        x = self.gc_out(x)
        return x

    def partial_grad(self, x, adjs, targets, weight=None):
        outputs = self.forward(x, adjs)
        if weight is None:
            loss = self.loss_f(outputs, targets)
        else:
            if self.multi_class:
                loss = self.loss_f_vec(outputs, targets)
                loss = loss.mean(1) * weight
            else:
                loss = self.loss_f_vec(outputs, targets) * weight
            loss = loss.sum()
        loss.backward()
        return loss.detach()
    
    def partial_grad_with_norm(self, x, adjs, targets, weight):
        num_samples = targets.size(0)
        outputs = self.forward(x, adjs)
        
        if self.multi_class:
            loss = self.loss_f_vec(outputs, targets)
            loss = loss.mean(1) * weight
        else:
            loss = self.loss_f_vec(outputs, targets) * weight
        loss = loss.sum()
        loss.backward()
        grad_per_sample = autograd_wl.calculate_sample_grad(self.gc_out)
        
        grad_per_sample = grad_per_sample*(1/weight/num_samples)
        return loss.detach(), grad_per_sample.cpu().numpy()

    def calculate_sample_grad(self, x, adjs, targets, batch_nodes):
        # use smart way
        outputs = self.forward(x, adjs)
        loss = self.loss_f(outputs, targets[batch_nodes])
        loss.backward()
        grad_per_sample = autograd_wl.calculate_sample_grad(self.gc_out)

        return grad_per_sample.cpu().numpy()
    
    def calculate_loss_grad(self, x, adjs, targets, batch_nodes):
        outputs = self.forward(x, adjs)
        loss = self.loss_f(outputs[batch_nodes], targets[batch_nodes])
        loss.backward()
        return loss.detach()

    def calculate_f1(self, x, adjs, targets, batch_nodes):
        outputs = self.forward(x, adjs)
        if self.multi_class:
            outputs[outputs > 0] = 1
            outputs[outputs <= 0] = 0
        else:
            outputs = outputs.argmax(dim=1)
        return f1_score(outputs[batch_nodes].cpu().detach(), targets[batch_nodes].cpu().detach(), average="micro")
    
"""
This is a plain implementation of GCN
Used for FastGCN, LADIES
"""

class GCN(Net):
    def __init__(self, nfeat, nhid, num_classes, layers, dropout, multi_class):
        super(Net, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.multi_class = multi_class

        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat,  nhid))
        for _ in range(layers-1):
            self.gcs.append(GraphConvolution(nhid,  nhid))
        self.gc_out = nn.Linear(nhid, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.gc_out.register_forward_hook(autograd_wl.capture_activations)
        self.gc_out.register_backward_hook(autograd_wl.capture_backprops)

        if multi_class:
            self.loss_f = nn.BCEWithLogitsLoss()
            self.loss_f_vec = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss_f = nn.CrossEntropyLoss()
            self.loss_f_vec = nn.CrossEntropyLoss(reduction='none')

class GraphSageGCN(Net):
    def __init__(self, nfeat, nhid, num_classes, layers, dropout, multi_class):
        super(Net, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.multi_class = multi_class

        self.gcs = nn.ModuleList()
        self.gcs.append(GraphSageConvolution(nfeat, nhid, use_lynorm=False))
        for _ in range(layers-1):
            self.gcs.append(GraphSageConvolution(2*nhid,  nhid, use_lynorm=False))
        self.gc_out = nn.Linear(2*nhid,  num_classes) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.gc_out.register_forward_hook(autograd_wl.capture_activations)
        self.gc_out.register_backward_hook(autograd_wl.capture_backprops)
        
        if multi_class:
            self.loss_f = nn.BCEWithLogitsLoss()
            self.loss_f_vec = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss_f = nn.CrossEntropyLoss()
            self.loss_f_vec = nn.CrossEntropyLoss(reduction='none')

