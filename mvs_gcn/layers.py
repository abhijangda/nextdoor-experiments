from utils import *

class GraphConvolution(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(GraphConvolution, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.reset_parameters()

    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.spmm(adj, x)
        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)
            
            
class GraphSageConvolution(nn.Module):
    def __init__(self, n_in, n_out, use_lynorm=True, bias=True):
        super(GraphSageConvolution, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out, bias=bias)
        self.reset_parameters()
        
        if use_lynorm:
            self.lynorm = nn.LayerNorm(2*n_out, elementwise_affine=True)
        else:
            self.lynorm = lambda x: x

    def forward(self, x, adj):
        out_node_num = adj.size(0)
        x = self.linear(x)
        support = torch.spmm(adj, x)
        x = torch.cat([x[:out_node_num], support], dim=1)
        x = self.lynorm(x)
        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)


class SimplifiedGraphConvolution(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(SimplifiedGraphConvolution, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        
    def forward(self, x, adjs):
        for adj in adjs:
            x = torch.spmm(adj, x)
        return x