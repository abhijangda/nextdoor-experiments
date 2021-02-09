from utils import *
import argparse
"""
Dataset arguments
"""
parser = argparse.ArgumentParser(
    description='Training GCN on Large-scale Graph Datasets')

parser.add_argument('--dataset', type=str, default='yelp',
                    help='Dataset name: cora/citeseer/pubmed/flickr/reddit/ppi/ppi-large')
parser.add_argument('--nhid', type=int, default=256,
                    help='Hidden state dimension')
parser.add_argument('--epoch_num', type=int, default=400,
                    help='Number of Epoch')
parser.add_argument('--pool_num', type=int, default=10,
                    help='Number of Pool')
parser.add_argument('--batch_num', type=int, default=20,
                    help='Maximum Batch Number')
parser.add_argument('--batch_size', type=int, default=512,
                    help='size of output node in a batch')
parser.add_argument('--n_layers', type=int, default=2,
                    help='Number of GCN layers')
parser.add_argument('--n_stops', type=int, default=1000,
                    help='Stop after number of batches that f1 dont increase')
parser.add_argument('--samp_num', type=int, default=512,
                    help='Number of sampled nodes per layer (only for ladies & factgcn)')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate')
parser.add_argument('--cuda', type=int, default=-1,
                    help='Avaiable GPU ID')
parser.add_argument('--is_ratio', type=float, default=1.0,
                    help='Importance sampling rate')
parser.add_argument('--show_grad_norm', type=int, default=0,
                    help='Whether show gradient norm 0-False, 1-True')
parser.add_argument('--cluster_bsize', type=int, default=5,
                    help='how many cluster selected each mini-batch')
args = parser.parse_args()
print(args)

# pubmed n_layers 2