from math import ceil
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GINConv, MLP, DenseGINConv, PANConv
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import PANPooling, SAGPooling, ASAPooling, EdgePooling, graclus
from torch_geometric.nn.pool.select import SelectTopK
from torch_geometric.nn.pool.connect import FilterEdges
from torch_geometric.nn.pool import TopKPooling
from torch_geometric.nn import dense_mincut_pool, dense_diff_pool, DMoNPooling
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import to_dense_adj, to_dense_batch, dense_to_sparse
from torch_geometric.data import Data

from scripts.sum_pool import sum_pool
from scripts.pooling.kmis.kmis_pool import KMISPooling
from scripts.pooling.rnd_sparse import RndSparse
from scripts.utils import batched_negative_edges


class GIN_Dual_Pool_Net(torch.nn.Module):
    def __init__(self, 
                 in_channels,           # Size of node features
                 out_channels,          # Number of classes
                 num_layers_pre=1,      # Number of GIN layers before first pooling
                 num_layers_mid=1,      # Number of GIN layers between poolings
                 num_layers_post=1,     # Number of GIN layers after second pooling
                 hidden_channels=64,    # Dimensionality of node embeddings
                 norm=True,             # Normalise Layers in the GIN MLP
                 activation='ELU',      # Activation of the MLP in GIN 
                 average_nodes=None,    # Needed for dense pooling methods
                 max_nodes=None,        # Needed for random pool
                 pooling1=None,         # First pooling method
                 pooling2=None,         # Second pooling method
                 pool_ratio1=0.5,       # Ratio for first pooling
                 pool_ratio2=0.2,       # Ratio for second pooling
                 ):
        super(GIN_Dual_Pool_Net, self).__init__()
        
        self.num_layers_pre = num_layers_pre
        self.num_layers_mid = num_layers_mid
        self.num_layers_post = num_layers_post
        self.hidden_channels = hidden_channels
        self.act = activation_resolver(activation)
        self.pooling1 = pooling1
        self.pooling2 = pooling2
        self.pool_ratio1 = pool_ratio1
        self.pool_ratio2 = pool_ratio2
        
        # Track if we're using dense pooling methods
        self.is_dense1 = pooling1 in ['diffpool', 'mincut', 'dmon', 'dense-random']
        self.is_dense2 = pooling2 in ['diffpool', 'mincut', 'dmon', 'dense-random']
        
        # Pre-pooling block (before first pooling)            
        self.conv_layers_pre = torch.nn.ModuleList()
        if pooling1 == 'panpool':
            for _ in range(num_layers_pre):
                self.conv_layers_pre.append(PANConv(in_channels, hidden_channels, filter_size=2))
                in_channels = hidden_channels
        else:
            for _ in range(num_layers_pre):
                mlp = MLP([in_channels, hidden_channels, hidden_channels], act=activation)
                self.conv_layers_pre.append(GINConv(nn=mlp, train_eps=False))
                in_channels = hidden_channels
        
        # First pooling block
        pooled_nodes1 = ceil(pool_ratio1 * average_nodes) if average_nodes else None
        self.pool1 = self._init_pooling_layer(pooling1, hidden_channels, pooled_nodes1, pool_ratio1, max_nodes)
        
        # Mid-pooling block (between first and second pooling)
        self.conv_layers_mid = torch.nn.ModuleList()
        for _ in range(num_layers_mid):
            mlp = MLP([hidden_channels, hidden_channels, hidden_channels], act=activation, norm=None)
            if self.is_dense1:
                self.conv_layers_mid.append(DenseGINConv(nn=mlp, train_eps=False))
            else:
                self.conv_layers_mid.append(GINConv(nn=mlp, train_eps=False))
        
        # Second pooling block
        pooled_nodes2 = ceil(pool_ratio2 * pooled_nodes1) if average_nodes else None
        self.pool2 = self._init_pooling_layer(pooling2, hidden_channels, pooled_nodes2, pool_ratio2, max_nodes)
        
        # Post-pooling block (after second pooling)
        self.conv_layers_post = torch.nn.ModuleList()
        for _ in range(num_layers_post):
            mlp = MLP([hidden_channels, hidden_channels, hidden_channels], act=activation, norm=None)
            if self.is_dense2:  # Use DenseGINConv if second pooling is dense
                self.conv_layers_post.append(DenseGINConv(nn=mlp, train_eps=False))
            else:
                self.conv_layers_post.append(GINConv(nn=mlp, train_eps=False))

        # Readout
        self.mlp = MLP([hidden_channels, hidden_channels, hidden_channels//2, out_channels], 
                        act=activation,
                        norm=None,
                        dropout=0.5)

    def _init_pooling_layer(self, pooling_type, hidden_channels, pooled_nodes, pool_ratio, max_nodes):
        """Helper method to initialize a pooling layer based on type"""
        if pooling_type is None:
            return None
            
        if pooling_type == 'diffpool' or pooling_type == 'mincut':
            return Linear(hidden_channels, pooled_nodes)
        elif pooling_type == 'dmon':
            return DMoNPooling(hidden_channels, pooled_nodes)
        elif pooling_type == 'dense-random':
            s_rnd = torch.randn(max_nodes, pooled_nodes)
            s_rnd.requires_grad = False
            return s_rnd
        elif pooling_type == 'topk':
            return TopKPooling(hidden_channels, ratio=pool_ratio)
        elif pooling_type == 'panpool':
            return PANPooling(hidden_channels, ratio=pool_ratio)
        elif pooling_type == 'sagpool':
            return SAGPooling(hidden_channels, ratio=pool_ratio)
        elif pooling_type == 'asapool':
            return ASAPooling(hidden_channels, ratio=pool_ratio)
        elif pooling_type == 'edgepool':
            return EdgePooling(hidden_channels)
        elif pooling_type == 'kmis':
            return KMISPooling(hidden_channels, k=5, aggr_x='sum')
        elif pooling_type == 'sparse-random':
            return RndSparse(in_channels=hidden_channels, ratio=pool_ratio, max_nodes=max_nodes)
        elif pooling_type == 'graclus':
            pass
        elif pooling_type == 'comp-graclus':
            pass
        else:
            raise KeyError(f"Unrecognized pooling method: {pooling_type}")

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            try:
                if isinstance(module, torch.Tensor):
                    continue  # Skip tensors like s_rnd
                module.reset_parameters()
            except AttributeError:
                if name != 'act':
                    for x in module:
                        x.reset_parameters()

    def _apply_pooling(self, pooling_type, pool_layer, x, adj, batch, mask=None):
        """Helper method to apply a pooling operation based on type"""
        aux_loss = 0
        
        if pooling_type in ['diffpool', 'mincut', 'dmon', 'dense-random']:
            if pooling_type == 'diffpool':
                s = pool_layer(x)
                x, adj, l1, l2 = dense_diff_pool(x, adj, s, mask)
                aux_loss = 0.1*l1 + 0.1*l2
            elif pooling_type == 'mincut':
                s = pool_layer(x)
                x, adj, l1, l2 = dense_mincut_pool(x, adj, s, mask)
                aux_loss = 0.5*l1 + l2
            elif pooling_type == 'dmon':
                _, x, adj, l1, l2, l3 = pool_layer(x, adj, mask)
                aux_loss = 0.3*l1 + 0.3*l2 + 0.3*l3
            elif pooling_type == 'dense-random':
                s = pool_layer[:x.size(1), :].unsqueeze(dim=0).expand(x.size(0), -1, -1).to(x.device)
                x, adj, _, _ = dense_diff_pool(x, adj, s, mask)
            return x, adj, batch, mask, aux_loss
            
        elif pooling_type in ['topk', 'sagpool', 'sparse-random']:
            x, adj, _, batch, _, _ = pool_layer(x, adj, edge_attr=None, batch=batch)
        elif pooling_type == 'asapool':
            x, adj, _, batch, _ = pool_layer(x, adj, batch=batch)
        # elif pooling_type == 'panpool':
        #     # Ensure adj is sparse (edge_index); if dense, convert it
        #     if adj.dim() == 3:  # Dense adj: (batch_size, num_nodes, num_nodes)
        #         adj = dense_to_sparse(adj)[0]  # Convert to edge_index
        #     x, adj, _, batch, _, _ = pool_layer(x, adj, batch=batch)
        elif pooling_type == 'panpool':
            if adj.dim() == 2:  # adj is edge_index
                num_nodes = x.size(0)
                adj = torch.sparse_coo_tensor(adj, torch.ones(adj.size(1), device=adj.device), (num_nodes, num_nodes))
            elif adj.dim() == 3:  # adj is dense
                adj = adj.to_sparse()
                x, adj, _, batch, _, _ = pool_layer(x, adj, batch=batch)
        elif pooling_type == 'edgepool':
            x, adj, batch, _ = pool_layer(x, adj, batch=batch)
        elif pooling_type == 'kmis':
            x, adj, _, batch, _, _ = pool_layer(x, adj, None, batch=batch)
        elif pooling_type in ['graclus', 'comp-graclus']:
            temp_data = Data(x=x, edge_index=adj, batch=batch, edge_attr=None)
            
            if pooling_type == 'graclus':
                cluster = graclus(adj, num_nodes=x.size(0))
            else:
                complement = batched_negative_edges(edge_index=adj, batch=batch, force_undirected=True)
                cluster = graclus(complement, num_nodes=x.size(0))
            
            pooled_data = sum_pool(cluster, temp_data)
            x = pooled_data.x
            adj = pooled_data.edge_index
            batch = pooled_data.batch
            
        return x, adj, batch, None, aux_loss

    def forward(self, data):
        x = data.x    
        adj = data.edge_index
        batch = data.batch
        total_aux_loss = 0
        
        ### pre-pooling block (before first pooling)
        if self.pooling1 == 'panpool':
            M = adj  # Store original adjacency as M for PANConv
            for layer in self.conv_layers_pre:  
                x, M = layer(x, M)
                x = self.act(x)
            adj = M  # Update adj for panpool
        else:
            for layer in self.conv_layers_pre:  
                x = self.act(layer(x, adj))
    
        ### first pooling block
        if self.pooling1 is not None:
            # Convert to dense representation if needed for first pooling
            if self.is_dense1:
                x, mask = to_dense_batch(x, batch)
                adj = to_dense_adj(adj, batch)
                x, adj, batch, mask, aux_loss = self._apply_pooling(self.pooling1, self.pool1, x, adj, batch, mask)
            else:
                x, adj, batch, _, aux_loss = self._apply_pooling(self.pooling1, self.pool1, x, adj, batch)
            total_aux_loss += aux_loss
                
        ### mid-pooling block (between poolings)
        for layer in self.conv_layers_mid:
            x = self.act(layer(x, adj))
            
        ### second pooling block
        if self.pooling2 is not None:
            # Convert to dense representation if needed for second pooling
            # but only if first pooling wasn't already dense
            if self.is_dense2 and not self.is_dense1:
                x, mask = to_dense_batch(x, batch)
                adj = to_dense_adj(adj, batch)
                x, adj, batch, mask, aux_loss = self._apply_pooling(self.pooling2, self.pool2, x, adj, batch, mask)
            else:
                x, adj, batch, mask, aux_loss = self._apply_pooling(self.pooling2, self.pool2, x, adj, batch, mask if self.is_dense1 else None)
            total_aux_loss += aux_loss
                
        ### post-pooling block
        for layer in self.conv_layers_post:  
            x = self.act(layer(x, adj))

        ### readout
        if self.is_dense2:  # If second pooling is dense, x is [batch_size, num_nodes, num_features]
            x = torch.sum(x, dim=1)  # Sum over node dimension to get [batch_size, num_features]
        else:
            x = global_add_pool(x, batch)  # Use global add pooling for sparse format [num_nodes, num_features]
            
        x = self.mlp(x)
        
        return F.log_softmax(x, dim=-1), total_aux_loss