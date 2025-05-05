# main.py - Double Pooling Version

import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader
from scripts.nn_model2 import GIN_Dual_Pool_Net  # Modified version for double pooling
from scripts.utils import EXPWL1Dataset, DataToFloat, log

# -------------------------------------------------------------------
# 1. Arguments
# -------------------------------------------------------------------
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument('--first_poolings', type=str,
    default="diffpool,dmon,mincut,ecpool,graclus,kmis,topk,panpool,asapool,sagpool",
    help="Comma-separated list of first pooling methods"
)
parser.add_argument('--second_poolings', type=str,
    default="diffpool,dmon,mincut,ecpool,graclus,kmis,topk,panpool,asapool,sagpool",
    help="Comma-separated list of second pooling methods"
)
parser.add_argument('--pool_combinations', type=str, default="",
    help="Comma-separated specific pool1+pool2 combinations to test (e.g., 'diffpool+sagpool,topk+graclus')"
)
parser.add_argument('--first_pool_ratio',  type=float, default=0.5)
parser.add_argument('--second_pool_ratio', type=float, default=0.2)
parser.add_argument('--batch_size',        type=int,   default=32)
parser.add_argument('--hidden_channels',   type=int,   default=64)
parser.add_argument('--num_layers_pre',    type=int,   default=2)
parser.add_argument('--num_layers_mid',    type=int,   default=1)
parser.add_argument('--num_layers_post',   type=int,   default=1)
parser.add_argument('--lr',                type=float, default=1e-4)
parser.add_argument('--epochs',            type=int,   default=500)
parser.add_argument('--runs',              type=int,   default=1)
parser.add_argument('--top_k',             type=int,   default=5,
    help="Number of top-performing combinations to plot in detail"
)
args = parser.parse_args()

# Parse pooling methods
first_pools = [p.strip() for p in args.first_poolings.split(',') if p.strip()]
second_pools = [p.strip() for p in args.second_poolings.split(',') if p.strip()]

# Parse explicit combinations if provided
if args.pool_combinations:
    pool_combos = []
    for combo in args.pool_combinations.split(','):
        if '+' in combo:
            p1, p2 = combo.split('+', 1)
            pool_combos.append((p1.strip(), p2.strip()))
        else:
            print(f"Warning: Ignoring invalid combination format: {combo}")
else:
    # Generate all combinations
    pool_combos = [(p1, p2) for p1 in first_pools for p2 in second_pools]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rng = np.random.default_rng(1)

# -------------------------------------------------------------------
# 2. Dataset setup
# -------------------------------------------------------------------
dataset = EXPWL1Dataset("data/EXPWL1/", transform=DataToFloat())
avg_nodes = int(dataset.data.num_nodes / len(dataset))
max_nodes = max(d.num_nodes for d in dataset)

# For cumulative plotting across poolings
all_combo_epoch_tests = {}

# -------------------------------------------------------------------
# 3. Helpers
# -------------------------------------------------------------------
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, aux = model(data)
        loss = F.nll_loss(out, data.y) + aux
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    correct = 0
    total_loss = 0
    for data in loader:
        data = data.to(device)
        out, _ = model(data)
        total_loss += float(F.nll_loss(out, data.y)) * data.num_graphs
        correct += int((out.argmax(-1) == data.y).sum())
    return correct / len(loader.dataset), total_loss / len(loader.dataset)

# -------------------------------------------------------------------
# 4. Sweep over pooling combinations
# -------------------------------------------------------------------
results = []
for pool1, pool2 in pool_combos:
    combo_name = f"{pool1}+{pool2}"
    print(f"\n=== Running double pooling: {combo_name} ===")
    all_combo_epoch_tests[combo_name] = []

    per_run_test_acc = []
    per_run_times = []

    for run in range(args.runs):
        idx = rng.permutation(len(dataset))
        ds = dataset[idx]
        train_ds = ds[len(ds)//5:]
        val_ds = ds[:len(ds)//10]
        test_ds = ds[len(ds)//10:len(ds)//5]

        train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, args.batch_size)
        test_loader = DataLoader(test_ds, args.batch_size)

        model = GIN_Dual_Pool_Net(
            in_channels=train_ds.num_features,
            out_channels=train_ds.num_classes,
            num_layers_pre=args.num_layers_pre,
            num_layers_mid=args.num_layers_mid,
            num_layers_post=args.num_layers_post,
            hidden_channels=args.hidden_channels,
            average_nodes=avg_nodes,
            pooling1=pool1,
            pooling2=pool2,
            pool_ratio1=args.first_pool_ratio,
            pool_ratio2=args.second_pool_ratio,
            max_nodes=max_nodes
        ).to(device)
        
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val_loss = np.inf
        best_test_acc = 0.0

        epoch_times = []
        epoch_tests = []

        for epoch in range(1, args.epochs+1):
            t0 = time.time()
            loss = train_epoch(model, train_loader, opt)
            t_ep = time.time() - t0

            tr_acc, _ = eval_epoch(model, train_loader)
            va_acc, va_loss = eval_epoch(model, val_loader)
            te_acc, _ = eval_epoch(model, test_loader)

            epoch_times.append(t_ep)
            epoch_tests.append(te_acc)

            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_test_acc = te_acc

            # log every epoch with time and some epoch ticks
            log(
                Pool=combo_name, Run=run+1, Epoch=epoch,
                Loss=f"{loss:.4f}", Train=f"{tr_acc:.3f}",
                Val=f"{va_acc:.3f}", Test=f"{te_acc:.3f}", Time=f"{t_ep:.2f}s"
            )

        per_run_test_acc.append(best_test_acc)
        per_run_times.append(np.mean(epoch_times))
        all_combo_epoch_tests[combo_name].append(epoch_tests)

        # Individual plots with epoch annotations
        # Accuracy vs Epoch
        plt.figure(figsize=(4,3))
        plt.plot(range(1, args.epochs+1), epoch_tests, marker='o', markevery=[0, args.epochs//2, args.epochs-1])
        for pt in (0, args.epochs//2, args.epochs-1):
            plt.annotate(str(pt+1), xy=(pt+1, epoch_tests[pt]), xytext=(5,5), textcoords='offset points')
        plt.title(f"{combo_name} — Run {run+1}")
        plt.xlabel("Epoch")
        plt.ylabel("Test Acc")
        plt.grid(True)
        plt.savefig(f"{combo_name}_run{run+1}_epoch_acc.png", bbox_inches='tight')
        plt.close()

        # Cumulative Time vs Accuracy with epoch markers
        cum_time = np.cumsum(epoch_times)
        plt.figure(figsize=(4,3))
        plt.plot(cum_time, epoch_tests, marker='s', markevery=[0, len(cum_time)//2, len(cum_time)-1])
        for idx in (0, len(cum_time)//2, len(cum_time)-1):
            plt.annotate(str(idx+1), xy=(cum_time[idx], epoch_tests[idx]), xytext=(5,-5), textcoords='offset points')
        plt.title(f"{combo_name} — Run {run+1}")
        plt.xlabel("Cumulative Time (s)")
        plt.ylabel("Test Acc")
        plt.grid(True)
        plt.savefig(f"{combo_name}_run{run+1}_time_acc.png", bbox_inches='tight')
        plt.close()

    # Aggregate results
    mean_acc = np.mean(per_run_test_acc)
    std_acc = np.std(per_run_test_acc)
    mean_t = np.mean(per_run_times)
    expressive = '✓' if mean_acc > 0.95 else '✗'
    
    results.append({
        "Pooling": combo_name,
        "s/epoch": f"{mean_t:.2f}s",
        "GIN layers": f"{args.num_layers_pre}+{args.num_layers_mid}+{args.num_layers_post}",
        "Pool Ratios": f"{args.first_pool_ratio:.1f},{args.second_pool_ratio:.1f}",
        "Test Acc": f"{mean_acc*100:.1f}±{std_acc*100:.1f}",
        "Expressive": expressive
    })

# Sort results by test accuracy for reporting
sorted_results = sorted(results, key=lambda x: float(x["Test Acc"].split('±')[0]), reverse=True)

# -------------------------------------------------------------------
# 5. Cumulative Plot: Avg Test Acc vs Epoch for Top K Combinations
# -------------------------------------------------------------------
plt.figure(figsize=(10,6))
# Get the top K performing combinations
top_k = min(args.top_k, len(sorted_results))
top_combos = [r["Pooling"] for r in sorted_results[:top_k]]

for name in top_combos:
    if name in all_combo_epoch_tests:
        avg_curve = np.mean(all_combo_epoch_tests[name], axis=0)
        plt.plot(range(1, args.epochs+1), avg_curve, label=name)

plt.title(f"Average Test Accuracy per Epoch for Top {top_k} Double Pooling Combinations")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig("top_double_poolings_epoch_acc.png", bbox_inches='tight')
plt.close()

# Generate a 2D heatmap of first vs second pooling performance
if not args.pool_combinations:  # Only if we tested all combinations
    # Create matrix for heatmap
    first_unique = sorted(set(first_pools))
    second_unique = sorted(set(second_pools))
    
    perf_matrix = np.zeros((len(first_unique), len(second_unique)))
    for i, p1 in enumerate(first_unique):
        for j, p2 in enumerate(second_unique):
            combo = f"{p1}+{p2}"
            for r in results:
                if r["Pooling"] == combo:
                    perf_matrix[i, j] = float(r["Test Acc"].split('±')[0])
                    break
    
    plt.figure(figsize=(10, 8))
    plt.imshow(perf_matrix, cmap='viridis')
    plt.colorbar(label='Test Accuracy (%)')
    plt.xticks(range(len(second_unique)), second_unique, rotation=45)
    plt.yticks(range(len(first_unique)), first_unique)
    plt.xlabel('Second Pooling Method')
    plt.ylabel('First Pooling Method')
    plt.title('Test Accuracy for Double Pooling Combinations')
    
    # Add text annotations
    for i in range(len(first_unique)):
        for j in range(len(second_unique)):
            plt.text(j, i, f"{perf_matrix[i, j]:.1f}", 
                     ha="center", va="center", 
                     color="white" if perf_matrix[i, j] < 70 else "black")
    
    plt.tight_layout()
    plt.savefig("double_pooling_heatmap.png", bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------------
# 6. Print summary table
# -------------------------------------------------------------------
print("\n" + "-"*90)
print(f"{'Pooling':<20} | {'s/epoch':<8} | {'GIN layers':<12} | {'Pool Ratios':<12} | {'Test Acc':<12} | {'Expr.'}")
print("-"*90)
for r in sorted_results:
    print(f"{r['Pooling']:<20} | {r['s/epoch']:<8} | {r['GIN layers']:<12} | "
          f"{r['Pool Ratios']:<12} | {r['Test Acc']:<12} | {r['Expressive']}")
print("-"*90)

# -------------------------------------------------------------------
# 7. Generate a template for DoubleGIN_Pool_Net class structure
# -------------------------------------------------------------------
with open("scripts/nn_model_double_template.py", "w") as f:
    f.write("""# Template for DoubleGIN_Pool_Net implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool

class DoubleGIN_Pool_Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels,
                 num_layers_pre, num_layers_mid, num_layers_post,
                 average_nodes, pooling1, pooling2, 
                 pool_ratio1, pool_ratio2, max_nodes):
        super().__init__()
        
        # Parameters
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.pooling1 = pooling1
        self.pooling2 = pooling2
        self.pool_ratio1 = pool_ratio1
        self.pool_ratio2 = pool_ratio2
        
        # Pre-pooling GIN layers
        self.pre_convs = nn.ModuleList()
        # First layer: in_channels -> hidden_channels
        self.pre_convs.append(GINConv(
            nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ), train_eps=True)
        )
        # Remaining pre-pooling layers
        for _ in range(num_layers_pre - 1):
            self.pre_convs.append(GINConv(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                ), train_eps=True)
            )
        
        # First pooling layer (implemented based on pooling1 type)
        # TODO: Implement pooling1 based on type
        
        # Mid-level GIN layers (between pooling operations)
        self.mid_convs = nn.ModuleList()
        for _ in range(num_layers_mid):
            self.mid_convs.append(GINConv(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                ), train_eps=True)
            )
        
        # Second pooling layer (implemented based on pooling2 type)
        # TODO: Implement pooling2 based on type
        
        # Post-pooling GIN layers
        self.post_convs = nn.ModuleList()
        for _ in range(num_layers_post):
            self.post_convs.append(GINConv(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                ), train_eps=True)
            )
        
        # Output layer
        self.lin = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Pre-pooling convolutions
        for conv in self.pre_convs:
            x = conv(x, edge_index)
        
        # First pooling operation
        # TODO: Apply first pooling based on self.pooling1
        # x, edge_index, batch, cluster, aux_loss1 = apply_pooling(self.pooling1, x, edge_index, batch, self.pool_ratio1)
        aux_loss1 = 0  # Placeholder
        
        # Mid-level convolutions
        for conv in self.mid_convs:
            x = conv(x, edge_index)
        
        # Second pooling operation
        # TODO: Apply second pooling based on self.pooling2
        # x, edge_index, batch, cluster, aux_loss2 = apply_pooling(self.pooling2, x, edge_index, batch, self.pool_ratio2)
        aux_loss2 = 0  # Placeholder
        
        # Post-pooling convolutions
        for conv in self.post_convs:
            x = conv(x, edge_index)
        
        # Global pooling and classification
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return F.log_softmax(x, dim=-1), aux_loss1 + aux_loss2
""")

print(f"\nA template for DoubleGIN_Pool_Net has been saved to scripts/nn_model_double_template.py")

# -------------------------------------------------------------------
# 8. (Colab) Load & Display Saved Plots
# -------------------------------------------------------------------
# In a Colab cell:
# from IPython.display import Image, display
# import glob
# # Top combinations
# display(Image('top_double_poolings_epoch_acc.png'))
# if os.path.exists('double_pooling_heatmap.png'):
#     display(Image('double_pooling_heatmap.png'))
# # Individual plots for top 3 combinations:
# for combo in [r["Pooling"] for r in sorted_results[:3]]:
#     for img in sorted(glob.glob(f"{combo}_*_epoch_acc.png")):
#         display(Image(img))