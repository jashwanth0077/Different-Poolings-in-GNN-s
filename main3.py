import argparse
import time
import numpy as np
import torch
torch.manual_seed(1)
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader
from scripts.nn_model import GIN_Pool_Net
from scripts.utils    import EXPWL1Dataset, DataToFloat, log

# -------------------------------------------------------------------
# 1. Arguments
# -------------------------------------------------------------------
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument('--hidden_sizes', type=str,
    default="32,64,128,256",
    help="Comma-separated list of hidden channel sizes to sweep (e.g., 32,64,128)")
parser.add_argument('--pooling', type=str, default="none",
    help="Pooling method to use; 'none' for no pooling.")
parser.add_argument('--pool_ratio',    type=float, default=0.1)
parser.add_argument('--batch_size',    type=int,   default=32)
parser.add_argument('--num_layers_pre',   type=int,   default=2)
parser.add_argument('--num_layers_post',  type=int,   default=1)
parser.add_argument('--lr',               type=float, default=1e-4)
parser.add_argument('--epochs',           type=int,   default=200)
parser.add_argument('--runs',             type=int,   default=3)
args = parser.parse_args()

# parse hidden sizes list
raw = [int(h) for h in args.hidden_sizes.split(',') if h.strip()]
hidden_sizes = raw
pool_method = None if args.pooling.lower() in ('none','nopool') else args.pooling

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rng = np.random.default_rng(1)

# -------------------------------------------------------------------
# 2. Dataset setup
# -------------------------------------------------------------------
dataset = EXPWL1Dataset("data/EXPWL1/", transform=DataToFloat())
avg_nodes = int(dataset.data.num_nodes / len(dataset))  
max_nodes = max(d.num_nodes for d in dataset)
max_nodes_sparse = max_nodes * args.batch_size

# store per-hidden-size, per-epoch validation curves
all_hidden_val = {str(h): [] for h in hidden_sizes}

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
    correct = total_loss = 0
    for data in loader:
        data = data.to(device)
        out, _ = model(data)
        total_loss += float(F.nll_loss(out, data.y)) * data.num_graphs
        correct += int((out.argmax(-1) == data.y).sum())
    return correct / len(loader.dataset), total_loss / len(loader.dataset)

# -------------------------------------------------------------------
# 4. Sweep hidden sizes
# -------------------------------------------------------------------
for hid in hidden_sizes:
    print(f"\n=== Hidden size: {hid} ===")
    per_run_val = []
    for run in range(args.runs):
        idx = rng.permutation(len(dataset))
        ds = dataset[idx]
        n = len(ds)
        val_ds = ds[:n//5]
        train_ds = ds[n//5:]

        train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   args.batch_size)

        mn = max_nodes_sparse if pool_method == "sparse-random" else max_nodes
        model = GIN_Pool_Net(
            in_channels     = train_ds.num_features,
            out_channels    = train_ds.num_classes,
            num_layers_pre  = args.num_layers_pre,
            num_layers_post = args.num_layers_post,
            hidden_channels = hid,
            average_nodes   = avg_nodes,
            pooling         = pool_method,
            pool_ratio      = args.pool_ratio,
            max_nodes       = mn
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val_acc = 0.0
        val_curve = []
        for epoch in range(1, args.epochs+1):
            _ = train_epoch(model, train_loader, opt)
            va_acc, va_loss = eval_epoch(model, val_loader)
            val_curve.append(va_acc)
        all_hidden_val[str(hid)].append(val_curve)

    # average across runs
    avg_curve = np.mean(all_hidden_val[str(hid)], axis=0)
    plt.plot(range(1, args.epochs+1), avg_curve, label=f"hid={hid}")

# -------------------------------------------------------------------
# 5. Plot and save
# -------------------------------------------------------------------
plt.figure(figsize=(6,4))
for hid, runs in all_hidden_val.items():
    avg_curve = np.mean(runs, axis=0)
    plt.plot(range(1, args.epochs+1), avg_curve, label=f"hid={hid}")
plt.title("Validation Accuracy per Epoch Across Hidden Sizes")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig("hidden_size_val_acc.png", bbox_inches='tight')
plt.close()

print("Plot saved to hidden_size_val_acc.png")
