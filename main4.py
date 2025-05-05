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
parser.add_argument('--poolings', type=str, default="none",
    help="Comma-separated list of pooling methods to evaluate; 'none' maps to no pooling.")
parser.add_argument('--pool_ratio',    type=float, default=0.1)
parser.add_argument('--batch_size',    type=int,   default=32)
parser.add_argument('--num_layers_pre',   type=int,   default=2)
parser.add_argument('--num_layers_post',  type=int,   default=1)
parser.add_argument('--lr',               type=float, default=1e-4)
parser.add_argument('--epochs',           type=int,   default=200)
parser.add_argument('--runs',             type=int,   default=3)
args = parser.parse_args()

# parse lists
hidden_sizes = [int(h) for h in args.hidden_sizes.split(',') if h.strip()]
raw_pools = [p.strip() for p in args.poolings.split(',') if p.strip()]
poolings = [None if p.lower() in ('none','nopool','no-pool','') else p for p in raw_pools]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rng = np.random.default_rng(1)

# -------------------------------------------------------------------
# 2. Dataset setup
# -------------------------------------------------------------------
dataset = EXPWL1Dataset("data/EXPWL1/", transform=DataToFloat())
avg_nodes = int(dataset.data.num_nodes / len(dataset))
max_nodes = max(d.num_nodes for d in dataset)
max_nodes_sparse = max_nodes * args.batch_size

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
# 4. Sweep over pooling methods and hidden sizes
# -------------------------------------------------------------------
for pool_method in poolings:
    name = str(pool_method) if pool_method is not None else "no-pool"
    print(f"\n=== Pooling: {name} ===")

    hid_scores = []  # mean best val acc per hidden size
    for hid in hidden_sizes:
        per_run_best = []
        for run in range(args.runs):
            # split data
            idx = rng.permutation(len(dataset))
            ds = dataset[idx]
            n = len(ds)
            val_ds = ds[:n//5]
            train_ds = ds[n//5:]

            # loaders
            train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)
            val_loader   = DataLoader(val_ds,   args.batch_size)

            # model
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

            # train and track validation
            best_val_acc = 0.0
            for epoch in range(1, args.epochs+1):
                _ = train_epoch(model, train_loader, opt)
                va_acc, _ = eval_epoch(model, val_loader)
                best_val_acc = max(best_val_acc, va_acc)

            per_run_best.append(best_val_acc)

        # average across runs
        mean_best = float(np.mean(per_run_best))
        hid_scores.append(mean_best)
        print(f"  hid={hid}: mean best val acc = {mean_best:.4f}")

    # plot hidden size vs validation accuracy for this pooling
    plt.figure(figsize=(6,4))
    plt.plot(hidden_sizes, hid_scores, marker='o')
    plt.title(f"Best Validation Acc vs Hidden Size (pool={name})")
    plt.xlabel("Hidden Channel Size")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"val_acc_vs_hidden_{name}.png", bbox_inches='tight')
    plt.close()
    print(f"Plot saved to val_acc_vs_hidden_{name}.png")