# main.py

import argparse
import time
import numpy as np
import torch
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
parser.add_argument('--poolings', nargs='+',
    default=[
        None, 'diffpool', 'dmon', 'mincut', 'ecpool',
        'graclus', 'kmis', 'topk', 'panpool',
        'asapool', 'sagpool', 'dense-random',
        'sparse-random', 'comp-graclus'
    ],
    help="List of pooling methods to sweep over. Use None for no-pool."
)
parser.add_argument('--pool_ratio',    type=float, default=0.1)
parser.add_argument('--batch_size',    type=int,   default=32)
parser.add_argument('--hidden_channels',  type=int,   default=64)
parser.add_argument('--num_layers_pre',   type=int,   default=2)
parser.add_argument('--num_layers_post',  type=int,   default=1)
parser.add_argument('--lr',               type=float, default=1e-4)
parser.add_argument('--epochs',           type=int,   default=500)
parser.add_argument('--runs',             type=int,   default=1)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rng = np.random.default_rng(1)

# -------------------------------------------------------------------
# 2. Dataset setup
# -------------------------------------------------------------------
dataset = EXPWL1Dataset("data/EXPWL1/", transform=DataToFloat())

avg_nodes = int(dataset.data.num_nodes / len(dataset))
max_nodes = max(d.num_nodes for d in dataset)
# special scaling for sparse-random
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
    correct = 0
    total_loss = 0
    for data in loader:
        data = data.to(device)
        out, _ = model(data)
        total_loss += float(F.nll_loss(out, data.y)) * data.num_graphs
        correct += int((out.argmax(-1) == data.y).sum())
    return correct / len(loader.dataset), total_loss / len(loader.dataset)

# -------------------------------------------------------------------
# 4. Sweep over pooling methods
# -------------------------------------------------------------------
results = []
for pool_method in args.poolings:
    name = str(pool_method) if pool_method is not None else "no-pool"
    print(f"\n=== Running pooling: {name} ===")

    per_run_test_acc = []
    per_run_times     = []

    for run in range(args.runs):
        # shuffle
        idx = rng.permutation(len(dataset))
        ds  = dataset[idx]
        train_ds = ds[len(ds)//5:]
        val_ds   = ds[:len(ds)//10]
        test_ds  = ds[len(ds)//10:len(ds)//5]

        train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   args.batch_size)
        test_loader  = DataLoader(test_ds,  args.batch_size)

        # pick right max_nodes
        mn = max_nodes_sparse if pool_method=="sparse-random" else max_nodes

        # model & optimizer
        model = GIN_Pool_Net(
            in_channels   = train_ds.num_features,
            out_channels  = train_ds.num_classes,
            num_layers_pre  = args.num_layers_pre,
            num_layers_post = args.num_layers_post,
            hidden_channels = args.hidden_channels,
            average_nodes   = avg_nodes,
            pooling         = pool_method,
            pool_ratio      = args.pool_ratio,
            max_nodes       = mn
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val_loss = np.inf
        best_test_acc = 0.0

        # per-epoch tracking
        epoch_times  = []
        epoch_tests  = []

        for epoch in range(1, args.epochs+1):
            t0 = time.time()
            loss = train_epoch(model, train_loader, opt)
            t_ep = time.time() - t0

            tr_acc, _ = eval_epoch(model, train_loader)
            va_acc, va_loss = eval_epoch(model, val_loader)
            te_acc, _ = eval_epoch(model, test_loader)

            epoch_times.append(t_ep)
            epoch_tests.append(te_acc)

            # checkpoint on val loss
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_test_acc = te_acc

            # log to screen each epoch
            log(
                Pool=name, Run=run+1, Epoch=epoch,
                Loss=f"{loss:.4f}",
                Train=f"{tr_acc:.3f}",
                Val=f"{va_acc:.3f}",
                Test=f"{te_acc:.3f}",
                Time=f"{t_ep:.2f}s"
            )

        per_run_test_acc.append(best_test_acc)
        per_run_times.append(np.mean(epoch_times))

        # save plots instead of show
        # Accuracy vs Epoch
        plt.figure(figsize=(4,3))
        plt.plot(range(1, args.epochs+1), epoch_tests)
        plt.title(f"{name} — Run {run+1}")
        plt.xlabel("Epoch")
        plt.ylabel("Test Acc")
        plt.grid(True)
        fname_acc = f"{name}_run{run+1}_epoch_acc.png"
        plt.savefig(fname_acc, bbox_inches='tight')
        plt.close()

        # Cumulative Time vs Accuracy
        plt.figure(figsize=(4,3))
        cumt = np.cumsum(epoch_times)
        plt.plot(cumt, epoch_tests)
        plt.title(f"{name} — Run {run+1}")
        plt.xlabel("Cumulative Time (s)")
        plt.ylabel("Test Acc")
        plt.grid(True)
        fname_time = f"{name}_run{run+1}_time_acc.png"
        plt.savefig(fname_time, bbox_inches='tight')
        plt.close()

    # aggregate across runs
    mean_acc = np.mean(per_run_test_acc)
    std_acc  = np.std(per_run_test_acc)
    mean_t   = np.mean(per_run_times)

    expressive = '✓' if mean_acc > 0.95 else '✗'

    results.append({
        "Pooling":      name,
        "s/epoch":      f"{mean_t:.2f}s",
        "GIN layers":   f"{args.num_layers_pre}+{args.num_layers_post}",
        "Pool Ratio":   "—" if pool_method is None else f"{args.pool_ratio:.1f}",
        "Test Acc":     f"{mean_acc*100:.1f}±{std_acc*100:.1f}",
        "Expressive":   expressive
    })

# -------------------------------------------------------------------
# 5. Print summary table
# -------------------------------------------------------------------
print("\n" + "-"*70)
print(f{"{'Pooling':<12} | {'s/epoch':<8} | {'GIN layers':<10} | {'Ratio':<6} | {'Test Acc':<12} | {'Expr.'}"})
print("-"*70)
for r in results:
    print(f"{r['Pooling']:<12} | {r['s/epoch']:<8} | {r['GIN layers']:<10} | "
          f"{r['Pool Ratio']:<6} | {r['Test Acc']:<12} | {r['Expressive']}")
print("-"*70)

# -------------------------------------------------------------------
# 6. (Colab) Load & Display Saved Plots
# -------------------------------------------------------------------
# In a Colab cell, run:
# from IPython.display import Image, display
# import glob
# for img in sorted(glob.glob("*.png")):
#     display(Image(img))
