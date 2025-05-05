import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader
from scripts.nn_model2 import GIN_Dual_Pool_Net
from scripts.utils import EXPWL1Dataset, DataToFloat, log

# -------------------------------------------------------------------
# 1. Arguments
# -------------------------------------------------------------------
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument('--pooling1', type=str, default="none",
                    help="Comma-separated list of first pooling methods")
parser.add_argument('--pooling2', type=str, default="none",
                    help="Comma-separated list of second pooling methods")
parser.add_argument('--pool_ratio1', type=float, default=0.5,
                    help="Pooling ratio for the first pooling layer")
parser.add_argument('--pool_ratio2', type=float, default=0.2,
                    help="Pooling ratio for the second pooling layer")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--num_layers_pre', type=int, default=2,
                    help="Number of GIN layers before first pooling")
parser.add_argument('--num_layers_mid', type=int, default=1,
                    help="Number of GIN layers between poolings")
parser.add_argument('--num_layers_post', type=int, default=1,
                    help="Number of GIN layers after second pooling")
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--runs', type=int, default=1)
args = parser.parse_args()

# Parse pooling lists
raw1 = [p.strip() for p in args.pooling1.split(',') if p.strip()]
raw2 = [p.strip() for p in args.pooling2.split(',') if p.strip()]
pooling1_list = [None if p.lower() in ('none', 'no-pool', 'nopool', '') else p for p in raw1]
pooling2_list = [None if p.lower() in ('none', 'no-pool', 'nopool', '') else p for p in raw2]

if len(pooling1_list) != len(pooling2_list):
    raise ValueError("pooling1 and pooling2 lists must have the same length")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rng = np.random.default_rng(1)

# -------------------------------------------------------------------
# 2. Dataset setup
# -------------------------------------------------------------------
dataset = EXPWL1Dataset("data/EXPWL1/", transform=DataToFloat())
avg_nodes = int(dataset.data.num_nodes / len(dataset))
max_nodes = max(d.num_nodes for d in dataset)

# For cumulative plotting across pooling pairs
all_pool_epoch_tests = {}

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
# 4. Sweep over pooling pairs
# -------------------------------------------------------------------
results = []
for pool1, pool2 in zip(pooling1_list, pooling2_list):
    name = f"{pool1 if pool1 else 'none'}-{pool2 if pool2 else 'none'}"
    print(f"\n=== Running pooling pair: {name} ===")
    all_pool_epoch_tests[name] = []

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
            in_channels=dataset.num_features,
            out_channels=dataset.num_classes,
            num_layers_pre=args.num_layers_pre,
            num_layers_mid=args.num_layers_mid,
            num_layers_post=args.num_layers_post,
            hidden_channels=args.hidden_channels,
            average_nodes=avg_nodes,
            max_nodes=max_nodes,
            pooling1=pool1,
            pooling2=pool2,
            pool_ratio1=args.pool_ratio1,
            pool_ratio2=args.pool_ratio2,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val_loss = np.inf
        best_test_acc = 0.0

        epoch_times = []
        epoch_tests = []

        for epoch in range(1, args.epochs + 1):
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

            log(
                Pool=name, Run=run+1, Epoch=epoch,
                Loss=f"{loss:.4f}", Train=f"{tr_acc:.3f}",
                Val=f"{va_acc:.3f}", Test=f"{te_acc:.3f}", Time=f"{t_ep:.2f}s"
            )

        per_run_test_acc.append(best_test_acc)
        per_run_times.append(np.mean(epoch_times))
        all_pool_epoch_tests[name].append(epoch_tests)

        # Individual plots with epoch annotations
        # Accuracy vs Epoch
        plt.figure(figsize=(4, 3))
        plt.plot(range(1, args.epochs + 1), epoch_tests, marker='o', markevery=[0, args.epochs//2, args.epochs-1])
        for pt in (0, args.epochs//2, args.epochs-1):
            plt.annotate(str(pt+1), xy=(pt+1, epoch_tests[pt]), xytext=(5, 5), textcoords='offset points')
        plt.title(f"{name} — Run {run+1}")
        plt.xlabel("Epoch")
        plt.ylabel("Test Acc")
        plt.grid(True)
        plt.savefig(f"{name}_run{run+1}_epoch_acc.png", bbox_inches='tight')
        plt.close()

        # Cumulative Time vs Accuracy with epoch markers
        cum_time = np.cumsum(epoch_times)
        plt.figure(figsize=(4, 3))
        plt.plot(cum_time, epoch_tests, marker='s', markevery=[0, len(cum_time)//2, len(cum_time)-1])
        for idx in (0, len(cum_time)//2, len(cum_time)-1):
            plt.annotate(str(idx+1), xy=(cum_time[idx], epoch_tests[idx]), xytext=(5, -5), textcoords='offset points')
        plt.title(f"{name} — Run {run+1}")
        plt.xlabel("Cumulative Time (s)")
        plt.ylabel("Test Acc")
        plt.grid(True)
        plt.savefig(f"{name}_run{run+1}_time_acc.png", bbox_inches='tight')
        plt.close()

    # Aggregate results
    mean_acc = np.mean(per_run_test_acc)
    std_acc = np.std(per_run_test_acc)
    mean_t = np.mean(per_run_times)
    expressive = '✓' if mean_acc > 0.95 else '✗'
    results.append({
        "Pooling1": pool1 if pool1 else 'none',
        "Pooling2": pool2 if pool2 else 'none',
        "s/epoch": f"{mean_t:.2f}s",
        "Test Acc": f"{mean_acc*100:.1f}±{std_acc*100:.1f}",
        "Expressive": expressive
    })

# -------------------------------------------------------------------
# 5. Cumulative Plot: Avg Test Acc vs Epoch for All Pooling Pairs
# -------------------------------------------------------------------
plt.figure(figsize=(6, 4))
for name, runs in all_pool_epoch_tests.items():
    avg_curve = np.mean(runs, axis=0)
    plt.plot(range(1, args.epochs + 1), avg_curve, label=name)
plt.title("Average Test Accuracy per Epoch Across Pooling Pairs")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig("cumulative_poolings_epoch_acc.png", bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------
# 6. Print summary table
# -------------------------------------------------------------------
print(f"\nSummary for dual pooling with layers {args.num_layers_pre}+{args.num_layers_mid}+{args.num_layers_post}, "
      f"ratios {args.pool_ratio1:.1f}/{args.pool_ratio2:.1f}")
print("-"*80)
print(f"{'Pooling1':<12} | {'Pooling2':<12} | {'s/epoch':<8} | {'Test Acc':<12} | {'Expr.'}")
print("-"*80)
for r in results:
    print(f"{r['Pooling1']:<12} | {r['Pooling2']:<12} | {r['s/epoch']:<8} | {r['Test Acc']:<12} | {r['Expressive']}")
print("-"*80)