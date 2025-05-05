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
parser.add_argument('--poolings', type=str,
    default="none,diffpool,dmon,mincut,ecpool,graclus,kmis,topk,panpool,asapool,sagpool,dense-random,sparse-random,comp-graclus",
    help="Comma-separated list of pooling methods (e.g., none,graclus,topk). 'none' maps to no-pool."
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

# split comma-separated string into list, map 'none' to None
raw = [p.strip() for p in args.poolings.split(',') if p.strip()]
args.poolings = [None if p.lower() in ('none','no-pool','nopool','') else p for p in raw]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rng = np.random.default_rng(1)

# -------------------------------------------------------------------
# 2. Dataset setup
# -------------------------------------------------------------------
dataset = EXPWL1Dataset("data/EXPWL1/", transform=DataToFloat())
avg_nodes = int(dataset.data.num_nodes / len(dataset))
max_nodes = max(d.num_nodes for d in dataset)
max_nodes_sparse = max_nodes * args.batch_size

# For cumulative plotting across poolings
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
# 4. Sweep over pooling methods
# -------------------------------------------------------------------
results = []
for pool_method in args.poolings:
    name = str(pool_method) if pool_method is not None else "no-pool"
    print(f"\n=== Running pooling: {name} ===")
    all_pool_epoch_tests[name] = []

    per_run_test_acc = []
    per_run_times     = []

    for run in range(args.runs):
        idx = rng.permutation(len(dataset))
        ds  = dataset[idx]
        train_ds = ds[len(ds)//5:]
        val_ds   = ds[:len(ds)//10]
        test_ds  = ds[len(ds)//10:len(ds)//5]

        train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   args.batch_size)
        test_loader  = DataLoader(test_ds,  args.batch_size)

        mn = max_nodes_sparse if pool_method == "sparse-random" else max_nodes
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

            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_test_acc = te_acc

            # log every epoch with time and some epoch ticks
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
        plt.figure(figsize=(4,3))
        # plt.plot(range(1, args.epochs+1), epoch_tests, marker='o', markevery=[0, mid_epoch, args.epochs-1])
        plt.plot(range(1, args.epochs+1), epoch_tests, marker='o', markevery=[0, args.epochs//2, args.epochs-1])
        for pt in (0, args.epochs//2, args.epochs-1):
            plt.annotate(str(pt+1), xy=(pt+1, epoch_tests[pt]), xytext=(5,5), textcoords='offset points')
        plt.title(f"{name} — Run {run+1}")
        plt.xlabel("Epoch")
        plt.ylabel("Test Acc")
        plt.grid(True)
        plt.savefig(f"{name}_run{run+1}_epoch_acc.png", bbox_inches='tight')
        plt.close()

        # Cumulative Time vs Accuracy with epoch markers
        cum_time = np.cumsum(epoch_times)
        plt.figure(figsize=(4,3))
        plt.plot(cum_time, epoch_tests, marker='s', markevery=[0, len(cum_time)//2, len(cum_time)-1])
        for idx in (0, len(cum_time)//2, len(cum_time)-1):
            plt.annotate(str(idx+1), xy=(cum_time[idx], epoch_tests[idx]), xytext=(5,-5), textcoords='offset points')
        plt.title(f"{name} — Run {run+1}")
        plt.xlabel("Cumulative Time (s)")
        plt.ylabel("Test Acc")
        plt.grid(True)
        plt.savefig(f"{name}_run{run+1}_time_acc.png", bbox_inches='tight')
        plt.close()

    # aggregate
    mean_acc = np.mean(per_run_test_acc)
    std_acc  = np.std(per_run_test_acc)
    mean_t   = np.mean(per_run_times)
    expressive = '✓' if mean_acc > 0.95 else '✗'
    results.append({
        "Pooling":    name,
        "s/epoch":    f"{mean_t:.2f}s",
        "GIN layers": f"{args.num_layers_pre}+{args.num_layers_post}",
        "Pool Ratio": "—" if pool_method is None else f"{args.pool_ratio:.1f}",
        "Test Acc":   f"{mean_acc*100:.1f}±{std_acc*100:.1f}",
        "Expressive": expressive
    })

# -------------------------------------------------------------------
# 5. Cumulative Plot: Avg Test Acc vs Epoch for All Poolings
# -------------------------------------------------------------------
plt.figure(figsize=(6,4))
for name, runs in all_pool_epoch_tests.items():
    avg_curve = np.mean(runs, axis=0)
    plt.plot(range(1, args.epochs+1), avg_curve, label=name)
plt.title("Average Test Accuracy per Epoch Across Poolings")
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
print("\n" + "-"*70)
print(f"{'Pooling':<12} | {'s/epoch':<8} | {'GIN layers':<10} | {'Ratio':<6} | {'Test Acc':<12} | {'Expr.'}")
print("-"*70)
for r in results:
    print(f"{r['Pooling']:<12} | {r['s/epoch']:<8} | {r['GIN layers']:<10} | "
          f"{r['Pool Ratio']:<6} | {r['Test Acc']:<12} | {r['Expressive']}")
print("-"*70)

# -------------------------------------------------------------------
# 7. (Colab) Load & Display Saved Plots
# -------------------------------------------------------------------
# In a Colab cell:
# from IPython.display import Image, display
# import glob
# # Individual:
# for img in sorted(glob.glob("*_epoch_acc.png")) + sorted(glob.glob("*_time_acc.png")):
#     display(Image(img))
# # Cumulative:
# display(Image('cumulative_poolings_epoch_acc.png'))

# main.py with modifications for all combinations of pooling methods

# main.py with modifications for all combinations of pooling methods

# import argparse
# import time
# import numpy as np
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import itertools  # Added for combination generation

# from torch_geometric.loader import DataLoader
# from scripts.nn_model import GIN_Pool_Net
# from scripts.nn_model2 import GIN_Dual_Pool_Net
# from scripts.utils import EXPWL1Dataset, DataToFloat, log

# # -------------------------------------------------------------------
# # 1. Arguments
# # -------------------------------------------------------------------
# parser = argparse.ArgumentParser(
#     formatter_class=argparse.RawTextHelpFormatter
# )
# parser.add_argument('--poolings', type=str,
#     default="none,diffpool,dmon,mincut,ecpool,graclus,kmis,topk,panpool,asapool,sagpool,dense-random,sparse-random,comp-graclus",
#     help="Comma-separated list of pooling methods (e.g., none,graclus,topk). 'none' maps to no-pool."
# )
# parser.add_argument('--pool_ratio', type=float, default=0.1,
#     help="Pooling ratio for single pooling or first pooling in dual mode")
# parser.add_argument('--batch_size', type=int, default=32)
# parser.add_argument('--hidden_channels', type=int, default=64)
# parser.add_argument('--num_layers_pre', type=int, default=2,
#     help="Number of GIN layers before first pooling")
# parser.add_argument('--num_layers_post', type=int, default=1, 
#     help="Number of GIN layers after pooling (last pooling in dual mode)")
# parser.add_argument('--num_layers_mid', type=int, default=1,
#     help="Number of GIN layers between first and second pooling in dual mode")
# parser.add_argument('--lr', type=float, default=1e-4)
# parser.add_argument('--epochs', type=int, default=500)
# parser.add_argument('--runs', type=int, default=1)

# # Add dual pooling specific arguments
# parser.add_argument('--dual_mode', action='store_true',
#     help="Enable dual pooling mode with two sequential pooling operations")
# parser.add_argument('--pooling2', type=str, default=None,
#     help="Comma-separated list of second pooling methods for dual mode")
# parser.add_argument('--pool_ratio2', type=float, default=0.2,
#     help="Pooling ratio for second pooling in dual mode")

# args = parser.parse_args()

# # split comma-separated string into list, map 'none' to None
# raw = [p.strip() for p in args.poolings.split(',') if p.strip()]
# args.poolings = [None if p.lower() in ('none','no-pool','nopool','') else p for p in raw]

# # Handle pooling2 argument same way
# if args.pooling2:
#     raw2 = [p.strip() for p in args.pooling2.split(',') if p.strip()]
#     args.pooling2s = [None if p.lower() in ('none', 'no-pool', 'nopool', '') else p for p in raw2]
# else:
#     args.pooling2s = [None]  # Default to None if not provided

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# rng = np.random.default_rng(1)

# # -------------------------------------------------------------------
# # 2. Dataset setup
# # -------------------------------------------------------------------
# dataset = EXPWL1Dataset("data/EXPWL1/", transform=DataToFloat())
# avg_nodes = int(dataset.data.num_nodes / len(dataset))
# max_nodes = max(d.num_nodes for d in dataset)
# max_nodes_sparse = max_nodes * args.batch_size

# # For cumulative plotting across poolings
# all_pool_epoch_tests = {}

# # -------------------------------------------------------------------
# # 3. Helpers
# # -------------------------------------------------------------------
# def train_epoch(model, loader, optimizer):
#     model.train()
#     total_loss = 0
#     for data in loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         out, aux = model(data)
#         loss = F.nll_loss(out, data.y) + aux
#         loss.backward()
#         optimizer.step()
#         total_loss += float(loss) * data.num_graphs
#     return total_loss / len(loader.dataset)

# @torch.no_grad()
# def eval_epoch(model, loader):
#     model.eval()
#     correct = 0
#     total_loss = 0
#     for data in loader:
#         data = data.to(device)
#         out, _ = model(data)
#         total_loss += float(F.nll_loss(out, data.y)) * data.num_graphs
#         correct += int((out.argmax(-1) == data.y).sum())
#     return correct / len(loader.dataset), total_loss / len(loader.dataset)

# # -------------------------------------------------------------------
# # 4. Model creation helper
# # -------------------------------------------------------------------
# def create_model(pool_method, pooling2=None):
#     """Create either single pooling or dual pooling model based on arguments"""
#     mn = max_nodes_sparse if pool_method == "sparse-random" or pooling2 == "sparse-random" else max_nodes
    
#     if args.dual_mode and pooling2 is not None:
#         # Using dual pooling model
#         model_name = f"{pool_method}-{pooling2}" if pool_method is not None else f"no-pool-{pooling2}"
#         model = GIN_Dual_Pool_Net(
#             in_channels=train_ds.num_features,
#             out_channels=train_ds.num_classes,
#             num_layers_pre=args.num_layers_pre,
#             num_layers_mid=args.num_layers_mid,
#             num_layers_post=args.num_layers_post,
#             hidden_channels=args.hidden_channels,
#             average_nodes=avg_nodes,
#             pooling1=pool_method,
#             pooling2=pooling2,
#             pool_ratio1=args.pool_ratio,
#             pool_ratio2=args.pool_ratio2,
#             max_nodes=mn
#         ).to(device)
#     else:
#         # Using single pooling model
#         model_name = str(pool_method) if pool_method is not None else "no-pool"
#         model = GIN_Pool_Net(
#             in_channels=train_ds.num_features,
#             out_channels=train_ds.num_classes,
#             num_layers_pre=args.num_layers_pre,
#             num_layers_post=args.num_layers_post,
#             hidden_channels=args.hidden_channels,
#             average_nodes=avg_nodes,
#             pooling=pool_method,
#             pool_ratio=args.pool_ratio,
#             max_nodes=mn
#         ).to(device)
    
#     return model, model_name

# # -------------------------------------------------------------------
# # 5. Sweep over pooling methods
# # -------------------------------------------------------------------
# results = []

# # If dual mode is enabled, iterate through all combinations of --poolings and --pooling2s
# if args.dual_mode:
#     print(f"\n=== Running in DUAL POOLING mode ===")
    
#     # Calculate total combinations for reporting
#     total_combinations = len(args.poolings) * len(args.pooling2s)
#     print(f"Testing {total_combinations} combinations of pooling methods:")
    
#     # Preview all combinations that will be tested
#     for pool_method in args.poolings:
#         pool1_name = pool_method if pool_method is not None else "no-pool"
#         for pooling2 in args.pooling2s:
#             pool2_name = pooling2 if pooling2 is not None else "no-pool"
#             print(f"  - {pool1_name} + {pool2_name}")
    
#     # Run experiments with nested for loops
#     for pool_method in args.poolings:
#         for pooling2 in args.pooling2s:
#             model_name = f"{pool_method if pool_method is not None else 'no-pool'}-{pooling2 if pooling2 is not None else 'no-pool'}"
#             print(f"\n=== Running dual pooling: {model_name} ===")
#             all_pool_epoch_tests[model_name] = []
        
#         per_run_test_acc = []
#         per_run_times = []
        
#         for run in range(args.runs):
#             idx = rng.permutation(len(dataset))
#             ds = dataset[idx]
#             train_ds = ds[len(ds)//5:]
#             val_ds = ds[:len(ds)//10]
#             test_ds = ds[len(ds)//10:len(ds)//5]

#             train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)
#             val_loader = DataLoader(val_ds, args.batch_size)
#             test_loader = DataLoader(test_ds, args.batch_size)
            
#             model, _ = create_model(pool_method, pooling2)
#             opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            
#             best_val_loss = np.inf
#             best_test_acc = 0.0
            
#             epoch_times = []
#             epoch_tests = []
            
#             for epoch in range(1, args.epochs+1):
#                 t0 = time.time()
#                 loss = train_epoch(model, train_loader, opt)
#                 t_ep = time.time() - t0
                
#                 tr_acc, _ = eval_epoch(model, train_loader)
#                 va_acc, va_loss = eval_epoch(model, val_loader)
#                 te_acc, _ = eval_epoch(model, test_loader)
                
#                 epoch_times.append(t_ep)
#                 epoch_tests.append(te_acc)
                
#                 if va_loss < best_val_loss:
#                     best_val_loss = va_loss
#                     best_test_acc = te_acc
                
#                 # log every epoch with time and some epoch ticks
#                 log(
#                     Pool=model_name, Run=run+1, Epoch=epoch,
#                     Loss=f"{loss:.4f}", Train=f"{tr_acc:.3f}",
#                     Val=f"{va_acc:.3f}", Test=f"{te_acc:.3f}", Time=f"{t_ep:.2f}s"
#                 )
            
#             per_run_test_acc.append(best_test_acc)
#             per_run_times.append(np.mean(epoch_times))
#             all_pool_epoch_tests[model_name].append(epoch_tests)
            
#             # Individual plots with epoch annotations
#             # Accuracy vs Epoch
#             plt.figure(figsize=(4,3))
#             plt.plot(range(1, args.epochs+1), epoch_tests, marker='o', markevery=[0, args.epochs//2, args.epochs-1])
#             for pt in (0, args.epochs//2, args.epochs-1):
#                 plt.annotate(str(pt+1), xy=(pt+1, epoch_tests[pt]), xytext=(5,5), textcoords='offset points')
#             plt.title(f"{model_name} — Run {run+1}")
#             plt.xlabel("Epoch")
#             plt.ylabel("Test Acc")
#             plt.grid(True)
#             plt.savefig(f"{model_name}_run{run+1}_epoch_acc.png", bbox_inches='tight')
#             plt.close()
            
#             # Cumulative Time vs Accuracy with epoch markers
#             cum_time = np.cumsum(epoch_times)
#             plt.figure(figsize=(4,3))
#             plt.plot(cum_time, epoch_tests, marker='s', markevery=[0, len(cum_time)//2, len(cum_time)-1])
#             for idx in (0, len(cum_time)//2, len(cum_time)-1):
#                 plt.annotate(str(idx+1), xy=(cum_time[idx], epoch_tests[idx]), xytext=(5,-5), textcoords='offset points')
#             plt.title(f"{model_name} — Run {run+1}")
#             plt.xlabel("Cumulative Time (s)")
#             plt.ylabel("Test Acc")
#             plt.grid(True)
#             plt.savefig(f"{model_name}_run{run+1}_time_acc.png", bbox_inches='tight')
#             plt.close()
        
#         # aggregate
#         mean_acc = np.mean(per_run_test_acc)
#         std_acc = np.std(per_run_test_acc)
#         mean_t = np.mean(per_run_times)
#         expressive = '✓' if mean_acc > 0.95 else '✗'
#         results.append({
#             "Pooling": model_name,
#             "s/epoch": f"{mean_t:.2f}s",
#             "GIN layers": f"{args.num_layers_pre}+{args.num_layers_mid}+{args.num_layers_post}",
#             "Pool Ratio": f"{args.pool_ratio:.1f}/{args.pool_ratio2:.1f}",
#             "Test Acc": f"{mean_acc*100:.1f}±{std_acc*100:.1f}",
#             "Expressive": expressive
#         })
        
# else:
#     # Original single pooling flow
#     for pool_method in args.poolings:
#         name = str(pool_method) if pool_method is not None else "no-pool"
#         print(f"\n=== Running pooling: {name} ===")
#         all_pool_epoch_tests[name] = []

#         per_run_test_acc = []
#         per_run_times = []

#         for run in range(args.runs):
#             idx = rng.permutation(len(dataset))
#             ds = dataset[idx]
#             train_ds = ds[len(ds)//5:]
#             val_ds = ds[:len(ds)//10]
#             test_ds = ds[len(ds)//10:len(ds)//5]

#             train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)
#             val_loader = DataLoader(val_ds, args.batch_size)
#             test_loader = DataLoader(test_ds, args.batch_size)

#             model, _ = create_model(pool_method)
#             opt = torch.optim.Adam(model.parameters(), lr=args.lr)

#             best_val_loss = np.inf
#             best_test_acc = 0.0

#             epoch_times = []
#             epoch_tests = []

#             for epoch in range(1, args.epochs+1):
#                 t0 = time.time()
#                 loss = train_epoch(model, train_loader, opt)
#                 t_ep = time.time() - t0

#                 tr_acc, _ = eval_epoch(model, train_loader)
#                 va_acc, va_loss = eval_epoch(model, val_loader)
#                 te_acc, _ = eval_epoch(model, test_loader)

#                 epoch_times.append(t_ep)
#                 epoch_tests.append(te_acc)

#                 if va_loss < best_val_loss:
#                     best_val_loss = va_loss
#                     best_test_acc = te_acc

#                 # log every epoch with time and some epoch ticks
#                 log(
#                     Pool=name, Run=run+1, Epoch=epoch,
#                     Loss=f"{loss:.4f}", Train=f"{tr_acc:.3f}",
#                     Val=f"{va_acc:.3f}", Test=f"{te_acc:.3f}", Time=f"{t_ep:.2f}s"
#                 )

#             per_run_test_acc.append(best_test_acc)
#             per_run_times.append(np.mean(epoch_times))
#             all_pool_epoch_tests[name].append(epoch_tests)

#             # Individual plots with epoch annotations
#             # Accuracy vs Epoch
#             plt.figure(figsize=(4,3))
#             plt.plot(range(1, args.epochs+1), epoch_tests, marker='o', markevery=[0, args.epochs//2, args.epochs-1])
#             for pt in (0, args.epochs//2, args.epochs-1):
#                 plt.annotate(str(pt+1), xy=(pt+1, epoch_tests[pt]), xytext=(5,5), textcoords='offset points')
#             plt.title(f"{name} — Run {run+1}")
#             plt.xlabel("Epoch")
#             plt.ylabel("Test Acc")
#             plt.grid(True)
#             plt.savefig(f"{name}_run{run+1}_epoch_acc.png", bbox_inches='tight')
#             plt.close()

#             # Cumulative Time vs Accuracy with epoch markers
#             cum_time = np.cumsum(epoch_times)
#             plt.figure(figsize=(4,3))
#             plt.plot(cum_time, epoch_tests, marker='s', markevery=[0, len(cum_time)//2, len(cum_time)-1])
#             for idx in (0, len(cum_time)//2, len(cum_time)-1):
#                 plt.annotate(str(idx+1), xy=(cum_time[idx], epoch_tests[idx]), xytext=(5,-5), textcoords='offset points')
#             plt.title(f"{name} — Run {run+1}")
#             plt.xlabel("Cumulative Time (s)")
#             plt.ylabel("Test Acc")
#             plt.grid(True)
#             plt.savefig(f"{name}_run{run+1}_time_acc.png", bbox_inches='tight')
#             plt.close()

#         # aggregate
#         mean_acc = np.mean(per_run_test_acc)
#         std_acc = np.std(per_run_test_acc)
#         mean_t = np.mean(per_run_times)
#         expressive = '✓' if mean_acc > 0.95 else '✗'
#         results.append({
#             "Pooling": name,
#             "s/epoch": f"{mean_t:.2f}s",
#             "GIN layers": f"{args.num_layers_pre}+{args.num_layers_post}",
#             "Pool Ratio": "—" if pool_method is None else f"{args.pool_ratio:.1f}",
#             "Test Acc": f"{mean_acc*100:.1f}±{std_acc*100:.1f}",
#             "Expressive": expressive
#         })

# # -------------------------------------------------------------------
# # 6. Cumulative Plot: Avg Test Acc vs Epoch for All Poolings
# # -------------------------------------------------------------------
# plt.figure(figsize=(6,4))
# for name, runs in all_pool_epoch_tests.items():
#     avg_curve = np.mean(runs, axis=0)
#     plt.plot(range(1, args.epochs+1), avg_curve, label=name)
# plt.title("Average Test Accuracy per Epoch Across Poolings")
# plt.xlabel("Epoch")
# plt.ylabel("Test Accuracy")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("cumulative_poolings_epoch_acc.png", bbox_inches='tight')
# plt.close()

# # -------------------------------------------------------------------
# # 7. Print summary table
# # -------------------------------------------------------------------
# print("\n" + "-"*80)  # Wider for dual pooling names
# print(f"{'Pooling':<20} | {'s/epoch':<8} | {'GIN layers':<12} | {'Ratio':<10} | {'Test Acc':<12} | {'Expr.'}")
# print("-"*80)
# for r in results:
#     print(f"{r['Pooling']:<20} | {r['s/epoch']:<8} | {r['GIN layers']:<12} | "
#           f"{r['Pool Ratio']:<10} | {r['Test Acc']:<12} | {r['Expressive']}")
# print("-"*80)

# # -------------------------------------------------------------------
# # 8. (Colab) Load & Display Saved Plots
# # -------------------------------------------------------------------
# # In a Colab cell:
# # from IPython.display import Image, display
# # import glob
# # # Individual:
# # for img in sorted(glob.glob("*_epoch_acc.png")) + sorted(glob.glob("*_time_acc.png")):
# #     display(Image(img))
# # # Cumulative:
# # display(Image('cumulative_poolings_epoch_acc.png'))