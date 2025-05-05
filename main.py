import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader
from scripts.nn_model import GIN_Pool_Net
from scripts.utils import EXPWL1Dataset, DataToFloat, log

rng = np.random.default_rng(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument('--pooling', type=str, default='mincut',
                    help="Options:\n None (no-pool)\n 'diffpool'\n 'mincut'\n" 
                    " 'dmon'\n 'edgepool'\n 'graclus'\n 'kmis'\n 'topk'\n 'panpool'\n"
                    " 'asapool'\n 'sagpool'\n 'dense-random'\n 'sparse-random'\n"
                    " 'comp-graclus'\n")
parser.add_argument('--pool_ratio', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--num_layers_pre', type=int, default=2)
parser.add_argument('--num_layers_post', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--runs', type=int, default=1)
args = parser.parse_args()
print(args)


def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, aux_loss = model(data)
        loss = F.nll_loss(out, data.y) + aux_loss
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(model, loader):
    model.eval()
    total_correct = 0
    total_loss = 0
    for data in loader:
        data = data.to(device)
        out, _ = model(data)
        loss = F.nll_loss(out, data.y)
        total_loss += float(loss) * data.num_graphs
        pred = out.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset), total_loss / len(loader.dataset)


### Dataset
path = "data/EXPWL1/"
dataset = EXPWL1Dataset(path, transform=DataToFloat()) 

# compute avg number of nodes
avg_nodes = int(dataset.data.num_nodes/len(dataset))

# compute max number of nodes
max_nodes = 0
for d in dataset:
    max_nodes = max(d.num_nodes, max_nodes)
if args.pooling == 'sparse-random':
    max_nodes *= args.batch_size

### Training
tot_acc = []
all_train_acc = []
all_val_acc = []
all_test_acc = []
all_epochs_time = []

for r in range(args.runs):  
    
    # Random shuffle the data
    rnd_idx = rng.permutation(len(dataset))
    dataset = dataset[list(rnd_idx)]
    
    train_dataset = dataset[len(dataset) // 5:]
    val_dataset = dataset[:len(dataset) // 10]
    test_dataset = dataset[len(dataset) // 10:len(dataset) // 5]
    
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size)
    test_loader = DataLoader(test_dataset, args.batch_size)
    
    # Init the GNN
    net_model = GIN_Pool_Net(in_channels=train_dataset.num_features, 
                        out_channels=train_dataset.num_classes,
                        num_layers_pre=args.num_layers_pre,
                        num_layers_post=args.num_layers_post,
                        hidden_channels=args.hidden_channels,
                        average_nodes=avg_nodes,
                        pooling=args.pooling,
                        pool_ratio=args.pool_ratio,
                        max_nodes=max_nodes
                        ).to(device)
    opt = torch.optim.Adam(net_model.parameters(), lr=args.lr)    
    
    # Train
    best_val = np.inf
    best_test = 0
    epoch_train_acc = []
    epoch_val_acc = []
    epoch_test_acc = []
    epoch_times = []
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        loss = train(net_model, train_loader, opt)
        train_acc, _ = test(net_model, train_loader)
        val_acc, val_loss = test(net_model, val_loader)
        test_acc, _ = test(net_model, test_loader)
        
        epoch_time = time.time() - start_time
        epoch_train_acc.append(train_acc)
        epoch_val_acc.append(val_acc)
        epoch_test_acc.append(test_acc)
        epoch_times.append(epoch_time)
        
        if val_loss < best_val:
            best_val = val_loss
            best_test = test_acc
        
        log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
        
    tot_acc.append(best_test)
    all_train_acc.append(epoch_train_acc)
    all_val_acc.append(epoch_val_acc)
    all_test_acc.append(epoch_test_acc)
    all_epochs_time.append(epoch_times)
    
    print(f"### Run {r:d} - val loss: {best_val:.3f}, test acc: {best_test:.3f}")
    
# Print final results
print("Accuracies in each run: ", tot_acc)    
print(f"test acc - mean: {np.mean(tot_acc):.3f}, std: {np.std(tot_acc):.3f}")

# Plot accuracy vs. epoch
for r in range(args.runs):
    plt.plot(range(1, args.epochs + 1), all_val_acc[r], label=f"Run {r+1} - Validation")
    plt.plot(range(1, args.epochs + 1), all_test_acc[r], label=f"Run {r+1} - Test")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch')
plt.legend()
plt.show()

# Plot time vs. accuracy
avg_time_per_epoch = np.mean([np.mean(times) for times in all_epochs_time])
avg_accuracy = np.mean(tot_acc)

plt.plot(range(1, args.runs + 1), [np.mean(times) for times in all_epochs_time], label='Time per epoch')
plt.axhline(y=avg_accuracy, color='r', linestyle='--', label=f'Average Accuracy: {avg_accuracy:.3f}')
plt.xlabel('Run')
plt.ylabel('Time (seconds)')
plt.title(f'Time per Run (Avg Time per Epoch: {avg_time_per_epoch:.3f} s)')
plt.legend()
plt.show()