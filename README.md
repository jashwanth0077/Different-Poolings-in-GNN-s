# Instructions to run the code
* Clone this repository:
```
!rm -rf Different-Poolings-in-GNN-s
!git clone https://github.com/jashwanth0077/Different-Poolings-in-GNN-s.git
%cd Different-Poolings-in-GNN-s
```
* Install required dependencies/libraries:
```
!pip uninstall -y torch torchvision torchaudio torchtext torchdata
!pip uninstall -y torch-scatter torch-sparse torch-geometric
!pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchtext==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
!pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
!pip install torch-geometric
!pip install matplotlib
```
* Run the command:
```
!python main.py \
  --poolings <arguments>\
  --epochs <n-epochs> \
  --runs <n-runs>
```
Here, arguments are comma-separated list of pooling methods from `{none, diffpool, dmon, mincut, ecpool, graclus, kmis, topk, panpool, asapool, sagpool, dense-random, sparse-random, comp-graclus}`, 'none' corresponds to no-pool.

\<n-epochs\> is the number of epochs.

\<n-runs\> is the number of runs.
# Repository structure
```
|
|-- data/EXPWL1
|   |-- processed
|   |-- raw
|-- scripts
|   |-- pooling
|       |-- kmis
|           |- kmis_pool.py
|           |- utils.py
|       |- rnd_sparse.py
|   |- nn_model.py
|   |- nn_model2.py
|   |- sum_pool.py
|   |- utils.py
|- README.md
|- main.py
|- main2.py
```
# Modules Implementations
main.py contains code for training single pool layer

main1.py contains code for training double pool layers

nn_model.py contains code for model of single pool layer

nn_model2.py contains code for model of double pool layer

kmis_pool.py contains code for kmis pooling layer

sum_pool.py contains aggregate function in MP layer
