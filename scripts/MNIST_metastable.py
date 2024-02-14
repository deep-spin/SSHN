# %%
import sys
sys.path.append("..")
from utils import HopfieldNet, Flatten, normmax_bisect
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from utils import entmax, SparseMAP_exactly_k
from collections import Counter
import numpy as np

# %%
torch.random.manual_seed(42)

# Define the transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5),
    Flatten()  # Normalize to [-1, 1]
])

# Load the MNIST dataset
mnist_dataset = datasets.MNIST(root='../datasets', train=True, download=False, transform=transform)
mnist_dataset_test = datasets.MNIST(root='../datasets', train=False, download=False, transform=transform)

# Create a DataLoader to iterate over the dataset
data_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=len(mnist_dataset), shuffle=True)
# Create a DataLoader to iterate over the dataset
data_loader_test = torch.utils.data.DataLoader(mnist_dataset_test, batch_size=len(mnist_dataset_test), shuffle=True)

for data in data_loader:
    X_train, labels_train = data

for data in data_loader_test:
    X_test, labels_test = data

def cccp(X, Q, alpha, beta, num_iters, k = None, normmax = False):
    Xi = Q
    for _ in range(num_iters):
        if normmax:
            P = normmax_bisect(X @ Xi *beta, alpha=alpha, n_iter=100, dim=0)
        elif k is not None:
            P = SparseMAP_exactly_k(X @ Xi *beta, k)
        else:
            P = entmax(X @ Xi *beta, alpha=alpha, dim=0)
        Xi = X.T @ P

    return P

num_iters = 5
eps = 1e-2
device = torch.device("cuda:" + "0")
X_train = X_train.to(device)
X_test = X_test.to(device)
n_samples = X_test.shape[0]
N = X_train.shape[0]
ctrs_total = []
for beta in [0.1, 1]:
    ctrs = []
    for alpha in [1, 1.5, 2]:

        P = cccp(X_train, X_test.T, alpha, beta, num_iters)
        eps_ = eps if alpha == 1 else 0
        sizes = (P > eps_).sum(dim=0)

        ctr = Counter(sizes.tolist())
        ctrs.append(ctr)
    
    for alpha in [2, 5]:
        P = cccp(X_train, X_test.T, alpha, beta, num_iters, None, True)
        eps_ = 0
        sizes = (P > eps_).sum(dim=0)

        ctr = Counter(sizes.tolist())
        ctrs.append(ctr)
    
    for k in [2,4,8]:
        P = cccp(X_train, X_test.T, alpha, beta, num_iters, k)
        eps_ = 0
        sizes = (P > eps_).sum(dim=0)

        ctr = Counter(sizes.tolist())
        ctrs.append(ctr)

    ctrs_total.append(ctrs)


for k in range(0, 11):

    print(f"{k} & "
        f"{ctrs_total[0][0][k] / n_samples * 100:.1f} & "
        f"{ctrs_total[0][1][k] / n_samples * 100:.1f} & "
        f"{ctrs_total[0][2][k] / n_samples * 100:.1f} & "
        f"{ctrs_total[0][3][k] / n_samples * 100:.1f} & "
        f"{ctrs_total[0][4][k] / n_samples * 100:.1f} & "
        f"{ctrs_total[0][5][k] / n_samples * 100:.1f} & "
        f"{ctrs_total[0][6][k] / n_samples * 100:.1f} & "
        f"{ctrs_total[0][7][k] / n_samples * 100:.1f} & "
        f"{ctrs_total[1][0][k] / n_samples * 100:.1f} & "
        f"{ctrs_total[1][1][k] / n_samples * 100:.1f} & "
        f"{ctrs_total[1][2][k] / n_samples * 100:.1f} & "
        f"{ctrs_total[1][3][k] / n_samples * 100:.1f} & "
        f"{ctrs_total[1][4][k] / n_samples * 100:.1f} & "
        f"{ctrs_total[1][5][k] / n_samples * 100:.1f} & "
        f"{ctrs_total[1][6][k] / n_samples * 100:.1f} & "
        f"{ctrs_total[1][7][k] / n_samples * 100:.1f} \\\\")