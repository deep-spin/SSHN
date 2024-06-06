import torch
import argparse
import itertools
import pandas as pd
import numpy as np

import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch.nn import Conv2d, Dropout, Linear, MaxPool2d, Module, ReLU, Sequential, Sigmoid
import os

#os.environ['TMPDIR'] = "/home/saul/hopfield-wip-main/Sparse_Hopfield_Networks/scripts/examples/tmp"
# Change the current working directory to the script's directory
# Import auxiliary modules.
from typing import Optional, Tuple

# Import Hopfield-specific modules.
from hflayers import HopfieldPooling
from datasets.loader import load_data, DummyDataset, load_ucsb

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import multiprocessing
from multiprocessing import Manager

def get_args():
    global num_processes
    global device
    device = "cuda:7"
    num_processes = 8
    parser = argparse.ArgumentParser(description='Examples of MIL benchmarks:')
    parser.add_argument('--dataset', default="tiger", type=str, choices=['fox', 'tiger', 'elephant'])
    parser.add_argument('--mode', default='normmax', type=str, choices=['softmax', 'sparsemax', "entmax", "sparsemap", "normmax"])
    parser.add_argument('--k', help='k exact ones for sparsemap', default=5, type=int)
    parser.add_argument('--alpha', help='alpha for normmax', default=10, type=float)

    args = parser.parse_args()
    return args

manager = Manager()
df_list = manager.list()

class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

class MyPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(MyPool, self).__init__(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        self.join()
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.03):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_auc = 0

    def early_stop(self, validation_auc):
        if validation_auc > self.max_validation_auc:
            self.max_validation_loss = validation_auc
            self.counter = 0
        elif validation_auc < (self.max_validation_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class HopfieldMIL(nn.Module):
    def __init__(self, feat_dim, emb_dims, emb_layers, num_heads, hidden_size, beta, dropout, alpha, k, sparseMAP = False, normmax = False):
        super(HopfieldMIL, self).__init__()
        emb = [nn.Linear(feat_dim, emb_dims), nn.ReLU()]
        for i in range(emb_layers - 1):
            emb.append(nn.Linear(emb_dims, emb_dims))
            emb.append(nn.ReLU())
        self.emb = nn.ModuleList(emb)

        self.hopfield_pooling = HopfieldPooling(
            input_size=emb_dims, hidden_size=hidden_size, output_size=emb_dims, num_heads=num_heads, alpha = alpha, sparseMAP=sparseMAP, normmax = normmax, k=k, scaling = beta,
        alpha_as_static=True, dropout=dropout)
        
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dims, 1)
        )

    def forward(self, x):

        H = x.float()
        for l in self.emb:
            H = l(H)
        H = self.hopfield_pooling(H)
        Y_prob = self.classifier(H).flatten()

        return Y_prob

def train_epoch(network: Module,
                optimizer: torch.optim.AdamW,
                data_loader: DataLoader,
                device
               ) -> Tuple[float, float, float]:
    """
    Execute one training epoch.
    
    :param network: network instance to train
    :param optimiser: optimiser instance responsible for updating network parameters
    :param data_loader: data loader instance providing training data
    :return: tuple comprising training loss, training error as well as accuracy
    """
    network.train()
    losses, errors, accuracies, rocs = [], [], [], []

    for data, target, mask in data_loader:
        
        data, target = data.to(device=device), target.to(device=device).float()

        # Process data by Hopfield-based network.
        out = network(data)
        
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(input=out, target=target, reduction=r'mean')

        # Update network parameters.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=network.parameters(), max_norm=1.0, norm_type=2)
        optimizer.step()

        # Compute performance measures of current model.
        accuracy = (out.sigmoid().round() == target).to(dtype=torch.float32).mean()
        accuracies.append(accuracy.detach().item())
        losses.append(loss.detach().item())
    
    # Report progress of training procedure.
    return sum(losses) / len(losses), sum(accuracies) / len(accuracies)

def set_seed(seed: int = 42) -> None:
    """
    Set seed for all underlying (pseudo) random number sources.
    
    :param seed: seed to be used
    :return: None
    """
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def eval_iter(network: Module,
              data_loader: DataLoader,
              device
             ) -> Tuple[float, float, float]:
    """
    Evaluate the current model.
    
    :param network: network instance to evaluate
    :param data_loader: data loader instance providing validation data
    :return: tuple comprising validation loss, validation error as well as accuracy
    """
    network.eval()
    # p_bar = tqdm(data_loader, total=len(data_loader))

    with torch.no_grad():
        losses, accuracies, preds, targets = [], [], [], []
        for data, target, mask in data_loader:

            # Now, target_scalar contains the integer value
            targets.append(target)
            data, target = data.to(device=device), target.to(device=device).float()

            # Process data by Hopfield-based network
            out = network(data)
            loss = F.binary_cross_entropy_with_logits(input=out, target=target, reduction=r'mean')

        

            accuracy = (out.sigmoid().round() == target).to(dtype=torch.float32).mean()
            accuracies.append(accuracy.detach().item())
            preds.append(out.sigmoid().squeeze().detach().cpu())
            losses.append(loss.detach().item())
        predictions = [arr.item() for arr in preds]
        targets = [arr.item() for arr in targets]
        auc = roc_auc_score(targets, predictions)
        return sum(losses) / len(losses), sum(accuracies) / len(accuracies), auc

def train(hyperparameters):
    set_seed()
    train_features = config["train_features"]
    train_labels = config["train_labels"]
    testset = config["testset"]
    lr, gamma, emb_dims, emb_layers, hid_dim, num_heads, beta, dropout = hyperparameters
    seed = config["seed"]

    skf_inner = StratifiedKFold(n_splits=9, random_state=seed, shuffle=True)
    train_subset_ids, val_subset_ids = next(skf_inner.split(train_features, train_labels))
    train_subset_features, train_subset_labels = [train_features[id] for id in train_subset_ids] \
        , [train_labels[id] for id in train_subset_ids]
    val_subset_features, val_subset_labels = [train_features[id] for id in val_subset_ids] \
        , [train_labels[id] for id in val_subset_ids]
    train_subset, val_subset = DummyDataset(train_subset_features, train_subset_labels) \
        , DummyDataset(val_subset_features, val_subset_labels)

    set_seed()
    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        collate_fn=testset.collate
    )
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        collate_fn=testset.collate
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        collate_fn=testset.collate
    )

    if args.mode == "sparsemap":
        alpha = None
        beta_rsh = torch.full((int(num_heads),), float(beta), dtype=torch.float).to(device)
        network = HopfieldMIL(args.feat_dim, emb_dims, emb_layers, num_heads, hid_dim, beta_rsh, dropout, alpha, args.k, True).to(device=device)
    
    elif args.mode == "normmax":
        alpha = args.alpha
        beta_rsh = torch.full((int(num_heads),), float(beta), dtype=torch.float).to(device)
        network = HopfieldMIL(args.feat_dim, emb_dims, emb_layers, num_heads, hid_dim, beta_rsh, dropout, alpha, args.k, False,True).to(device=device).to(device=device)
    
    elif args.mode == "sparsemax":
        alpha = 2.0
        beta_rsh = torch.full((int(num_heads),), float(beta), dtype=torch.float).to(device)
        alpha_rsh = torch.full((int(num_heads),), alpha, dtype=torch.float).to(device)
        network = HopfieldMIL(args.feat_dim, emb_dims, emb_layers, num_heads, hid_dim, beta_rsh, dropout, alpha_rsh, args.k, False).to(device=device).to(device=device)
    
    elif args.mode == "entmax":
        alpha = 1.5
        beta_rsh = torch.full((int(num_heads),), float(beta), dtype=torch.float).to(device)
        alpha_rsh = torch.full((int(num_heads),), alpha, dtype=torch.float).to(device)
        network = HopfieldMIL(args.feat_dim, emb_dims, emb_layers, num_heads, hid_dim, beta_rsh, dropout, alpha_rsh, args.k, False).to(device=device).to(device=device)
    
    elif args.mode == "softmax":
        alpha = 1.0

        beta_rsh = torch.full((int(num_heads),), float(beta), dtype=torch.float).to(device)
        alpha_rsh = torch.full((int(num_heads),), alpha, dtype=torch.float).to(device)
        network = HopfieldMIL(args.feat_dim, emb_dims, emb_layers, num_heads, hid_dim, beta_rsh, dropout, alpha_rsh, args.k, False).to(device=device).to(device=device)

    hyperparameters_dict = {"seed": seed,
        "run": config["run"],
        "lr": lr,
        "gamma": gamma,
        "emb_dims" : emb_dims,
        "emb_layers" : emb_layers,
        "hidden_size": hid_dim,
        "n_heads": num_heads,
        "beta": beta,
        "dropout": dropout,
        "alpha": alpha
    }
    optimizer = torch.optim.AdamW(params=network.parameters(), lr=lr, weight_decay=1e-4)
    early_stopper = EarlyStopper()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    best_auc = 0.0
    for epoch in range(50):  # loop over the dataset multiple times
        _ = train_epoch(network, optimizer, trainloader, device)
        scheduler.step()

        val_loss, val_acc, val_auc = eval_iter(network, valloader, device)
        if best_auc<val_auc:
            test_loss, test_acc, test_auc = eval_iter(network, testloader, device)
        if early_stopper.early_stop(val_auc):
            break
    hyperparameters_dict['acc_test'], hyperparameters_dict["acc_val"], hyperparameters_dict["auc_test"], hyperparameters_dict["auc_val"], hyperparameters_dict["acc_test"], hyperparameters_dict["acc_val"]  = test_acc, val_acc, test_auc, val_auc, test_acc, val_acc
    
    df_list.append(hyperparameters_dict)

    # Convert the shared list to a DataFrame
    df = pd.DataFrame(list(df_list))
    with open(csv_filename, 'w') as file:
        # Save the DataFrame to a CSV file
        df.to_csv(file, index=False)

def wrapper_function(hyperparameter_chunk):
    for hyperparameters in hyperparameter_chunk:
        train(hyperparameters)


def main(args):
    features, labels = load_data(args) if args.dataset!="ucsb" else load_ucsb()
    args.feat_dim = features[0].shape[-1]
    args.max_len = max([features[id].shape[0] for id in range(len(features))])
    global config
    config = {
        "lr": [1e-3, 1e-5],
        "lr_decay": [0.98, 0.96],
        "emb_dims": [32, 128],
        "emb_layers": [2],
        "hid_dim": [32, 64],
        "num_heads": [12],
        "beta": [0.1, 1.0, 10.0],
        "dropout":[0.0, 0.75]
    }
    # Generate all combinations of hyperparameters
    hyperparameter_combinations = list(itertools.product(*config.values()))
    hyperparameter_chunks = [hyperparameter_combinations[i:i + len(hyperparameter_combinations) // num_processes] for i in range(0, len(hyperparameter_combinations), len(hyperparameter_combinations) // num_processes)]

    seed = 0
    skf_outer = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
    for outer_iter, (train_ids, test_ids) in enumerate(skf_outer.split(features, labels)):
        train_features, train_labels = [features[id] for id in train_ids], [labels[id] for id in train_ids]
        test_features, test_labels = [features[id] for id in test_ids], [labels[id] for id in test_ids]
        testset = DummyDataset(test_features, test_labels)

        config["run"] = outer_iter
        config["seed"] = seed
        config["train_features"] = train_features
        config["train_labels"] = train_labels
        config["testset"] = testset
        
        with MyPool(num_processes) as pool:
            # Map the partial function over hyperparameter_chunks´
            pool.map(wrapper_function, hyperparameter_chunks)
        pool.close()


    with open(csv_filename, 'r') as file:
        df = pd.read_csv(file)

    # Specify the subset of columns you want to consider
    subset_columns = ['lr', 'gamma', 'emb_dims', 'emb_layers', 'hidden_size', 'n_heads', 'beta', 'dropout']

    # Identify sets of equal values for the specified columns
    unique_sets = df[subset_columns].apply(tuple, axis=1)

    # Create a dictionary to store the mean for each set of columns
    mean_dict = {}

    for unique_set in unique_sets.unique():
        # Find rows with the same set of columns
        rows_with_set = df.loc[unique_sets == unique_set]
        # Compute the mean for the specific column "col3"
        mean_value = rows_with_set["auc_val"].mean()

        # Store the mean in the dictionary with the unique set as the key
        mean_dict[unique_set] = mean_value

    hyperparameters = max(mean_dict, key=mean_dict.get)
    hyperparameters = tuple(int(value) if value.is_integer() else value for value in hyperparameters)

       # Retrieve the values from "col4" for the rows corresponding to max_mean_set
    aucs = df.loc[unique_sets == hyperparameters, 'auc_test'].mean()
    aucs_total = [aucs]
    seeds = [1,2,3,4]
    for seed in seeds:
        aucs_seed = []
        skf_outer = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
        for outer_iter, (train_ids, test_ids) in enumerate(skf_outer.split(features, labels)):
            train_features, train_labels = [features[id] for id in train_ids], [labels[id] for id in train_ids]
            test_features, test_labels = [features[id] for id in test_ids], [labels[id] for id in test_ids]
            testset = DummyDataset(test_features, test_labels)
            config["run"] = outer_iter
            config["seed"] = seed
            config["train_features"] = train_features
            config["train_labels"] = train_labels
            config["testset"] = testset

            train(hyperparameters)

                    # Read the CSV file into a DataFrame
            with open(csv_filename, 'r') as file:
                df = pd.read_csv(file)


            condition = (df.iloc[:, 2:-5] == hyperparameters).all(axis=1)

            # Create an additional condition for "run" equality with the next row
            run_condition = ((df['run'] == outer_iter) & (df['seed'] == seed))
            # Combine both conditions
            combined_condition = condition & run_condition
            aucs_seed.append(df[combined_condition]['auc_test'].values[0])
        aucs_total.append(sum(aucs_seed)/len(aucs_seed))
    print(f"dataset:{args.dataset}, method:{args.mode} auc:{sum(aucs_total)/len(aucs_total)}")
    df["acc_total"] = sum(aucs_total)/len(aucs_total)
    aucs_array = np.array(aucs_total)
    df["std_total"] = np.std(aucs_array)
    with open(csv_filename, 'w') as file:
        # Save the DataFrame to a CSV file
        df.to_csv(file, index=False)

if __name__ == '__main__':
    global args
    global csv_filename
    args = get_args()
    if args.mode == "sparsemap":
        csv_filename = f"results/{args.dataset}_{args.mode}_k_{args.k}.csv"
    elif args.mode == "normmax":
        csv_filename = f"results/{args.dataset}_{args.mode}_alpha_{args.alpha}.csv"
    else:
        csv_filename = f"results/{args.dataset}_{args.mode}.csv"
    main(args)
