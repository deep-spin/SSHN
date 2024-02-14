# %%
import sys
import argparse
# Import general modules used for e.g. plotting.
import torch
import pandas as pd
import itertools
import random
import numpy as np

# Import Hopfield-specific modules.
from hflayers import HopfieldPooling

# Import auxiliary modules.
from typing import Optional, Tuple

# Importing PyTorch specific modules.
from torch import Tensor
from torch.nn import Conv2d, Dropout, Linear, MaxPool2d, Module, ReLU, Sequential, Sigmoid
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score

import multiprocessing
from multiprocessing import Manager
from functools import partial
from multiprocessing.pool import Pool


from datasets.mnist_bags import MNISTBags

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

# %%
sys.path.append(r'./AttentionDeepMIL')

# %%
from model import Attention, GatedAttention

def get_args():
    global device
    global num_processes
    num_processes = 8
    device = "cuda:6"
    parser = argparse.ArgumentParser(description='Examples of MIL benchmarks:')
    parser.add_argument('--dataset', default='MNIST', type=str, choices=['MNIST'])
    parser.add_argument('--method', default='sparsemap', type=str, choices=["gated", "attention",'softmax', "entmax", "sparsemax" 'sparse', "adaptively", "sparsemap", "normmax"])
    parser.add_argument('--k_sparsemap', help='k exact ones for sparsemap', default=5, type=int)
    parser.add_argument('--train_size', help='train size', default=2000, type=int)
    parser.add_argument('--multiply', help='multiply features to get more columns', default=False, type=bool)
    parser.add_argument('--k_data', help='k data', default=5, type=int)
    parser.add_argument('--alpha', help='alpha for normmax', default=2, type=int)
    parser.add_argument('--mean', help='mean of bags size', default=14, type=int)
    parser.add_argument('--var', help='var of bags size', default=5, type=int)
    parser.add_argument('--normalize_hopfield_space', help='Apply y_psi normalization', default=False, type=bool)
    args = parser.parse_args()
    return args

# %%
def train_epoch(network: Module,
                optimizer: AdamW,
                data_loader: DataLoader
               ) -> Tuple[float, float, float]:
    """
    Execute one training epoch.
    
    :param network: network instance to train
    :param optimizer: optimizer instance responsible for updating network parameters
    :param data_loader: data loader instance providing training data
    :return: tuple comprising training loss, training error as well as accuracy
    """
    network.train()
    losses, errors, accuracies, aucs = [], [], [], []
    predictions, targets = [], []
    for data, target in data_loader:

        data, target = data.to(device=device), target[0].to(device=device)
        # Process data by Hopfield-based network.
        loss = network.calculate_objective(data, target)[0]

        # Update network parameters.
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(parameters=network.parameters(), max_norm=1.0, norm_type=2)
        optimizer.step()

        # Compute performance measures of current model.
        error, prediction, prob = network.calculate_classification_error(data, target)
        # Convert PyTorch tensors to NumPy arrays

        accuracy = (prediction == target).to(dtype=torch.float32).mean()
        accuracies.append(accuracy.detach().item())
        errors.append(error)
        losses.append(loss.detach().item())
        prediction = prediction.cpu().numpy()
        target = target.cpu()
        targets.append(target)
        predictions.append(prob.detach())
    predictions = [arr.item() for arr in predictions]
    auc = roc_auc_score(targets, predictions)
    # Report progress of validation procedure.
    return sum(losses) / len(losses), sum(errors) / len(errors), sum(accuracies) / len(accuracies), auc


def eval_iter(network: Module,
              data_loader: DataLoader
             ) -> Tuple[float, float, float]:
    """
    Evaluate the current model.
    
    :param network: network instance to evaluate
    :param data_loader: data loader instance providing validation data
    :return: tuple comprising validation loss, validation error as well as accuracy
    """
    network.eval()
    with torch.no_grad():
        losses, errors, accuracies = [], [], []
        predictions, targets = [], []
        for data, target in data_loader:
            data, target = data.to(device=device), target[0].to(device=device)
            
            data = torch.squeeze(data, dim=-1)
            # Process data by Hopfield-based network.
            loss = network.calculate_objective(data, target)[0]

            # Compute performance measures of current model.
            error, prediction, prob = network.calculate_classification_error(data, target)
            # Convert PyTorch tensors to NumPy arrays
            accuracy = (prediction == target).to(dtype=torch.float32).mean()
            accuracies.append(accuracy.detach().item())
            errors.append(error)
            losses.append(loss.detach().item())
            prediction = prediction.cpu().numpy()
            target = target.cpu()

            # Now, target_scalar contains the integer value
            targets.append(target)
            predictions.append(prob.detach())
        
        predictions = [arr.item() for arr in predictions]
        #print(predictions)
        auc = roc_auc_score(targets, predictions)
        # Report progress of validation procedure.
        return sum(losses) / len(losses), sum(errors) / len(errors), sum(accuracies) / len(accuracies), auc

    
def operate(network: Module,
            optimizer: AdamW,
            scheduler: torch.optim.lr_scheduler.ExponentialLR,
            data_loader_train: DataLoader,
            data_loader_val: DataLoader,
            data_loader_test: DataLoader,
            num_epochs: int = 1
           ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train the specified network by gradient descent using backpropagation.
    
    :param network: network instance to train
    :param optimizer: optimizer instance responsible for updating network parameters
    :param data_loader_train: data loader instance providing training data
    :param data_loader_eval: data loader instance providing validation data
    :param num_epochs: amount of epochs to train
    :return: data frame comprising training as well as evaluation performance
    """
    # Initialize variables and lists
    losses, errors, accuracies, aucs = {r'train': [], r'eval': []}, {r'train': [], r'eval': []}, {r'train': [], r'eval': []}, {r'train': [], r'eval': []}
    alphas = []
    best_eval_auc = float('-inf')  # Initialize with negative infinity
    patience = 5  # Number of epochs to wait for improvement
    no_improvement_count = 0

    for epoch in range(num_epochs):
        try:
            alphas.append(network.get_parameter_value())
        except:
            alphas.append(None)
        # Train network.
        performance = train_epoch(network, optimizer, data_loader_train)
        scheduler.step()
        losses[r'train'].append(performance[0])
        errors[r'train'].append(performance[1])
        accuracies[r'train'].append(performance[2])
        aucs[r'train'].append(performance[3])
        
        # Evaluate current model.
        performance = eval_iter(network, data_loader_val)
        losses[r'eval'].append(performance[0])
        errors[r'eval'].append(performance[1])
        accuracies[r'eval'].append(performance[2])
        eval_auc = performance[3]
        aucs[r'eval'].append(eval_auc)

        if eval_auc > best_eval_auc:
            performance1 = eval_iter(network, data_loader_test)
            test_acc = performance1[2]
            val_acc = performance[2]
            test_auc = performance1[3]
            best_eval_auc = eval_auc
            test_loss = performance1[0]
            val_loss = performance[0]
            no_improvement_count = 0  # Reset the counter
        else:
            no_improvement_count += 1
        
        if no_improvement_count >= patience:
            print(f"Early stopping at epoch {epoch} as no improvement in evaluation AUC.")
            break
    return val_loss, test_loss, best_eval_auc, test_auc, val_acc, test_acc

# %%
def set_seed(seed: int = 42) -> None:
    """
    Set seed for all underlying (pseudo) random number sources.
    
    :param seed: seed to be used
    :return: None
    """
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class HfPooling(Module):    
    def __init__(self, alpha = 1.0, normmax=False, sparseMAP = False, k=2, beta = 1.0, hidden_size = 32, alpha_as_static = True, n_heads = 1, dropout = 0.0, normalize_hopfield_space = False):
        """
        Initialize a new instance of a Hopfield-based pooling network.
        
        Note: all hyperparameters of the network are fixed for demonstration purposes.
        Morevover, most of the notation of the original implementation is kept in order
        to be easier comparable (partially ignoring PEP8).
        """
        super(HfPooling, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = Sequential(
            Conv2d(1, 20, kernel_size=5),
            ReLU(),
            MaxPool2d(2, stride=2),
            Conv2d(20, 50, kernel_size=5),
            ReLU(),
            MaxPool2d(2, stride=2)
        )
        self.feature_extractor_part2 = Sequential(
            Linear(50 * 4 * 4, self.L),
            ReLU(),
        )
        self.hopfield_pooling = HopfieldPooling(
            input_size=self.L, hidden_size=hidden_size, output_size=self.L, num_heads=n_heads, alpha = alpha, normmax=normmax, sparseMAP=sparseMAP, k=k, scaling = beta,
        alpha_as_static=alpha_as_static, dropout=dropout, normalize_hopfield_space=normalize_hopfield_space)
        self.dp = Dropout(
            p=0.1
        )
        self.classifier = Sequential(
            Linear(self.L * self.K, 1),
            Sigmoid()
        )
   
    def get_parameter_value(self):
        return self.hopfield_pooling.get_parameter_value()
    
    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Compute result of Hopfield-based pooling network on specified data.
        
        :param input: data to be processed by the Hopfield-based pooling network
        :return: result as computed by the Hopfield-based pooling network
        """

        x = input.squeeze(0).squeeze(0)
        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)
        H = H.unsqueeze(0)
        H = self.hopfield_pooling(H)
        H = H.squeeze(0)
        #H = self.dp(H)
        Y_prob = self.classifier(H)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, None

    def calculate_classification_error(self, input: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute classification error of current model.
        
        :param input: data to be processed by the Hopfield-based pooling network
        :param target: target to be used to compute the classification error of the current model
        :return: classification error as well as predicted class
        """
        Y = target.float()
        Y_prob, Y_hat, _ = self.forward(input)
        error = 1.0 - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat, Y_prob

    def calculate_objective(self, input: Tensor, target: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Compute objective of the current model.
        
        :param input: data to be processed by the Hopfield-based pooling network
        :param target: target to be used to compute the objective of the current model
        :return: objective as well as dummy A (see accompanying paper for more information)
        """
        Y = target.float()
        Y_prob, _, A = self.forward(input)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=(1.0 - 1e-5))
        neg_log_likelihood = -1.0 * (Y * torch.log(Y_prob) + (1.0 - Y) * torch.log(1.0 - Y_prob))

        return neg_log_likelihood, A


def train(hyperparameters):
    global device
    set_seed()
    args = config["args"]
    data_loader_train = config["data_loader_train"]
    data_loader_test = config["data_loader_test"]
    data_loader_val = config["data_loader_val"]
    run = config["run"]
    if args.method == "attention":
        n_epochs, lr, gamma = hyperparameters
        hyperparameters_dict = {"run": run,
            "n_epochs": hyperparameters[0],
            "lr": hyperparameters[1],
            "gamma": hyperparameters[2],
        }
        network = Attention().to(device=device)

    
    elif args.method == "gated":
        hyperparameters_dict = {"run": run,
            "n_epochs": hyperparameters[0],
            "lr": hyperparameters[1],
            "gamma": hyperparameters[2],
        }
        n_epochs, lr, gamma = hyperparameters
        network = GatedAttention().to(device=device)
    
    elif args.method == "sparsemap":
        hyperparameters_dict = {"run": run,
            "n_epochs": hyperparameters[0],
            "lr": hyperparameters[1],
            "gamma": hyperparameters[2],
            "hidden_size": hyperparameters[3],
            "n_heads": hyperparameters[4],
            "beta": hyperparameters[5],
            "dropout": hyperparameters[6],
            "alpha": None
        }
        n_epochs, lr, gamma, hidden_size, n_heads, beta, dropout = hyperparameters
        beta_rsh = torch.full((int(n_heads),), float(beta), dtype=torch.float).to(device)
        network = HfPooling(sparseMAP = True, k=args.k_sparsemap, beta = beta_rsh, hidden_size = int(hidden_size), alpha_as_static=True, n_heads = int(n_heads), dropout = dropout, normalize_hopfield_space=args.normalize_hopfield_space).to(device=device)
    
    elif args.method == "adaptively":
        hyperparameters_dict = {"run": run,
            "n_epochs": hyperparameters[0],
            "lr": hyperparameters[1],
            "gamma": hyperparameters[2],
            "hidden_size": hyperparameters[3],
            "n_heads": hyperparameters[4],
            "beta": hyperparameters[5],
            "alpha": hyperparameters[6],
            "dropout": hyperparameters[7],
        }
        #alpha = torch.rand(n_heads) * 0.99 + 1.01
        n_epochs, lr, gamma, hidden_size, n_heads, beta, alpha, dropout = hyperparameters
        beta_rsh = torch.full((int(n_heads),), float(beta), dtype=torch.float).to(device)
        alpha_rsh = torch.full((int(n_heads),), alpha, dtype=torch.float).to(device)
        network = HfPooling(alpha = alpha_rsh, beta = beta_rsh, hidden_size = int(hidden_size), alpha_as_static=False, n_heads = int(n_heads), dropout = dropout).to(device=device)
    
    elif args.method == "sparse":
        hyperparameters_dict = {"run": run,
            "n_epochs": hyperparameters[0],
            "lr": hyperparameters[1],
            "gamma": hyperparameters[2],
            "hidden_size": hyperparameters[3],
            "n_heads": hyperparameters[4],
            "beta": hyperparameters[5],
            "alpha": hyperparameters[6],
            "dropout": hyperparameters[7],
        }
        #alpha = torch.rand(n_heads) * 0.99 + 1.01
        n_epochs, lr, gamma, hidden_size, n_heads, beta, alpha, dropout = hyperparameters
        beta_rsh = torch.full((int(n_heads),), float(beta), dtype=torch.float).to(device)
        alpha_rsh = torch.full((int(n_heads),), alpha, dtype=torch.float).to(device)
        network = HfPooling(alpha = alpha_rsh, beta = beta_rsh, hidden_size = int(hidden_size), alpha_as_static=True, n_heads = int(n_heads), dropout = dropout).to(device=device)
    
    elif args.method == "sparsemax":
        alpha = 2.0
        hyperparameters_dict = {"run": run,
            "n_epochs": hyperparameters[0],
            "lr": hyperparameters[1],
            "gamma": hyperparameters[2],
            "hidden_size": hyperparameters[3],
            "n_heads": hyperparameters[4],
            "beta": hyperparameters[5],
            "dropout": hyperparameters[6],
            "alpha": alpha,
        }
        n_epochs, lr, gamma, hidden_size, n_heads, beta, dropout = hyperparameters
        beta_rsh = torch.full((int(n_heads),), float(beta), dtype=torch.float).to(device)
        alpha_rsh = torch.full((int(n_heads),), alpha, dtype=torch.float).to(device)
        network = HfPooling(alpha = alpha_rsh, beta = beta_rsh, hidden_size = int(hidden_size), n_heads = int(n_heads), dropout=dropout, normalize_hopfield_space=args.normalize_hopfield_space).to(device=device)
    
    elif args.method == "normmax":
        hyperparameters_dict = {"run": run,
            "n_epochs": hyperparameters[0],
            "lr": hyperparameters[1],
            "gamma": hyperparameters[2],
            "hidden_size": hyperparameters[3],
            "n_heads": hyperparameters[4],
            "beta": hyperparameters[5],
            "dropout": hyperparameters[6],
            "alpha": args.alpha,
        }
        n_epochs, lr, gamma, hidden_size, n_heads, beta, dropout = hyperparameters
        beta_rsh = torch.full((int(n_heads),), float(beta), dtype=torch.float).to(device)
        network = HfPooling(normmax = True, alpha = args.alpha, beta = beta_rsh, hidden_size = int(hidden_size), n_heads = int(n_heads), dropout=dropout, normalize_hopfield_space=args.normalize_hopfield_space).to(device=device)
    
    elif args.method == "entmax":
        alpha = 1.5
        hyperparameters_dict = {"run": run,
            "n_epochs": hyperparameters[0],
            "lr": hyperparameters[1],
            "gamma": hyperparameters[2],
            "hidden_size": hyperparameters[3],
            "n_heads": hyperparameters[4],
            "beta": hyperparameters[5],
            "dropout": hyperparameters[6],
            "alpha": alpha,
        }
        n_epochs, lr, gamma, hidden_size, n_heads, beta, dropout = hyperparameters
        beta_rsh = torch.full((int(n_heads),), float(beta), dtype=torch.float).to(device)
        alpha_rsh = torch.full((int(n_heads),), alpha, dtype=torch.float).to(device)
        network = HfPooling(alpha = alpha_rsh, beta = beta_rsh, hidden_size = int(hidden_size), n_heads = int(n_heads), dropout=dropout, normalize_hopfield_space=args.normalize_hopfield_space).to(device=device)
    
    elif args.method == "softmax":
        alpha = 1.0
        hyperparameters_dict = {"run": run,
            "n_epochs": hyperparameters[0],
            "lr": hyperparameters[1],
            "gamma": hyperparameters[2],
            "hidden_size": hyperparameters[3],
            "n_heads": hyperparameters[4],
            "beta": hyperparameters[5],
            "dropout": hyperparameters[6],
            "alpha": alpha,
        }
        n_epochs, lr, gamma, hidden_size, n_heads, beta, dropout = hyperparameters
        beta_rsh = torch.full((int(n_heads),), float(beta), dtype=torch.float).to(device)
        alpha_rsh = torch.full((int(n_heads),), alpha, dtype=torch.float).to(device)
        network = HfPooling(alpha = alpha_rsh, beta = beta_rsh, hidden_size = int(hidden_size), n_heads = int(n_heads), dropout=dropout, normalize_hopfield_space=args.normalize_hopfield_space).to(device=device)
    
    optimizer = AdamW(params=network.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    val_loss, test_loss, val_auc, test_auc, val_acc, test_acc = operate(
        network=network,
        optimizer=optimizer,
        scheduler = scheduler,
        data_loader_train=data_loader_train,
        data_loader_val=data_loader_val,
        data_loader_test=data_loader_test,
        num_epochs=int(n_epochs))
    

    hyperparameters_dict['loss_test'], hyperparameters_dict["loss_val"], hyperparameters_dict["auc_test"], hyperparameters_dict["auc_val"], hyperparameters_dict["acc_test"], hyperparameters_dict["acc_val"]  = test_loss, val_loss,test_auc, val_auc, test_acc, val_acc
    
    # Append the data row to the DataFrame
    df_list.append(hyperparameters_dict)
    # Convert the shared list to a DataFrame
    df = pd.DataFrame(list(df_list))
    
    with open(csv_filename, 'w') as file:
        # Save the DataFrame to a CSV file
        df.to_csv(file, index=False)

def wrapper_function(hyperparameter_chunk):
    for hyperparameters in hyperparameter_chunk:
        train(hyperparameters)

if __name__ == '__main__':
    args = get_args()
    global csv_filename
    global config
    if args.method == "sparsemap":
        csv_filename = f"/results/MNIST_k_{args.k_data}_{args.method}_k_{args.k_sparsemap}_trainsize_{args.train_size}.csv"
    elif args.method == "normmax":
        csv_filename = f"/results/MNIST_k_{args.k_data}_{args.method}_alpha_{args.alpha}_trainsize_{args.train_size}.csv"
    else:
        csv_filename = f"/results/MNIST_k_{args.k_data}_{args.method}_trainsize_{args.train_size}.csv"
    
    if args.method == "entmax" or args.method == "sparsemax" or args.method == "softmax" or args.method == "sparsemap" or args.method == "normmax":
        config = {
            "n_epochs": [50],
            "lr": [1e-5, 1e-6],
            "gamma": [0.98, 0.96],
            "hidden_size": [16, 64],
            "num_heads": [8, 16],
            "beta": [0.25, 1.0, 2.0, 8.0],
            "dropout": [0.0, 0.75]
        }

    elif args.method == "sparse" or args.method == "adaptively":
        config = {
            "n_epochs": [50],
            "lr": [1e-4, 1e-5],
            "gamma": [0.98, 0.96],
            "hidden_size": [16, 64],
            "num_heads": [8, 16],
            "beta": [0.25, 1.0, 2.0, 8.0],
            "alpha": [1.2, 1.4, 1.6, 1.8],
            "dropout": [0.0, 0.75]
        }

    else:
        config = {
            "n_epochs": [50],
            "lr": [1e-4, 1e-5],
            "gamma": [0.98, 0.96]}
    seed = 50000 #Use seeds at least traindataset size seeds larger
    eval_bags, eval_labels = MNISTBags(
        target_number=9,
        mean_bag_length=args.mean,
        var_bag_length=args.var,
        num_bag=1000,
        pos_per_bag=args.k_data,
        train=False,
        seed = seed
    )._create_bags()  

    torch.manual_seed(42)
    # Create DataLoader for the test set
    data_loader_test = DataLoader(list(zip(eval_bags[:500], eval_labels[:500])), batch_size=1, shuffle=True, num_workers=8)

    # Create DataLoader for the validation set
    data_loader_val = DataLoader(list(zip(eval_bags[500:], eval_labels[500:])), batch_size=1, shuffle=True, num_workers=8)

    # Generate all combinations of hyperparameters
    hyperparameter_combinations = list(itertools.product(*config.values()))
    hyperparameter_chunks = [hyperparameter_combinations[i:i + len(hyperparameter_combinations) // num_processes] for i in range(0, len(hyperparameter_combinations), len(hyperparameter_combinations) // num_processes)]
    seed = 0

    data_loader_train = DataLoader(MNISTBags(
            target_number=9,
            mean_bag_length=args.mean,
            var_bag_length=args.var,
            num_bag=args.train_size,
            train=True,
            pos_per_bag=args.k_data,
            seed = seed
        ), batch_size=1, shuffle=True, num_workers=8)
    run = 0

    # Create a partial function with fixed arguments
    #partial_wrapper_function = partial(wrapper_function, args, data_loader_train, data_loader_test, data_loader_val, run)
    config["args"] = args
    config["data_loader_train"] = data_loader_train
    config["data_loader_test"] = data_loader_test
    config["data_loader_val"] = data_loader_val
    config["run"] = run  
    
    with MyPool(num_processes) as pool:
      # Map the partial function over hyperparameter_chunks´
        pool.map(wrapper_function, hyperparameter_chunks)
    pool.close()

    seeds = [10000, 20000, 30000, 40000]
    with open(csv_filename, 'r') as file:
        df = pd.read_csv(file)

    max_auc_value = df['acc_val'].max()
    best_auc_rows = df[df['acc_val'] == max_auc_value]
    max_row = best_auc_rows.loc[best_auc_rows['loss_val'].idxmin()]
    accs = [max_row['acc_test']]
    #Include alpha or not
    hyperparameters = tuple(max_row.iloc[1:-6]) if args.method in ["adaptively", "sparse", "gated", "attention"] else tuple(max_row.iloc[1:-7])

    for i, seed in enumerate(seeds):
        # Create data loader of training set.
        data_loader_train = DataLoader(MNISTBags(
            target_number=9,
            mean_bag_length=args.mean,
            var_bag_length=args.var,
            num_bag=args.train_size,
            train=True,
            pos_per_bag=args.k_data,
            seed = seed
        ), batch_size=1, shuffle=True, num_workers=8)
        
        config["data_loader_train"] = data_loader_train
        config["run"] = i + 1
        train(hyperparameters)
        # Read the CSV file into a DataFrame
        with open(csv_filename, 'r') as file:
            df = pd.read_csv(file)

        # Assuming hyperparameters is obtained as described earlier
        if args.method in ["adaptively", "sparse", "gated", "attention"]:
            condition = (df.iloc[:, 1:-6] == hyperparameters).all(axis=1)
        else:
            condition = (df.iloc[:, 1:-7] == hyperparameters).all(axis=1)
        # Create an additional condition for "run" equality with the next row
        run_condition = (df['run'] == i + 1)
        # Combine both conditions
        combined_condition = condition & run_condition
        accs.append(df[combined_condition]['acc_test'].values[0])

    print(f"dataset:{args.dataset}, method:{args.method} auc:{sum(accs)/len(accs)}")
    df["acc_total"] = sum(accs)/len(accs)
    aucs_array = np.array(accs)
    df["std_total"] = np.std(aucs_array)
    # Save the DataFrame to a CSV file
    with open(csv_filename, 'w') as file:
        # Save the DataFrame to a CSV file
        df.to_csv(file, index=False)
        # Save the DataFrame to a CSV file