"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import random


"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import random


class MNISTBags(data_utils.Dataset):
    def __init__(
            self,
            target_number=9,
            mean_bag_length=10,
            var_bag_length=2,
            num_bag=500,
            pos_per_bag=1,
            seed=1,
            train=True):
        self.target_number = target_number
        self.pos_per_bag = pos_per_bag
        self.train = train
        self.num_bag = num_bag
        self.bag_size_variance = var_bag_length
        self.bag_size_mean = mean_bag_length
        self.r = np.random.RandomState(seed)
        self.seed = seed

        self.num_in_train = 60000
        self.num_in_test = 10000

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        if self.train:
            loader = data_utils.DataLoader(
                datasets.MNIST(
                    '../datasets',
                    train=True,
                    download=True,
                    transform=transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(
                                (0.1307,
                                 ),
                                (0.3081,
                                 ))])),
                batch_size=self.num_in_train,
                shuffle=False)
        else:
            loader = data_utils.DataLoader(
                datasets.MNIST(
                    '../datasets',
                    train=False,
                    download=True,
                    transform=transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(
                                (0.1307,
                                 ),
                                (0.3081,
                                 ))])),
                batch_size=self.num_in_test,
                shuffle=False)

        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        bags_list = []
        labels_list = []

        pos_idx = [i for i, j in enumerate(
            all_labels) if j == self.target_number]
        neg_idx = [i for i, j in enumerate(
            all_labels) if j != self.target_number]

        pos_images = []
        neg_images = []

        for i, img in enumerate(all_imgs):
            if all_labels[i] == self.target_number:
                pos_images.append(img)
            else:
                neg_images.append(img)

        self.all_pos_img = pos_images
        self.all_neg_img = neg_images

        for i in range(int(self.num_bag/2)):
            # Increment the seed for each iteration
            seed_for_iteration = self.seed + i

            # Create a new random state for this iteration
            random_state_for_iteration = np.random.RandomState(seed_for_iteration)


            bag_length = int(random_state_for_iteration.normal(self.bag_size_mean, self.bag_size_variance, 1))
            bag_length = max(self.pos_per_bag, bag_length)
            _pos_per_bag = int(random_state_for_iteration.randint(self.pos_per_bag, bag_length + 1))

            # Randomly sample positive and negative instances
            _pos_idx = random_state_for_iteration.choice(pos_idx, _pos_per_bag, replace=False).tolist() + \
                        random_state_for_iteration.choice(neg_idx, bag_length - _pos_per_bag, replace=False).tolist()

            if self.pos_per_bag == 1:
                _neg_idx = random_state_for_iteration.choice(neg_idx, bag_length, replace=False).tolist()
            else:
               _pos_per_bag = random_state_for_iteration.randint(0, self.pos_per_bag)
               _neg_idx = random_state_for_iteration.choice(pos_idx, _pos_per_bag, replace=False).tolist() + \
                    random_state_for_iteration.choice(neg_idx, bag_length - _pos_per_bag, replace=False).tolist()
               
            # Assert that the size of positive and negative instances in the bag is the same
            assert len(_pos_idx) == len(_neg_idx)
            bags_list.append(all_imgs[_neg_idx])
            labels_list.append(0)
            bags_list.append(all_imgs[_pos_idx])
            labels_list.append(1)

        return bags_list, torch.tensor(labels_list)

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = self.train_labels_list[index]
        else:
            bag = self.test_bags_list[index]
            label = self.test_labels_list[index]

        return bag, label
