import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import random




def load_data(dataset):
    data_dir = './dataset'
    if dataset == "mnist":
        train_dataset = datasets.MNIST(data_dir, train=True,download=True,
                                       transform=transforms.Compose(
                                           [transforms.ToTensor()]))
        test_dataset = datasets.MNIST(data_dir, train=False,download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                      ]))
    elif dataset == "fmnist":
        train_dataset = datasets.FashionMNIST(data_dir, train=True,download=True,
                                              transform=transforms.Compose(
                                                  [transforms.ToTensor()]))

        test_dataset = datasets.FashionMNIST(data_dir, train=False,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                             ]))
    elif dataset == "cifar10":
        train_dataset = datasets.CIFAR10(data_dir, train=True,download=True,
                                              transform=transforms.Compose(
                                                  [transforms.ToTensor()]))

        test_dataset = datasets.CIFAR10(data_dir, train=False,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                             ]))
    elif dataset == "cifar100":
        train_dataset = datasets.CIFAR100(data_dir, train=True,download=True,
                                              transform=transforms.Compose(
                                                  [transforms.ToTensor()]))

        test_dataset = datasets.CIFAR100(data_dir, train=False,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                             ]))
    elif dataset == "tiny":
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
            ])
        }
        data_dir = "dataset/tiny-imagenet-200/"
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'val', 'test']}
        train_dataset = image_datasets['train']
        test_dataset = image_datasets['val']
    else:
        raise NotImplementedError

    X_train, y_train = train_dataset.data, train_dataset.targets
    X_test, y_test = test_dataset.data, test_dataset.targets

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    return X_train, y_train, X_test, y_test, train_dataset, test_dataset


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    print('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def load_distil_dataset(dataset):
    X_train, y_train, X_test, y_test, train_dataset, test_dataset = load_data(dataset)
    return train_dataset

def partition_data(dataset, partition, beta=0.4, num_users=10):
    n_parties = num_users
    X_train, y_train, X_test, y_test, train_dataset, test_dataset = load_data(dataset)
    data_size = y_train.shape[0]

    net_dataidx_map = {}
    if partition == "iid":
        idxs = np.random.permutation(data_size)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "dirichlet":
        min_size = 0
        min_require_size = 10
        label = np.unique(y_test).shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(label):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)  # shuffle the label
                # random [0.5963643 , 0.03712018, 0.04907753, 0.1115522 , 0.2058858 ]
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array(   # 0 or x
                    [p * (len(idx_j) < data_size / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    train_data_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return train_dataset, test_dataset, net_dataidx_map, train_data_cls_counts

def partition_data1(dataset, partition, beta=0.4, num_users=10,
                   num_train_classes=9,
                   num_overlap_classes=9,
                   num_test_only_classes=1):

    n_parties = num_users
    X_train, y_train, X_test, y_test, train_dataset, test_dataset = load_data(dataset)
    data_size = y_train.shape[0]

    net_dataidx_map = {}

    # ================= IID =================
    if partition == "iid":
        idxs = np.random.permutation(data_size)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    # ================= Dirichlet =================
    elif partition == "dirichlet":
        min_size = 0
        min_require_size = 10
        label = np.unique(y_test).shape[0]

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(label):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array(
                    [p * (len(idx_j) < data_size / n_parties)
                     for p, idx_j in zip(proportions, idx_batch)]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min(len(idx_j) for idx_j in idx_batch)

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    # ================= Partial-OOD（新增） =================
    elif partition == "partial_ood":
        num_classes = np.unique(y_train).shape[0]
        all_classes = np.arange(num_classes)

        for client in range(n_parties):
            np.random.shuffle(all_classes)

            overlap_classes = set(all_classes[:num_overlap_classes])
            train_only_classes = set(
                all_classes[num_overlap_classes:num_train_classes]
            )
            # test_only_classes 只用于概念定义，不参与训练划分
            # test_only_classes = set(
            #     all_classes[num_train_classes:
            #                 num_train_classes + num_test_only_classes]
            # )

            train_classes = overlap_classes | train_only_classes

            idx = np.where(np.isin(y_train, list(train_classes)))[0]
            np.random.shuffle(idx)
            net_dataidx_map[client] = idx

    else:
        raise NotImplementedError

    train_data_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return train_dataset, test_dataset, net_dataidx_map, train_data_cls_counts

