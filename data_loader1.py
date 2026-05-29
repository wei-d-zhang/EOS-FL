import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset, Dataset
import numpy as np
import matplotlib.pyplot as plt

#torch.manual_seed(543)
np.random.seed(500)

def create_client_dataloaders(dataname,num_clients, alpha, batch_size, test_ratio=0.2):
    if dataname == 'Fashion':
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif dataname == 'Cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataname == 'Cifar100':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    else:
        print("coming soon")
        return

    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    train_labels = np.array(train_dataset.targets)
    test_labels = np.array(test_dataset.targets)

    test_labels_indices_offset = len(train_dataset)
    test_labels_indices = np.arange(len(test_labels)) + test_labels_indices_offset

    labels = np.concatenate((train_labels, test_labels))
    indices = np.concatenate((np.arange(len(train_labels)), test_labels_indices))

    unique_labels = np.unique(labels)
    label_indices = {label: indices[labels == label] for label in unique_labels}

    dirichlet_dist = np.random.dirichlet(alpha * np.ones(num_clients), size=len(unique_labels))

    client_datasets = []
    for client_idx in range(num_clients):
        client_data_indices = []
        for label_idx, (class_label, class_idx) in enumerate(label_indices.items()):
            num_samples_class = int(len(class_idx) * dirichlet_dist[label_idx][client_idx])
            client_data_indices.extend(np.random.choice(class_idx, num_samples_class, replace=False))
        client_datasets.append(Subset(combined_dataset, client_data_indices))
        
    client_train_datasets = []
    client_test_datasets = []
    for client_dataset in client_datasets:
        train_size = int((1 - test_ratio) * len(client_dataset))
        test_size = len(client_dataset) - train_size
        client_train_dataset, client_test_dataset = random_split(client_dataset, [train_size, test_size])
        client_train_datasets.append(client_train_dataset)
        client_test_datasets.append(client_test_dataset)

    client_train_loaders = [DataLoader(client_train_dataset, batch_size=batch_size, shuffle=True) for client_train_dataset in client_train_datasets]
    client_test_loaders = [DataLoader(client_test_dataset, batch_size=batch_size, shuffle=False) for client_test_dataset in client_test_datasets]

    return client_train_loaders, client_test_loaders

def create_client_dataloaders_pathological(dataname, num_clients, num_shards, batch_size, test_ratio=0.2):
    if dataname == 'Fashion':
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif dataname == 'Cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataname == 'Cifar100':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    else:
        print("coming soon")
        return
        
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    train_labels = np.array(train_dataset.targets)
    test_labels = np.array(test_dataset.targets)

    test_labels_indices_offset = len(train_dataset)
    test_labels_indices = np.arange(len(test_labels)) + test_labels_indices_offset
    labels = np.concatenate((train_labels, test_labels))
    indices = np.concatenate((np.arange(len(train_labels)), test_labels_indices))

    sorted_indices = np.argsort(labels)
    sorted_labels = labels[sorted_indices]
    num_samples_per_shard = len(labels) // num_shards

    client_datasets = [[] for _ in range(num_clients)]
    shard_indices = np.array_split(sorted_indices, num_shards)

    shards_per_client = num_shards // num_clients
    for client_idx in range(num_clients):
        assigned_shards = np.random.choice(num_shards, shards_per_client, replace=False)
        for shard_idx in assigned_shards:
            client_datasets[client_idx].extend(shard_indices[shard_idx])

    client_train_datasets = []
    client_test_datasets = []
    for client_data_indices in client_datasets:
        client_dataset = Subset(combined_dataset, client_data_indices)
        train_size = int((1 - test_ratio) * len(client_dataset))
        test_size = len(client_dataset) - train_size
        client_train_dataset, client_test_dataset = random_split(client_dataset, [train_size, test_size])
        client_train_datasets.append(client_train_dataset)
        client_test_datasets.append(client_test_dataset)

    client_train_loaders = [DataLoader(client_train_dataset, batch_size=batch_size, shuffle=True) for client_train_dataset in client_train_datasets]
    client_test_loaders = [DataLoader(client_test_dataset, batch_size=batch_size, shuffle=False) for client_test_dataset in client_test_datasets]

    return client_train_loaders, client_test_loaders

def create_iid_client_dataloaders(dataname, num_clients, batch_size, test_ratio=0.2):
    if dataname == 'Fashion':
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif dataname == 'Cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataname == 'Cifar100':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    else:
        print("coming soon")
        return

    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    
    total_data_size = len(combined_dataset)
    client_data_size = total_data_size // num_clients

    indices = np.random.permutation(total_data_size)
    client_datasets = [Subset(combined_dataset, indices[i * client_data_size: (i + 1) * client_data_size]) 
                       for i in range(num_clients)]

    client_train_datasets = []
    client_test_datasets = []
    for client_dataset in client_datasets:
        train_size = int((1 - test_ratio) * len(client_dataset))
        test_size = len(client_dataset) - train_size
        client_train_dataset, client_test_dataset = random_split(client_dataset, [train_size, test_size])
        client_train_datasets.append(client_train_dataset)
        client_test_datasets.append(client_test_dataset)

    client_train_loaders = [DataLoader(client_train_dataset, batch_size=batch_size, shuffle=True) 
                            for client_train_dataset in client_train_datasets]
    client_test_loaders = [DataLoader(client_test_dataset, batch_size=batch_size, shuffle=False) 
                           for client_test_dataset in client_test_datasets]

    return client_train_loaders, client_test_loaders

