import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset, Dataset
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from collections import defaultdict
import random

#torch.manual_seed(543)
np.random.seed(500)

def create_client_dataloaders_partial_ood(dataname,
                                         num_clients,
                                         num_train_classes,
                                         num_overlap_classes,
                                         batch_size):
    """
    简化实现：
    - 取 CIFAR-10
    - 随机挑选 num_overlap_classes 作为公共 overlap 类（出现在所有客户端的训练集中）
    - 对余下 10 - num_overlap_classes 个类随机分配给每个客户端，保证每 client 包含 num_train_classes（不含 overlap）
    - 每个 client 的 train dataset 由这些类的样本组成
    - test dataset = train_classes + 剩余类（模拟 test 有一些 train 没有的类）
    返回： train_loaders list, test_loaders list, info dict
    """
    assert dataname == 'Cifar10', "当前实现只支持 CIFAR-10"
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    num_classes = 10
    class_to_indices_train = defaultdict(list)
    class_to_indices_test = defaultdict(list)
    for idx, (_, label) in enumerate(trainset):
        class_to_indices_train[label].append(idx)
    for idx, (_, label) in enumerate(testset):
        class_to_indices_test[label].append(idx)

    all_classes = list(range(num_classes))
    overlap_classes = random.sample(all_classes, num_overlap_classes)
    remaining_classes = [c for c in all_classes if c not in overlap_classes]

    train_loaders = []
    test_loaders = []
    client_train_classes = []

    for i in range(num_clients):
        # pick non-overlap train classes for this client
        need = num_train_classes
        # ensure there's enough unique classes; allow repeats across clients
        chosen_non_overlap = random.sample(remaining_classes, need)
        train_classes = sorted(list(set(chosen_non_overlap + overlap_classes)))
        client_train_classes.append(train_classes)

        # build train indices
        train_indices = []
        for c in train_classes:
            # sample subset of class indices to reduce local dataset size
            available = class_to_indices_train[c]
            take_n = min(len(available), 500)  # cap per-class samples
            train_indices += random.sample(available, take_n)

        test_classes = list(set(train_classes + [c for c in all_classes if c not in train_classes]))  # test includes all classes (so includes missing ones)
        # build test indices: use standard testset indices but could subset
        test_indices = []
        for c in test_classes:
            available = class_to_indices_test[c]
            test_indices += available  # use full test class set

        train_loader = DataLoader(Subset(trainset, train_indices), batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
        test_loader = DataLoader(Subset(testset, test_indices), batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    info = {
        'client_train_classes': client_train_classes,
        'overlap_classes': overlap_classes
    }
    return train_loaders, test_loaders, info

def create_client_dataloaders_partial_ood1(
        dataname, 
        num_clients,
        num_train_classes,       # 每个客户端训练集的类别数
        num_overlap_classes,     # 训练和测试共有的类别数
        num_test_only_classes,   # 仅测试独有的类别数
        batch_size):

    # ---------- 加载数据 ----------
    if dataname == 'Fashion':
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif dataname == 'Cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif dataname == 'Cifar100':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        num_classes = 100
    elif dataname == 'Gear':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        full_dataset = datasets.ImageFolder(root='./data/Gear', transform=transform)
        num_classes = len(full_dataset.classes)
        
        # 按类别按 8:2 分层划分
        targets = np.array(full_dataset.targets)
        indices = np.arange(len(full_dataset))
        
        train_idx_list, test_idx_list = [], []
        rng = np.random.RandomState(500)
        for c in range(num_classes):
            cls_idx = indices[targets == c]
            rng.shuffle(cls_idx)
            split = int(0.8 * len(cls_idx))
            train_idx_list.extend(cls_idx[:split])
            test_idx_list.extend(cls_idx[split:])
            
        train_dataset = Subset(full_dataset, train_idx_list)
        test_dataset = Subset(full_dataset, test_idx_list)
    else:
        print("Dataset not implemented")
        return
    
    def get_targets(dataset):
        if hasattr(dataset, 'targets'):
            return np.array(dataset.targets)
        elif isinstance(dataset, Subset):
            return np.array(dataset.dataset.targets)[dataset.indices]
        else:
            raise AttributeError("Dataset object has no targets attribute")

    train_labels = get_targets(train_dataset)
    test_labels = get_targets(test_dataset)

    all_classes = np.arange(num_classes)

    client_train_loaders = []
    client_test_loaders = []

    client_class_info = []

    for client_idx in range(num_clients):
        np.random.shuffle(all_classes)

        # ---------- 类别划分 ----------
        overlap_classes = set(all_classes[:num_overlap_classes])  
        train_only_classes = set(all_classes[num_overlap_classes:num_train_classes])
        test_only_classes = set(all_classes[num_train_classes:num_train_classes + num_test_only_classes])

        # 保存信息便于检查
        client_class_info.append({
            'overlap': overlap_classes,
            'train_only': train_only_classes,
            'test_only': test_only_classes
        })

        # ---------- 构建训练集 ----------
        train_classes = overlap_classes.union(train_only_classes)
        train_idx = np.where(np.isin(train_labels, list(train_classes)))[0]

        # ---------- 构建测试集 ----------
        test_classes = overlap_classes.union(test_only_classes)
        test_idx = np.where(np.isin(test_labels, list(test_classes)))[0]

        # ------------ DataLoader ------------
        train_loader = DataLoader(Subset(train_dataset, train_idx),
                                  batch_size=batch_size, shuffle=True)

        test_loader = DataLoader(Subset(test_dataset, test_idx),
                                 batch_size=batch_size, shuffle=False)

        client_train_loaders.append(train_loader)
        client_test_loaders.append(test_loader)

    return client_train_loaders, client_test_loaders, client_class_info


def create_client_dataloaders_partial_ood_dirichlet(
        dataname,
        num_clients,
        num_train_classes,
        num_overlap_classes,
        num_test_only_classes,
        alpha,               # Dirichlet 参数
        batch_size=64):

    # ---------- 加载数据 ----------
    if dataname == 'Fashion':
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif dataname == 'Cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif dataname == 'Cifar100':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        num_classes = 100
    else:
        print("Dataset not implemented")
        return

    # ------ 提取标签 ------
    train_labels = np.array(train_dataset.targets)
    test_labels = np.array(test_dataset.targets)

    # 记录每个客户端的类别划分信息
    all_classes = np.arange(num_classes)
    client_class_info = []

    # 每个客户端最终的数据索引
    client_train_indices = [[] for _ in range(num_clients)]
    client_test_indices = [[] for _ in range(num_clients)]


    # ======================================================
    #   Step 1：为每个客户端分配 overlap / train-only / test-only 类别
    # ======================================================
    for client_idx in range(num_clients):
        np.random.shuffle(all_classes)

        overlap = set(all_classes[:num_overlap_classes])
        train_only = set(all_classes[num_overlap_classes : num_train_classes])
        test_only = set(all_classes[num_train_classes : num_train_classes + num_test_only_classes])

        client_class_info.append({
            "overlap": overlap,
            "train_only": train_only,
            "test_only": test_only
        })


    # ======================================================
    #   Step 2：对每一类执行 Dirichlet 分配，为训练和测试各自划分样本
    # ======================================================
    def dirichlet_split(indices_per_class, is_train_part=True):
        """
        indices_per_class: dict[class → indices]
        return: dict[client → indices]
        is_train_part: True=训练集, False=测试集
        """
        client_indices = {i: [] for i in range(num_clients)}

        for cls, idx_list in indices_per_class.items():
            if len(idx_list) == 0:
                continue

            # 根据 Dirichlet 分布按比例分配给客户端
            proportions = np.random.dirichlet(alpha=[alpha] * num_clients)
            proportions = (proportions / proportions.sum())  # ensure normalization

            # 按比例切分
            sizes = (proportions * len(idx_list)).astype(int)
            # 如果因为四舍五入总数不一致，补到最后一个客户端
            sizes[-1] = len(idx_list) - sizes[:-1].sum()

            start = 0
            for cid in range(num_clients):
                part = idx_list[start : start + sizes[cid]]
                client_indices[cid].extend(part)
                start += sizes[cid]

        return client_indices


    # ======================================================
    #   Step 3：训练集按 Dirichlet 分
    # ======================================================
    train_classes_per_client = [info["overlap"].union(info["train_only"]) for info in client_class_info]

    # 对于每个客户端，训练集中允许的类别不同，因此需要逐类处理
    train_classes_unique = set().union(*train_classes_per_client)

    for cls in train_classes_unique:
        # 这类属于训练标签的样本
        cls_indices = np.where(train_labels == cls)[0].tolist()

        # 生成 per-client 划分
        split = dirichlet_split({cls: cls_indices})

        # 仅允许分配给"包含该类"的客户端
        for cid in range(num_clients):
            if cls in train_classes_per_client[cid]:
                client_train_indices[cid].extend(split[cid])


    # ======================================================
    #   Step 4：测试集按 Dirichlet 分
    # ======================================================
    test_classes_per_client = [info["overlap"].union(info["test_only"]) for info in client_class_info]

    test_classes_unique = set().union(*test_classes_per_client)

    for cls in test_classes_unique:
        cls_indices = np.where(test_labels == cls)[0].tolist()

        split = dirichlet_split({cls: cls_indices})

        for cid in range(num_clients):
            if cls in test_classes_per_client[cid]:
                client_test_indices[cid].extend(split[cid])


    # ======================================================
    #   Step 5：构建 DataLoader
    # ======================================================
    client_train_loaders = []
    client_test_loaders = []

    for cid in range(num_clients):
        train_loader = DataLoader(
            Subset(train_dataset, client_train_indices[cid]),
            batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            Subset(test_dataset, client_test_indices[cid]),
            batch_size=batch_size, shuffle=False
        )
        client_train_loaders.append(train_loader)
        client_test_loaders.append(test_loader)

    return client_train_loaders, client_test_loaders, client_class_info
