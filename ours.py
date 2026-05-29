# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import csv
from torchvision import datasets, transforms
from torch.utils.data import Subset
from model import *
from data_loader1 import *
import torch.nn.functional as F
import time
from overlap_data import create_client_dataloaders_partial_ood1

torch.manual_seed(643)
np.random.seed(600)


# ===============================================================
#                       Client Class
# ===============================================================
class Client(object):
    def __init__(self, local_trainloader, local_testloader, args, client_id):
        self.trainloader = local_trainloader
        self.testloader = local_testloader
        self.args = args
        self.client_id = client_id

        model_mapping = {
            'Fashion': Fashion_ResNet18,
            'Cifar10': cifar10_ResNet18,
            'Cifar100': cifar100_ResNet18
        }
        self.net = model_mapping[self.args.dataset]().to(self.args.device)

        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, net):
        net.train()
        for epoch in range(self.args.local_epochs):
            correct, total = 0, 0
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                self.optimizer.zero_grad()
                outputs = net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f"Client {self.client_id} | Epoch {epoch+1} | Train Acc: {100*correct/total:.2f}%")

        return self.net.state_dict(), self.get_train_classes()

    def test(self, net):
        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100.0 * correct / total
    
    def fine_tune(self, net):
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.args.lr * 0.1)
        for epoch in range(self.args.ft_epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                loss = self.criterion(net(inputs), labels)
                loss.backward()
                optimizer.step()
        return net.state_dict()

    def get_train_classes(self):
        classes = set()
        for _, labels in self.trainloader:
            labels = labels.cpu().numpy()
            for l in np.unique(labels):
                classes.add(int(l))
        return sorted(list(classes))


# ===============================================================
#             Load Public Test Set for Sensitivity Analysis
# ===============================================================
def load_public_testset(dataset_name):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if dataset_name == 'Cifar10':
        return datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    elif dataset_name == 'Cifar100':
        return datasets.CIFAR100('./data', train=False, download=True, transform=transform)
    elif dataset_name == 'Fashion':
        transform1 = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
        return datasets.FashionMNIST('./data', train=False, download=True, transform=transform1)
    else:
        raise ValueError("Unknown dataset:", dataset_name)


# ===============================================================
#                       FedAvg with Grafting
# ===============================================================
class FedAvg(object):
    def __init__(self, clients, args, samples_per_class=20):
        self.clients = clients
        self.args = args
        self.samples_per_class = samples_per_class

        # Global model
        model_mapping = {
            'Fashion': Fashion_ResNet18,
            'Cifar10': cifar10_ResNet18,
            'Cifar100': cifar100_ResNet18
        }
        self.global_model = model_mapping[self.args.dataset]().to(self.args.device)

        # Load public test set for sensitivity analysis
        self.public_testset = load_public_testset(args.dataset)

        self.clients_acc = [[] for _ in range(args.num_clients)]


    # -----------------------------------------------------------
    #            Weighted Average of Local Models
    # -----------------------------------------------------------
    def avg_weights(self, w):
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = w_avg[key] / len(w)
        return w_avg


    # -----------------------------------------------------------
    #       Collect Samples for Each Class (from Public Dataset)
    # -----------------------------------------------------------
    def collect_samples_by_class(self):
        samples = {}
        for img, label in self.public_testset:
            label = int(label)
            if label not in samples:
                samples[label] = []
            if len(samples[label]) < self.samples_per_class:
                samples[label].append(img.unsqueeze(0).to(self.args.device))

            # early stop if all classes collected enough
            if len(samples) == self.args.num_classes and \
               all(len(v) >= self.samples_per_class for v in samples.values()):
                break

        print("Collected samples for classes:", samples.keys())
        return samples


    # -----------------------------------------------------------
    #         Compute Gradient-based Sensitivity Per Class
    # -----------------------------------------------------------
    def compute_param_sensitivity_by_class(self, model_state_dict, samples_by_class, target_classes):
        device = self.args.device
        model = copy.deepcopy(self.global_model).to(device)
        model.load_state_dict(model_state_dict)
        model.eval()

        class_sens = {}

        for c in target_classes:
            if c not in samples_by_class:
                continue

            # init accumulator
            accum = {name: torch.zeros_like(param) 
                     for name, param in model.named_parameters()}

            count = 0
            for x in samples_by_class[c]:
                model.zero_grad()
                out = model(x)
                logit = out[:, c].sum()
                logit.backward()

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        accum[name] += param.grad.abs().detach()
                count += 1

            if count == 0:
                continue

            for name in accum:
                accum[name] = accum[name] / count

            class_sens[c] = accum
            print(f"Sensitivity for class {c} computed.")

        return class_sens


    # -----------------------------------------------------------
    #                    Parameter Grafting
    # -----------------------------------------------------------
    def adapt_local_with_global(self, local_state, global_state, importance_scores, topk_frac, mix_alpha):
        adapted = copy.deepcopy(local_state)

        # flatten importance scores to determine threshold
        all_scores = torch.cat([v.view(-1) for v in importance_scores.values()])
        k = max(1, int(topk_frac * all_scores.numel()))
        threshold = torch.topk(all_scores, k).values.min().item()

        # graft
        for name in adapted.keys():
            if name not in importance_scores:
                continue

            score = importance_scores[name]
            mask = (score >= threshold)
            if mask.sum() == 0:
                continue

            local_param = adapted[name].to(self.args.device)
            global_param = global_state[name].to(self.args.device)

            mixed = local_param.clone()
            mixed[mask] = (1-mix_alpha)*local_param[mask] + mix_alpha*global_param[mask]
            adapted[name] = mixed.cpu()

        return adapted


    # -----------------------------------------------------------
    #                  Main Federated Training Loop
    # -----------------------------------------------------------
    def train(self):
        for rnd in range(self.args.comm_round):
            print(f"\n===== Round {rnd+1} / {self.args.comm_round} =====")

            # ① 同步最新全局模型
            for client in self.clients:
                client.net.load_state_dict(self.global_model.state_dict())

            # ② 客户端本地训练
            local_weights, client_classes = [], []
            for client in self.clients:
                w, train_classes = client.train(client.net)
                local_weights.append(w)
                client_classes.append(train_classes)
                print(f"Client {client.client_id} train classes: {train_classes}")

            # ③ FedAvg 聚合
            global_weights = self.avg_weights(local_weights)
            self.global_model.load_state_dict(global_weights)
            print("Server aggregated global model.")

            # ④ 使用公共数据集计算敏感度
            samples_by_class = self.collect_samples_by_class()

            known_classes = sorted(samples_by_class.keys())
            client_missing = []
            union_missing = set()

            for classes in client_classes:
                missing = sorted(list(set(known_classes) - set(classes)))
                client_missing.append(missing)
                union_missing.update(missing)

            target_classes = sorted(list(union_missing))
            print("Missing classes:", target_classes)

            class_sens = self.compute_param_sensitivity_by_class(global_weights,
                                                                 samples_by_class,
                                                                 target_classes)

            # ⑤ 对每个客户端做 grafting 个性化适配
            adapted_states = []
            for i, client in enumerate(self.clients):
                missing = client_missing[i]
                if len(missing) == 0:
                    adapted_states.append(local_weights[i])
                    continue

                # aggregate importance for that client's missing classes
                importance = {}
                for c in missing:
                    if c in class_sens:
                        for name, sens in class_sens[c].items():
                            if name not in importance:
                                importance[name] = sens.clone()
                            else:
                                importance[name] += sens.clone()

                # avg over missing classes
                for name in importance:
                    importance[name] /= len(missing)

                adapted = self.adapt_local_with_global(local_weights[i],
                                                       global_weights,
                                                       importance,
                                                       self.args.topk_frac,
                                                       self.args.mix_alpha)
                adapted_states.append(adapted)
                print(f"Client {i} grafted for classes {missing}")

            # ⑥ 测试
            
            for i, client in enumerate(self.clients):
                client.net.load_state_dict(adapted_states[i])
                acc_local = client.test(client.net)
                acc_global = client.test(self.global_model)
                self.clients_acc[i].append(acc_local)
                print(f"Client {i}: personalized={acc_local:.2f} | global={acc_global:.2f}")
                
            final_accs = []
            for cid, client in enumerate(self.clients):
                client.fine_tune(client.net)
                acc = client.test(client.net)
                final_accs.append(acc)
                print(f"Client {cid} Final Acc: {acc:.2f}%")

            print(f"Final Personalized Acc = {np.mean(final_accs):.2f}%")


        #self.save_results()

    def save_results(self):
        fname = f'./results/{self.args.dataset}/{self.args.method}.csv'
        with open(fname, 'w') as f:
            writer = csv.writer(f)
            for acc_list in self.clients_acc:
                writer.writerow(acc_list)
        print("Results saved.")


# ===============================================================
#                       Run System
# ===============================================================
class Args:
    def __init__(self):
        self.comm_round = 1
        self.all_clients = 10
        self.num_clients = 10
        self.local_epochs = 100
        self.lr = 0.001
        self.batch_size = 64
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.dataset = 'Fashion'  ##'Cifar10' 'Fashion'
        # grafting parameters
        self.method = 'ours_Fashion_miss3'
        self.topk_frac = 0.8
        self.mix_alpha = 0.8
        self.ft_epochs = 1

        # dataset class count
        self.num_classes = 10


args = Args()

begin = time.time()

train_loaders, test_loaders, info = create_client_dataloaders_partial_ood1(
    dataname=args.dataset,
    num_clients=args.num_clients,
    num_train_classes=5,
    num_overlap_classes=5,
    num_test_only_classes=5,
    batch_size=args.batch_size
)
print(info)

clients = [Client(train_loaders[i], test_loaders[i], args, i)
           for i in range(args.all_clients)]

fedavg = FedAvg(clients, args)
fedavg.train()

print("Running time:", time.time() - begin)
