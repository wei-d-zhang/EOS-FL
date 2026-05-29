# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import csv
from torchvision import datasets, transforms,models
from torch.utils.data import Subset
from model import *
from data_loader1 import *
import torch.nn.functional as F
import time
from overlap_data import create_client_dataloaders_partial_ood1

torch.manual_seed(643)
np.random.seed(600)

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
        self.net = model_mapping.get(self.args.dataset, lambda: None)().to(self.args.device)
        
        #self.net = models.resnet18(pretrained=False, num_classes=100).to(self.args.device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, net):
        net.train()
        for epoch in range(self.args.local_epochs):
            #print(f'本地训练：{epoch}')
            correct = 0
            total = 0
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
            train_acc = 100 * correct / total
            print(f"Client {self.client_id} | Epoch {epoch+1}/{self.args.local_epochs} | "
                  f"Train Acc: {train_acc:.2f}%")
                
        return self.net.state_dict()
    

    def test(self, net):
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        #print(f'Client Test Accuracy: {accuracy}%')
        return accuracy

class FedAvg(object):
    def __init__(self, clients, args):
        self.clients = clients
        self.args = args
        
        if self.args.dataset == 'Fashion':
            self.global_model = Fashion_ResNet18().to(self.args.device)
        elif self.args.dataset == 'Cifar10':
            self.global_model = cifar10_ResNet18().to(self.args.device)
        elif self.args.dataset == 'Cifar100':
            self.global_model = cifar100_ResNet18().to(self.args.device)
        else:
            print("coming soon")
        
        #self.global_model = models.resnet18(pretrained=False, num_classes=100).to(self.args.device)
        self.clients_acc = [[] for i in range(args.num_clients)]
        
    def avg_weights(self,w):
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = w_avg[key].float() / len(w)
        return w_avg
    
    def save_results(self):
        summary_file = f'./results/{self.args.dataset}/{self.args.method}.csv'
        with open(summary_file, 'w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            for client_id in range(len(self.clients)):
                all_accs = self.clients_acc[client_id]
                csv_writer.writerow(all_accs)

    def train(self):
        for comm_round in range(self.args.comm_round):
            print(f'\n--- Communication Round {comm_round+1}/{self.args.comm_round} ---')

            for client in self.clients:
                client.net.load_state_dict(self.global_model.state_dict())

            local_weights = []
            for client in self.clients:
                local_w = client.train(client.net)
                local_weights.append(local_w)

            global_weights = self.avg_weights(local_weights)
            self.global_model.load_state_dict(global_weights)

            accuracies = []
            global_accuracies = []
            for client_id, client in enumerate(self.clients):
                acc = self.clients[client_id].test(client.net)
                acc2 =  self.clients[client_id].test(self.global_model)
                self.clients_acc[client_id].append(acc)
                accuracies.append(acc)
                global_accuracies.append(acc2)
            print(f'Global Model Test Accuracy: {np.mean(global_accuracies)}%')
            print(f'Local Model Test Accuracy: {np.mean(accuracies)}%')
            
        self.save_results()

class Args:
    def __init__(self):
        self.comm_round = 1  # Communication rounds
        self.all_clients = 10  # Total number of clients
        self.num_clients = 10  # The number of clients selected per round
        self.local_epochs = 100  # Number of client local training rounds
        self.lr = 0.001  # Learning rate
        self.batch_size = 64  # Batch size
        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        self.dataset = 'Cifar100' ####Fashion/Cifar10/Cifar100
        self.non_iid = 'Dirichlet' ##Dirichlet/Pathological/iid
        self.dirichlet_alpha = 0.5 #dirichlet coefficient /non-IID degree
        self.num_shard = 50 #The number of categories into which the data set is divided
        self.method = 'FedAvg_Cifar100_5'  #FedAvg/Ditto/Our/FedALA/Local

args = Args()
begin_time = time.time()

train_loaders, test_loaders, info = create_client_dataloaders_partial_ood1(
    dataname=args.dataset,
    num_clients=args.num_clients,
    num_train_classes=50,      # 训练集类别数量
    num_overlap_classes=50,    # 训练测试共有类别
    num_test_only_classes=50,  # 测试独有类别数量
    #alpha=args.dirichlet_alpha,                # Dirichlet 控制异质性
    batch_size=args.batch_size
)
print(info)

clients = [Client(train_loaders[i], test_loaders[i], args, i) for i in range(args.all_clients)]
fedavg = FedAvg(clients, args)
fedavg.train()
end_time = time.time()
run_time = end_time - begin_time
print(f"Running time: {run_time} seconds")
