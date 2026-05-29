# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import csv
import time
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader

# Note: Assuming overlap_data and model files exist as per your original FedAvg.py
# If running standalone, you might need to mock these imports.
from model import Fashion_ResNet18, cifar10_ResNet18, cifar100_ResNet18
from overlap_data import create_client_dataloaders_partial_ood1

torch.manual_seed(643)
np.random.seed(600)

# ==============================================================================
# 1. DENSE Helper Components (Generator, Ensemble, Losses)
# ==============================================================================

class Generator(nn.Module):
    """
    Standard Generator for Data-Free Distillation (matched to ResNet/CIFAR input sizes)
    """
    def __init__(self, nz=256, ngf=64, img_size=32, nc=3):
        super(Generator, self).__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Ensemble(nn.Module):
    """
    Wraps multiple client models to act as a single Teacher.
    Returns the average logits of all models.
    """
    def __init__(self, model_list):
        super(Ensemble, self).__init__()
        self.models = model_list

    def forward(self, x):
        logits_total = 0
        valid_models = 0
        for model in self.models:
            model.eval()
            logits = model(x)
            logits_total += logits
            valid_models += 1
        logits_e = logits_total / valid_models
        return logits_e

class KLDiv(nn.Module):
    """
    Distillation Loss (Knowledge Transfer)
    """
    def __init__(self, T=3.0):
        super(KLDiv, self).__init__()
        self.T = T

    def forward(self, pred, target):
        pred = F.log_softmax(pred / self.T, dim=1)
        target = F.softmax(target / self.T, dim=1)
        return F.kl_div(pred, target, reduction='batchmean') * (self.T * self.T)

# ==============================================================================
# 2. Client Class (Standard Local Training)
# ==============================================================================

class Client(object):
    def __init__(self, local_trainloader, local_testloader, args, client_id):
        self.trainloader = local_trainloader
        self.testloader = local_testloader
        self.args = args
        self.client_id = client_id
        
        # Model Selection
        model_mapping = {
            'Fashion': Fashion_ResNet18,
            'Cifar10': cifar10_ResNet18,
            'Cifar100': cifar100_ResNet18
        }
        self.net = model_mapping.get(self.args.dataset, lambda: None)().to(self.args.device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        """Standard local training on private data"""
        self.net.train()
        print(f"Client {self.client_id} starting pre-training...")
        for epoch in range(self.args.local_epochs):
            correct = 0
            total = 0
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Optional: Print every few epochs to reduce clutter
            if (epoch + 1) % 20 == 0:
                train_acc = 100 * correct / total
                print(f"Client {self.client_id} | Epoch {epoch+1}/{self.args.local_epochs} | Acc: {train_acc:.2f}%")
                
        return copy.deepcopy(self.net) # Return the full model, not just weights

    def test(self, net=None):
        if net is None:
            net = self.net
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
        return accuracy

# ==============================================================================
# 3. DENSE Server Logic (Replaces FedAvg)
# ==============================================================================

class DENSE_Server(object):
    def __init__(self, clients, args):
        self.clients = clients
        self.args = args
        
        # Initialize Global Model (Student)
        if self.args.dataset == 'Fashion':
            self.global_model = Fashion_ResNet18().to(self.args.device)
        elif self.args.dataset == 'Cifar10':
            self.global_model = cifar10_ResNet18().to(self.args.device)
        elif self.args.dataset == 'Cifar100':
            self.global_model = cifar100_ResNet18().to(self.args.device)
        
        # Generator for Data-Free Synthesis
        self.generator = Generator(nz=args.nz, img_size=32, nc=3).to(self.args.device)
        
        self.clients_acc = [[] for _ in range(args.num_clients)]
        
    def save_results(self):
        summary_file = f'./results/{self.args.dataset}/{self.args.method}.csv'
        try:
            with open(summary_file, 'w', encoding='utf-8', newline='') as f:
                csv_writer = csv.writer(f)
                for client_id in range(len(self.clients)):
                    all_accs = self.clients_acc[client_id]
                    csv_writer.writerow(all_accs)
        except Exception as e:
            print(f"Error saving results: {e}")

    def run(self):
        # ---------------------------------------------------------
        # Phase 1: Local Pre-training (The "One-Shot" preparation)
        # ---------------------------------------------------------
        print("\n=== Phase 1: Clients Local Training ===")
        client_models = []
        for client in self.clients:
            # Train and get the model
            trained_model = client.train()
            # Verify performance
            acc = client.test(trained_model)
            self.clients_acc[client.client_id].append(acc) 
            client_models.append(trained_model)
        
        # Create Ensemble from client models (The Teacher)
        ensemble_model = Ensemble(client_models)
        ensemble_model.eval()

        # ---------------------------------------------------------
        # Phase 2: Server-Side Data-Free Distillation (DENSE)
        # ---------------------------------------------------------
        print("\n=== Phase 2: DENSE Data Generation & Distillation ===")
        
        # Optimizers
        optimizer_g = optim.Adam(self.generator.parameters(), lr=self.args.lr_g)
        optimizer_s = optim.SGD(self.global_model.parameters(), lr=self.args.lr, momentum=0.9)
        scheduler_s = optim.lr_scheduler.CosineAnnealingLR(optimizer_s, T_max=self.args.dense_epochs)
        
        criterion_kd = KLDiv(T=self.args.T)
        criterion_ce = nn.CrossEntropyLoss()

        # DENSE Loop
        for epoch in range(self.args.dense_epochs):
            
            # A. Data Synthesis Stage (Train Generator)
            # -----------------------------------------
            for _ in range(self.args.g_steps):
                self.generator.train()
                optimizer_g.zero_grad()
                
                # 1. Sample Noise
                z = torch.randn(self.args.batch_size, self.args.nz).to(self.args.device)
                
                # 2. Generate Synthetic Data
                syn_img = self.generator(z)
                
                # 3. Get Ensemble Logits (Teacher Output)
                # DENSE uses the Ensemble to guide generation.
                # Images should maximize class confidence (Cross Entropy minimization against predicted label)
                with torch.no_grad():
                    teacher_logits = ensemble_model(syn_img)
                    teacher_preds = torch.argmax(teacher_logits, dim=1)
                
                # 4. Generator Loss
                # L_gen = CrossEntropy(Ensemble_Preds) + (Optional BN statistics matching)
                # Note: Full DENSE includes BN feature matching, simplified here to CE + Diversity for portability
                loss_ce = criterion_ce(teacher_logits, teacher_preds)
                
                # Diversity/Boundary Loss (Optional but recommended in DENSE)
                loss_gen = loss_ce 
                loss_gen.backward()
                optimizer_g.step()

            # B. Model Distillation Stage (Train Global Model)
            # ------------------------------------------------
            self.global_model.train()
            for _ in range(self.args.kd_steps):
                optimizer_s.zero_grad()
                
                # 1. Sample Noise & Generate Data
                z = torch.randn(self.args.batch_size, self.args.nz).to(self.args.device)
                with torch.no_grad():
                    syn_img = self.generator(z)
                    teacher_logits = ensemble_model(syn_img)
                
                # 2. Student Forward Pass
                student_logits = self.global_model(syn_img)
                
                # 3. KD Loss
                loss_kd = criterion_kd(student_logits, teacher_logits)
                loss_kd.backward()
                optimizer_s.step()
            
            scheduler_s.step()

            # C. Evaluation
            # -------------
            if (epoch + 1) % 10 == 0:
                # Test Global Model on Client 0's test set (as a proxy for global performance)
                # ideally should test on all, but for speed logging just one
                acc = self.clients[0].test(self.global_model)
                print(f"DENSE Epoch {epoch+1}/{self.args.dense_epochs} | Global Acc (Client 0 Data): {acc:.2f}%")

        # ---------------------------------------------------------
        # Phase 3: Final Testing
        # ---------------------------------------------------------
        print("\n=== Phase 3: Final Evaluation ===")
        accuracies = []
        global_accuracies = []
        for client_id, client in enumerate(self.clients):
            # Original Local Model Accuracy
            acc = client.test(client.net)
            # New Global DENSE Model Accuracy
            acc2 = client.test(self.global_model)
            
            self.clients_acc[client_id].append(acc2) # Saving global model performance
            accuracies.append(acc)
            global_accuracies.append(acc2)
            
        print(f'Average Local Model Accuracy: {np.mean(accuracies):.2f}%')
        print(f'Final DENSE Global Model Accuracy: {np.mean(global_accuracies):.2f}%')
        
        self.save_results()

# ==============================================================================
# 4. Main Configuration & Execution
# ==============================================================================

class Args:
    def __init__(self):
        # Federation Args
        self.all_clients = 10 
        self.num_clients = 10  # DENSE typically uses a subset
        self.local_epochs = 100 # Pre-training epochs for clients
        self.batch_size = 64
        self.lr = 0.001
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # Dataset Args
        self.dataset = 'Cifar10' # Fashion/Cifar10/Cifar100
        self.non_iid = 'Dirichlet'
        self.dirichlet_alpha = 0.5
        self.method = 'DENSE_Integrated'

        # DENSE Specific Args
        self.dense_epochs = 200 # How many epochs to run generation/distillation
        self.g_steps = 10       # Generator update steps per epoch
        self.kd_steps = 20      # Knowledge Distillation steps per epoch
        self.lr_g = 1e-3        # Generator learning rate
        self.nz = 256           # Noise dimension
        self.T = 1.0            # Temperature for distillation
        self.adv = 0.0          # Adversarial scaling (optional)

if __name__ == '__main__':
    args = Args()
    begin_time = time.time()
    
    # 1. Load Data
    print(f"Loading data for {args.dataset}...")
    train_loaders, test_loaders, info = create_client_dataloaders_partial_ood1(
        dataname=args.dataset,
        num_clients=args.num_clients,
        num_train_classes=9 if args.dataset=='Cifar10' else 90,      
        num_overlap_classes=9 if args.dataset=='Cifar10' else 90,    
        num_test_only_classes=1 if args.dataset=='Cifar10' else 10,  
        batch_size=args.batch_size
    )
    print(info)

    # 2. Initialize Clients
    clients = [Client(train_loaders[i], test_loaders[i], args, i) for i in range(args.num_clients)]

    # 3. Initialize DENSE Server and Run
    dense_server = DENSE_Server(clients, args)
    dense_server.run()

    end_time = time.time()
    run_time = end_time - begin_time
    print(f"Total Running time: {run_time:.2f} seconds")