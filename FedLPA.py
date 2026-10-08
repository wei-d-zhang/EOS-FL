# -*- coding: utf-8 -*-

import os
import copy
import csv
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from model import *
from data_loader1 import *
from overlap_data import create_client_dataloaders_partial_ood1


torch.manual_seed(643)
np.random.seed(600)


# ============================================================
# Utilities for FedLPA layer-wise posterior approximation
# ============================================================

class LayerFisherCollector(object):

    def __init__(self, model, damping=1e-3):
        self.model = model
        self.damping = damping

        self.layers = []
        self.layer_inputs = {}
        self.layer_grads = {}
        self.handles = []

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.layers.append((name, module))
                self.layer_inputs[name] = None
                self.layer_grads[name] = None

                self.handles.append(
                    module.register_forward_pre_hook(self._make_forward_hook(name))
                )

                # full backward hook gives grad_output of the module
                self.handles.append(
                    module.register_full_backward_hook(self._make_backward_hook(name))
                )

    def _make_forward_hook(self, name):
        def hook(module, inputs):
            # Save input activation without graph.
            self.layer_inputs[name] = inputs[0].detach()
        return hook

    def _make_backward_hook(self, name):
        def hook(module, grad_input, grad_output):
            # grad_output[0] is gradient wrt layer output/pre-activation
            self.layer_grads[name] = grad_output[0].detach()
        return hook

    @staticmethod
    def _conv_input_matrix(x, layer):
        """
        x: [N, C_in, H, W]
        Return activation matrix [C_in*kH*kW (+ bias), num_positions]
        """
        patches = torch.nn.functional.unfold(
            x,
            kernel_size=layer.kernel_size,
            dilation=layer.dilation,
            padding=layer.padding,
            stride=layer.stride
        )
        # [N, Cin*kH*kW, L] -> [Cin*kH*kW, N*L]
        a = patches.permute(1, 0, 2).reshape(patches.size(1), -1)

        if layer.bias is not None:
            ones = torch.ones(
                1, a.size(1), device=a.device, dtype=a.dtype
            )
            a = torch.cat([a, ones], dim=0)

        return a

    @staticmethod
    def _conv_grad_matrix(g):
        """
        g: [N, C_out, H_out, W_out]
        Return [C_out, N*H_out*W_out]
        """
        return g.permute(1, 0, 2, 3).reshape(g.size(1), -1)

    @staticmethod
    def _linear_input_matrix(x, layer):
        """
        x: [N, Din] -> [Din (+bias), N]
        """
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)

        a = x.t()

        if layer.bias is not None:
            ones = torch.ones(
                1, a.size(1), device=a.device, dtype=a.dtype
            )
            a = torch.cat([a, ones], dim=0)

        return a

    @staticmethod
    def _linear_grad_matrix(g):
        """
        g: [N, Dout] -> [Dout, N]
        """
        if g.dim() > 2:
            g = g.reshape(g.size(0), -1)
        return g.t()

    def collect(self, dataloader, criterion, device, max_batches=None):
        self.model.eval()

        A_sum = {}
        B_sum = {}
        count = {}

        for name, _ in self.layers:
            A_sum[name] = None
            B_sum[name] = None
            count[name] = 0

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs = inputs.to(device)
            labels = labels.to(device)

            self.model.zero_grad()

            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            for name, layer in self.layers:
                x = self.layer_inputs[name]
                g = self.layer_grads[name]

                if x is None or g is None:
                    continue

                if isinstance(layer, nn.Linear):
                    a = self._linear_input_matrix(x, layer)
                    b = self._linear_grad_matrix(g)

                elif isinstance(layer, nn.Conv2d):
                    a = self._conv_input_matrix(x, layer)
                    b = self._conv_grad_matrix(g)

                # Average second moments for current batch.
                A_batch = (a @ a.t()) / max(a.size(1), 1)
                B_batch = (b @ b.t()) / max(b.size(1), 1)

                if A_sum[name] is None:
                    A_sum[name] = A_batch.detach().clone()
                    B_sum[name] = B_batch.detach().clone()
                else:
                    A_sum[name] += A_batch.detach()
                    B_sum[name] += B_batch.detach()

                count[name] += 1

        factors = {}

        for name, layer in self.layers:
            if count[name] == 0:
                continue

            A = A_sum[name] / count[name]
            B = B_sum[name] / count[name]

            # pi_l in the paper balances the two Kronecker factors.
            # A practical K-FAC choice:
            # pi = sqrt( (tr(A)/dimA) / (tr(B)/dimB) )
            trace_A = torch.trace(A) / A.size(0)
            trace_B = torch.trace(B) / B.size(0)

            pi = torch.sqrt(
                torch.clamp(trace_A / (trace_B + 1e-12), min=1e-12)
            )

            A = A + (pi * np.sqrt(self.damping)) * torch.eye(
                A.size(0), device=A.device, dtype=A.dtype
            )
            B = B + (np.sqrt(self.damping) / (pi + 1e-12)) * torch.eye(
                B.size(0), device=B.device, dtype=B.dtype
            )

            factors[name] = {
                'A': A.detach().cpu(),
                'B': B.detach().cpu()
            }

        self.model.zero_grad()
        return factors

    def close(self):
        for handle in self.handles:
            handle.remove()


# ============================================================
# Client
# ============================================================

class Client(object):
    def __init__(self, local_trainloader, local_testloader, args, client_id):
        self.trainloader = local_trainloader
        self.testloader = local_testloader
        self.args = args
        self.client_id = client_id

        model_mapping = {
            'Fashion': Fashion_ResNet18,
            'Cifar10': cifar10_ResNet18,
            'Cifar100': cifar100_ResNet18,
            'Gear': Gear_ResNet18,
        }

        self.net = model_mapping.get(
            self.args.dataset, lambda: None
        )().to(self.args.device)

        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=args.lr,
        )

        self.criterion = nn.CrossEntropyLoss()

    def train(self, net, epochs=None):
        net.train()
        num_epochs = epochs if epochs is not None else self.args.local_epochs

        for epoch in range(num_epochs):
            correct = 0
            total = 0
            running_loss = 0.0

            for inputs, labels in self.trainloader:
                inputs = inputs.to(self.args.device)
                labels = labels.to(self.args.device)

                self.optimizer.zero_grad()

                outputs = net(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * labels.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = 100.0 * correct / max(total, 1)

            print(
                f"Client {self.client_id} | "
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Acc: {train_acc:.2f}%"
            )

        return copy.deepcopy(self.net.state_dict())

    def estimate_posterior(self):
        """
        Estimate {M_kl, A_kl, B_kl} after local training.

        M_kl is reconstructed from the trained weight and bias:
            M = [W | b]
        so that vec(M) corresponds to the full trainable layer parameters.
        """
        collector = LayerFisherCollector(
            self.net,
            damping=self.args.fisher_damping
        )

        factors = collector.collect(
            self.trainloader,
            self.criterion,
            self.args.device,
            max_batches=self.args.fisher_batches
        )

        posterior = {}

        named_modules = dict(self.net.named_modules())

        for name, factor in factors.items():
            layer = named_modules[name]

            if isinstance(layer, nn.Linear):
                W = layer.weight.detach().cpu()

                if layer.bias is not None:
                    b = layer.bias.detach().cpu().view(-1, 1)
                    M = torch.cat([W, b], dim=1)
                else:
                    M = W

            elif isinstance(layer, nn.Conv2d):
                W = layer.weight.detach().cpu()
                W = W.reshape(W.size(0), -1)

                if layer.bias is not None:
                    b = layer.bias.detach().cpu().view(-1, 1)
                    M = torch.cat([W, b], dim=1)
                else:
                    M = W

            posterior[name] = {
                'M': M,
                'A': factor['A'],
                'B': factor['B']
            }

        collector.close()
        return posterior

    def test(self, net):
        net.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs = inputs.to(self.args.device)
                labels = labels.to(self.args.device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / max(total, 1)
        return accuracy


# ============================================================
# FedLPA Server
# ============================================================

class FedLPA(object):
    def __init__(self, clients, args):
        self.clients = clients
        self.args = args

        if self.args.dataset == 'Fashion':
            self.global_model = Fashion_ResNet18().to(self.args.device)
        elif self.args.dataset == 'Cifar10':
            self.global_model = cifar10_ResNet18().to(self.args.device)
        elif self.args.dataset == 'Cifar100':
            self.global_model = cifar100_ResNet18().to(self.args.device)
        elif self.args.dataset == 'Gear':
            self.global_model = Gear_ResNet18().to(self.args.device)
        else:
            raise ValueError("Unsupported dataset.")

        self.clients_acc = [[] for _ in range(args.num_clients)]
        self.global_acc = []

    # --------------------------------------------------------
    # Simple averaging for non-FedLPA parameters:
    # BatchNorm buffers, running mean/var, etc.
    # --------------------------------------------------------
    def avg_weights(self, weights):
        w_avg = copy.deepcopy(weights[0])

        for key in w_avg.keys():
            values = []

            for i in range(len(weights)):
                if torch.is_floating_point(weights[i][key]):
                    values.append(weights[i][key].float())
                else:
                    values.append(weights[i][key].float())

            mean_value = torch.stack(values, dim=0).mean(dim=0)

            if torch.is_floating_point(w_avg[key]):
                w_avg[key] = mean_value.to(w_avg[key].dtype)
            else:
                w_avg[key] = mean_value.round().to(w_avg[key].dtype)

        return w_avg

    # --------------------------------------------------------
    # Eq. (14):
    # min_M 1/2 || sum_k B_k M A_k - sum_k B_k M_k A_k ||_F^2
    # --------------------------------------------------------
    def optimize_global_layer(self, client_layer_posteriors, init_M):
        """
        Directly optimize the global layer parameter M using autograd,
        following the quadratic objective in FedLPA Eq. (14).
        """

        device = self.args.device

        M = nn.Parameter(init_M.to(device).clone())

        optimizer = optim.Adam(
            [M],
            lr=self.args.server_lr
        )

        target_Z = torch.zeros_like(M, device=device)

        A_list = []
        B_list = []

        for item in client_layer_posteriors:
            A = item['A'].to(device)
            B = item['B'].to(device)
            Mk = item['M'].to(device)

            A_list.append(A)
            B_list.append(B)

            # z_bar = sum_k B_k M_k A_k
            target_Z = target_Z + B @ Mk @ A

        for _ in range(self.args.server_epochs):
            optimizer.zero_grad()

            prediction = torch.zeros_like(M)

            # sum_k B_k M A_k
            for A, B in zip(A_list, B_list):
                prediction = prediction + B @ M @ A

            # Frobenius norm is equivalent to vectorized L2 norm.
            loss = 0.5 * torch.sum((prediction - target_Z) ** 2)

            loss.backward()
            optimizer.step()

            if loss.item() < self.args.server_tol:
                break

        return M.detach().cpu()

    def write_layer_matrix_to_state(self, state_dict, layer_name, M):
        """
        Convert optimized M back to weight and bias tensors.
        """

        weight_key = layer_name + '.weight'
        bias_key = layer_name + '.bias'

        weight = state_dict[weight_key]
        original_shape = weight.shape

        if weight.dim() == 2:
            # Linear
            if bias_key in state_dict:
                new_weight = M[:, :-1].reshape(original_shape)
                new_bias = M[:, -1].reshape(state_dict[bias_key].shape)

                state_dict[weight_key] = new_weight.to(
                    dtype=state_dict[weight_key].dtype
                )
                state_dict[bias_key] = new_bias.to(
                    dtype=state_dict[bias_key].dtype
                )
            else:
                state_dict[weight_key] = M.reshape(original_shape).to(
                    dtype=state_dict[weight_key].dtype
                )

        elif weight.dim() == 4:
            # Conv2d
            if bias_key in state_dict:
                new_weight = M[:, :-1].reshape(original_shape)
                new_bias = M[:, -1].reshape(state_dict[bias_key].shape)

                state_dict[weight_key] = new_weight.to(
                    dtype=state_dict[weight_key].dtype
                )
                state_dict[bias_key] = new_bias.to(
                    dtype=state_dict[bias_key].dtype
                )
            else:
                state_dict[weight_key] = M.reshape(original_shape).to(
                    dtype=state_dict[weight_key].dtype
                )

        return state_dict

    def layer_wise_posterior_aggregation(
        self,
        local_weights,
        local_posteriors
    ):

        global_state = self.avg_weights(local_weights)

        # Find all layers appearing in client posterior results.
        all_layer_names = set()

        for posterior in local_posteriors:
            all_layer_names.update(posterior.keys())

        for layer_name in sorted(all_layer_names):

            layer_data = []

            for posterior in local_posteriors:
                if layer_name in posterior:
                    layer_data.append(posterior[layer_name])

            if len(layer_data) == 0:
                continue

            # FedLPA initialization: averaged local parameter matrix.
            M0 = torch.stack(
                [item['M'].float() for item in layer_data],
                dim=0
            ).mean(dim=0)

            optimized_M = self.optimize_global_layer(
                layer_data,
                M0
            )

            global_state = self.write_layer_matrix_to_state(
                global_state,
                layer_name,
                optimized_M
            )

        return global_state

    def save_results(self):
        os.makedirs(
            f'./results/{self.args.dataset}',
            exist_ok=True
        )

        summary_file = (
            f'./results/{self.args.dataset}/{self.args.method}.csv'
        )

        with open(
            summary_file,
            'w',
            encoding='utf-8',
            newline=''
        ) as f:

            csv_writer = csv.writer(f)

            # Global model accuracy in the first row.
            csv_writer.writerow(self.global_acc)

            # Each following row corresponds to one client.
            for client_id in range(len(self.clients)):
                csv_writer.writerow(self.clients_acc[client_id])

    def train(self):
        print('\n==============================')
        print(' FedLPA One-Shot Training')
        print('==============================\n')

        # ----------------------------------------------------
        # 1. Broadcast identical initialization once.
        # ----------------------------------------------------
        initial_state = copy.deepcopy(
            self.global_model.state_dict()
        )

        for client in self.clients:
            client.net.load_state_dict(initial_state)

        # ----------------------------------------------------
        # 2. Local training.
        # ----------------------------------------------------
        local_weights = []
        local_posteriors = []

        for client in self.clients:
            print(
                f'\n--- Client {client.client_id}: '
                f'Local Training ---'
            )

            local_w = client.train(client.net)
            local_weights.append(local_w)

            print(
                f'--- Client {client.client_id}: '
                f'Posterior Estimation ---'
            )

            posterior = client.estimate_posterior()
            local_posteriors.append(posterior)

        # ----------------------------------------------------
        # 3. One-shot layer-wise posterior aggregation.
        # ----------------------------------------------------
        print('\n--- Server: Layer-wise Posterior Aggregation ---')

        global_weights = self.layer_wise_posterior_aggregation(
            local_weights,
            local_posteriors
        )

        self.global_model.load_state_dict(global_weights)

        # ----------------------------------------------------
        # 4. Evaluation.
        # ----------------------------------------------------
        print('\n--- Evaluation: Updating Local Models, Fine-tuning & Testing ---')
        
        # 统计纯全局模型的平均精度（更新前的 global_model 在各本地测试集上的表现）
        global_accuracies_before_ft = []
        for client in self.clients:
            g_acc = client.test(self.global_model)
            global_accuracies_before_ft.append(g_acc)
        mean_global_acc = np.mean(global_accuracies_before_ft)
        self.global_acc.append(mean_global_acc)

        local_accuracies_after_ft = []

        for client_id, client in enumerate(self.clients):
            print(f'\n--- Client {client_id}: Updating global model to local & Fine-tuning ---')
            
            # Step 4.1: 先把 global_model 的权重同步给 local model
            client.net.load_state_dict(copy.deepcopy(global_weights))
            
            # Step 4.2: 在本地训练集上微调一轮
            client.train(client.net, epochs=self.args.finetune_epochs)
            
            # Step 4.3: 测试微调后的本地模型精度
            finetuned_local_acc = client.test(client.net)
            self.clients_acc[client_id].append(finetuned_local_acc)
            local_accuracies_after_ft.append(finetuned_local_acc)

        mean_local_acc = np.mean(local_accuracies_after_ft)

        print('\n==============================')
        print(f'FedLPA Global Model Accuracy (Before FT): {mean_global_acc:.4f}%')
        print(f'Finetuned Local Model Mean Accuracy:     {mean_local_acc:.4f}%')
        print('==============================\n')

        self.save_results()


# ============================================================
# Arguments
# ============================================================

class Args:
    def __init__(self):

        # ----------------------------------------------------
        # One-shot FL
        # ----------------------------------------------------
        self.comm_round = 1

        self.all_clients = 10
        self.num_clients = 10

        # ----------------------------------------------------
        # Local training & Fine-tuning
        # ----------------------------------------------------
        self.local_epochs = 100
        self.finetune_epochs = 1  # 评估阶段微调的轮数
        self.lr = 0.001

        self.batch_size = 64

        self.device = (
            'cuda:1'
            if torch.cuda.is_available()
            else 'cpu'
        )

        # Fashion / Cifar10 / Cifar100 / Gear
        self.dataset = 'Gear'

        # ----------------------------------------------------
        # FedLPA hyperparameters
        # ----------------------------------------------------

        # Fisher/Laplace damping lambda.
        self.fisher_damping = 1e-4

        # Use only part of local data to estimate Fisher.
        # None means use all local batches.
        self.fisher_batches = 20

        # Server optimization of Eq. (14).
        self.server_lr = 1e-2
        self.server_epochs = 1
        self.server_tol = 1e-8

        self.method = 'FedLPA_Gear_0.5'


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':

    args = Args()

    begin_time = time.time()

    train_loaders, test_loaders, info = (
        create_client_dataloaders_partial_ood1(
            dataname=args.dataset,
            num_clients=args.num_clients,
            num_train_classes=4,
            num_overlap_classes=4,
            num_test_only_classes=5,
            # alpha=args.dirichlet_alpha,
            batch_size=args.batch_size
        )
    )

    print(info)

    clients = [
        Client(
            train_loaders[i],
            test_loaders[i],
            args,
            i
        )
        for i in range(args.all_clients)
    ]

    fedlpa = FedLPA(clients, args)
    fedlpa.train()

    end_time = time.time()

    run_time = end_time - begin_time

    print(f'Running time: {run_time:.2f} seconds')