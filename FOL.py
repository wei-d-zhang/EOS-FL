# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import csv
import random
import time

from model import *
from data_loader1 import *
from overlap_data import create_client_dataloaders_partial_ood1


# ============================================================
# Random seed
# ============================================================
torch.manual_seed(643)
np.random.seed(600)
random.seed(643)


# ============================================================
# Utility functions
# ============================================================
def build_model(args):
    model_mapping = {
        'Fashion': Fashion_ResNet18,
        'Cifar10': cifar10_ResNet18,
        'Cifar100': cifar100_ResNet18,
        'Gear': Gear_ResNet18
    }
    return model_mapping.get(args.dataset, lambda: None)().to(args.device)


def model_cosine_similarity(model_a, model_b):
    """
    Eq.(8): cosine similarity between model parameters.
    Only parameters with identical shapes are compared.
    """
    vec_a, vec_b = [], []

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    for key in state_a.keys():
        if key in state_b and state_a[key].shape == state_b[key].shape:
            if torch.is_floating_point(state_a[key]):
                vec_a.append(state_a[key].detach().float().flatten())
                vec_b.append(state_b[key].detach().float().flatten())

    if len(vec_a) == 0:
        return 0.0

    vec_a = torch.cat(vec_a)
    vec_b = torch.cat(vec_b)

    return F.cosine_similarity(vec_a.unsqueeze(0),
                               vec_b.unsqueeze(0)).item()


# ============================================================
# Client
# ============================================================
class Client(object):

    def __init__(self, local_trainloader, local_testloader,
                 args, client_id):

        self.trainloader = local_trainloader
        self.testloader = local_testloader
        self.args = args
        self.client_id = client_id

        self.net = build_model(args)

        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )

        self.criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Basic supervised training
    # --------------------------------------------------------
    def train_model(self, net, epochs=None, lr=None, verbose=True):

        if epochs is None:
            epochs = self.args.local_epochs

        if lr is None:
            lr = self.args.lr

        net.train()

        optimizer = optim.SGD(
            net.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=self.args.weight_decay
        )

        for epoch in range(epochs):

            correct = 0
            total = 0

            for inputs, labels in self.trainloader:

                inputs = inputs.to(self.args.device)
                labels = labels.to(self.args.device)

                optimizer.zero_grad()

                outputs = net(inputs)

                loss = self.criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)

                correct += (predicted == labels).sum().item()

            train_acc = 100 * correct / max(total, 1)

            if verbose:
                print(
                    f"Client {self.client_id} | "
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train Acc: {train_acc:.2f}%"
                )

        return net

    # --------------------------------------------------------
    # Original interface for compatibility with FedAvg.py
    # --------------------------------------------------------
    def train(self, net):
        net = self.train_model(net)
        return net.state_dict()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
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

        accuracy = 100 * correct / max(total, 1)

        return accuracy

    # --------------------------------------------------------
    # Fine-tuning: Eq.(4)
    # --------------------------------------------------------
    def fine_tune(self, neighbor_model):

        adapted_model = copy.deepcopy(neighbor_model).to(
            self.args.device
        )

        adapted_model = self.train_model(
            adapted_model,
            epochs=self.args.finetune_epochs,
            lr=self.args.finetune_lr,
            verbose=False
        )

        return adapted_model

    # --------------------------------------------------------
    # Alignment-aware structured pruning
    #
    # Practical implementation of Eq.(5):
    # importance =
    #   magnitude importance
    # + alignment importance relative to local model
    #
    # For equal-shaped parameters, parameters highly aligned
    # with the local model receive higher retention scores.
    # --------------------------------------------------------
    def alignment_prune(self, neighbor_model, local_model):

        pruned_model = copy.deepcopy(neighbor_model)

        neighbor_state = pruned_model.state_dict()
        local_state = local_model.state_dict()

        prune_ratio = self.args.prune_ratio
        lambda_align = self.args.lambda_align

        for key in neighbor_state.keys():

            weight = neighbor_state[key]

            if not torch.is_floating_point(weight):
                continue

            # Skip BatchNorm running statistics
            if "running_" in key:
                continue

            # Only prune tensors that contain enough parameters
            if weight.numel() < 100:
                continue

            importance = weight.abs()

            # Alignment regularization for shared layers
            if (key in local_state and
                    local_state[key].shape == weight.shape):

                local_weight = local_state[key].to(weight.device)

                alignment = 1.0 / (
                    (weight - local_weight).abs() + 1e-6
                )

                # Normalize alignment
                alignment = alignment / (
                    alignment.mean() + 1e-8
                )

                importance = (
                    importance
                    + lambda_align * alignment * weight.abs().mean()
                )

            # Magnitude/alignment threshold
            threshold = torch.quantile(
                importance.flatten(),
                prune_ratio
            )

            mask = (importance >= threshold).to(weight.dtype)

            neighbor_state[key] = weight * mask

        pruned_model.load_state_dict(neighbor_state)

        return pruned_model.to(self.args.device)

    # --------------------------------------------------------
    # Post-pruning fine-tuning: Eq.(6)
    # --------------------------------------------------------
    def post_fine_tune(self, pruned_model):

        repaired_model = self.train_model(
            pruned_model,
            epochs=self.args.post_finetune_epochs,
            lr=self.args.finetune_lr,
            verbose=False
        )

        return repaired_model

    # --------------------------------------------------------
    # Full neighbor model alignment pipeline
    # --------------------------------------------------------
    def align_neighbor_model(self,
                             neighbor_model,
                             local_model):

        # Stage 1: Fine-tuning
        model = self.fine_tune(neighbor_model)

        # Stage 2: Alignment-aware pruning
        model = self.alignment_prune(
            model,
            local_model
        )

        # Stage 3: Post-pruning fine-tuning
        model = self.post_fine_tune(model)

        return model

    # --------------------------------------------------------
    # Eq.(11): optimize ensemble weights
    #
    # The model architectures may be heterogeneous because
    # the ensemble is performed at the logit level.
    # --------------------------------------------------------
    def optimize_ensemble_weights(self, selected_models):

        num_models = len(selected_models)

        if num_models == 1:
            return torch.ones(
                1,
                device=self.args.device
            )

        # Learnable logits -> softmax weights
        ensemble_logits = nn.Parameter(
            torch.ones(
                num_models,
                device=self.args.device
            ) / num_models
        )

        optimizer = optim.Adam(
            [ensemble_logits],
            lr=self.args.ensemble_lr
        )

        for model in selected_models:
            model.eval()

        # Optimize Eq.(11)
        for epoch in range(self.args.ensemble_epochs):

            for inputs, labels in self.trainloader:

                inputs = inputs.to(self.args.device)
                labels = labels.to(self.args.device)

                optimizer.zero_grad()

                weights = torch.softmax(
                    ensemble_logits,
                    dim=0
                )

                ensemble_output = None

                for i, model in enumerate(selected_models):

                    with torch.no_grad():
                        output = model(inputs)

                    if ensemble_output is None:
                        ensemble_output = weights[i] * output
                    else:
                        ensemble_output += weights[i] * output

                loss = self.criterion(
                    ensemble_output,
                    labels
                )

                loss.backward()

                optimizer.step()

        final_weights = torch.softmax(
            ensemble_logits.detach(),
            dim=0
        )

        return final_weights

    # --------------------------------------------------------
    # Weighted ensemble logits: Eq.(10)
    # --------------------------------------------------------
    def ensemble_logits(self,
                        selected_models,
                        ensemble_weights,
                        inputs):

        teacher_logits = None

        for i, model in enumerate(selected_models):

            model.eval()

            with torch.no_grad():
                output = model(inputs)

            if teacher_logits is None:
                teacher_logits = ensemble_weights[i] * output
            else:
                teacher_logits += ensemble_weights[i] * output

        return teacher_logits

    # --------------------------------------------------------
    # Regularized knowledge distillation: Eq.(12)
    #
    # KL(teacher || student)
    # + lambda * ||theta_new - theta_old||^2
    # --------------------------------------------------------
    def knowledge_distillation(self,
                               local_model,
                               selected_models,
                               ensemble_weights):

        student_model = copy.deepcopy(local_model).to(
            self.args.device
        )

        old_state = {
            key: value.detach().clone()
            for key, value
            in local_model.state_dict().items()
        }

        optimizer = optim.SGD(
            student_model.parameters(),
            lr=self.args.distill_lr,
            momentum=0.9,
            weight_decay=self.args.weight_decay
        )

        temperature = self.args.temperature

        for epoch in range(self.args.distill_epochs):

            student_model.train()

            total_loss = 0.0

            for inputs, labels in self.trainloader:

                inputs = inputs.to(self.args.device)
                labels = labels.to(self.args.device)

                optimizer.zero_grad()

                # Teacher ensemble
                teacher_output = self.ensemble_logits(
                    selected_models,
                    ensemble_weights,
                    inputs
                )

                # Student output
                student_output = student_model(inputs)

                # KL distillation loss
                kd_loss = F.kl_div(
                    F.log_softmax(
                        student_output / temperature,
                        dim=1
                    ),
                    F.softmax(
                        teacher_output / temperature,
                        dim=1
                    ),
                    reduction='batchmean'
                ) * (temperature ** 2)

                # Optional supervised loss improves stability
                ce_loss = self.criterion(
                    student_output,
                    labels
                )

                # Parameter regularization
                reg_loss = torch.tensor(
                    0.0,
                    device=self.args.device
                )

                current_state = student_model.state_dict()

                for key in current_state.keys():

                    if (key in old_state and
                            torch.is_floating_point(
                                current_state[key]
                            )):

                        reg_loss += torch.sum(
                            (
                                current_state[key]
                                - old_state[key]
                            ) ** 2
                        )

                loss = (
                    kd_loss
                    + self.args.distill_ce_weight * ce_loss
                    + self.args.lambda_kd * reg_loss
                )

                loss.backward()

                optimizer.step()

                total_loss += loss.item()

            print(
                f"Client {self.client_id} | "
                f"KD Epoch {epoch + 1}/{self.args.distill_epochs} | "
                f"Loss: {total_loss / max(len(self.trainloader), 1):.4f}"
            )

        return student_model


# ============================================================
# FOL Controller
# ============================================================
class FOL(object):

    def __init__(self, clients, args):

        self.clients = clients
        self.args = args

        self.clients_acc = [
            []
            for _ in range(args.num_clients)
        ]

        # Personalized models
        self.personal_models = [
            copy.deepcopy(client.net)
            for client in clients
        ]

    # --------------------------------------------------------
    # Save results in the same format as FedAvg.py
    # --------------------------------------------------------
    def save_results(self):

        summary_file = (
            f'./results/{self.args.dataset}/'
            f'{self.args.method}.csv'
        )

        with open(
            summary_file,
            'w',
            encoding='utf-8',
            newline=''
        ) as f:

            csv_writer = csv.writer(f)

            for client_id in range(len(self.clients)):

                csv_writer.writerow(
                    self.clients_acc[client_id]
                )

    # --------------------------------------------------------
    # Stage 1: local model pre-training Eq.(3)
    # --------------------------------------------------------
    def pretrain(self):

        print("\n================================================")
        print("Stage 1: Local Model Pre-training")
        print("================================================")

        for client_id, client in enumerate(self.clients):

            print(
                f"\nClient {client_id}: "
                f"pre-training..."
            )

            model = copy.deepcopy(
                self.personal_models[client_id]
            )

            model = client.train_model(
                model,
                epochs=self.args.pretrain_epochs,
                lr=self.args.lr
            )

            self.personal_models[client_id] = model

    # --------------------------------------------------------
    # Collect Q neighbor models
    #
    # In the original paper, neighbors are determined by
    # communication opportunities. Here we use random sampling
    # to simulate one-shot peer encounters.
    # --------------------------------------------------------
    def collect_neighbors(self, client_id):

        candidate_ids = [
            i
            for i in range(len(self.clients))
            if i != client_id
        ]

        q = min(
            self.args.collection_size,
            len(candidate_ids)
        )

        neighbor_ids = random.sample(
            candidate_ids,
            q
        )

        neighbor_models = [
            copy.deepcopy(self.personal_models[i])
            for i in neighbor_ids
        ]

        return neighbor_ids, neighbor_models

    # --------------------------------------------------------
    # Algorithm 2: Top-K selection
    # validation score + cosine similarity tie-breaker
    # --------------------------------------------------------
    def top_k_selection(self,
                        client,
                        local_model,
                        adapted_models):

        candidates = []

        # Neighbor candidates
        for model in adapted_models:

            score = client.test(model)

            similarity = model_cosine_similarity(
                local_model,
                model
            )

            candidates.append(
                {
                    'model': model,
                    'score': score,
                    'similarity': similarity
                }
            )

        # Current local model
        local_score = client.test(local_model)

        candidates.append(
            {
                'model': local_model,
                'score': local_score,
                'similarity': 1.0
            }
        )

        # Validation score descending,
        # cosine similarity as tie-breaker
        candidates.sort(
            key=lambda x: (
                x['score'],
                x['similarity']
            ),
            reverse=True
        )

        selected_num = min(
            self.args.top_k,
            len(candidates)
        )

        selected_models = [
            candidates[i]['model']
            for i in range(selected_num)
        ]

        print(
            f"Top-{selected_num} "
            f"scores: "
            f"{[round(candidates[i]['score'], 2) for i in range(selected_num)]}"
        )

        return selected_models

    # --------------------------------------------------------
    # Main FOL training procedure
    # --------------------------------------------------------
    def train(self):

        # ================================================
        # Stage 1
        # ================================================
        self.pretrain()

        # ================================================
        # Model collection rounds
        # ================================================
        for collection_round in range(
                self.args.collection_rounds):

            print(
                f"\n"
                f"================================================\n"
                f"Model Collection Round "
                f"{collection_round + 1}/"
                f"{self.args.collection_rounds}\n"
                f"================================================"
            )

            # Store updates after all clients finish.
            # This avoids order-dependent updates.
            updated_models = []

            for client_id, client in enumerate(self.clients):

                print(
                    f"\n-------------------------------\n"
                    f"Client {client_id}\n"
                    f"-------------------------------"
                )

                local_model = copy.deepcopy(
                    self.personal_models[client_id]
                )

                # ----------------------------------------
                # Step 1: collect neighbor models
                # ----------------------------------------
                neighbor_ids, neighbor_models = (
                    self.collect_neighbors(client_id)
                )

                print(
                    f"Collected neighbors: {neighbor_ids}"
                )

                # ----------------------------------------
                # Step 2: alignment
                # Fine-tune -> prune -> post fine-tune
                # ----------------------------------------
                adapted_models = []

                for idx, neighbor_model in enumerate(
                        neighbor_models):

                    print(
                        f"Aligning neighbor "
                        f"{neighbor_ids[idx]} ..."
                    )

                    adapted_model = (
                        client.align_neighbor_model(
                            neighbor_model,
                            local_model
                        )
                    )

                    adapted_models.append(
                        adapted_model
                    )

                # ----------------------------------------
                # Step 3: Top-K selection
                # ----------------------------------------
                selected_models = self.top_k_selection(
                    client,
                    local_model,
                    adapted_models
                )

                # ----------------------------------------
                # Step 4: Optimize ensemble weights
                # ----------------------------------------
                ensemble_weights = (
                    client.optimize_ensemble_weights(
                        selected_models
                    )
                )

                print(
                    "Ensemble weights:",
                    ensemble_weights.detach()
                    .cpu()
                    .numpy()
                    .round(4)
                )

                # ----------------------------------------
                # Step 5: Knowledge distillation
                # ----------------------------------------
                updated_model = (
                    client.knowledge_distillation(
                        local_model,
                        selected_models,
                        ensemble_weights
                    )
                )

                updated_models.append(
                    updated_model
                )

            # --------------------------------------------
            # Synchronous update of personalized models
            # --------------------------------------------
            self.personal_models = updated_models

            # --------------------------------------------
            # Evaluation
            # --------------------------------------------
            accuracies = []

            for client_id, client in enumerate(
                    self.clients):

                acc = client.test(
                    self.personal_models[client_id]
                )

                self.clients_acc[client_id].append(
                    acc
                )

                accuracies.append(acc)

            print(
                f"\nRound "
                f"{collection_round + 1} "
                f"| Mean Personalized Accuracy: "
                f"{np.mean(accuracies):.2f}%"
            )

        self.save_results()


# ============================================================
# Args
# ============================================================
class Args:

    def __init__(self):

        # ----------------------------------------------------
        # General settings
        # ----------------------------------------------------
        self.all_clients = 10
        self.num_clients = 10

        self.batch_size = 64

        self.device = (
            'cuda:0'
            if torch.cuda.is_available()
            else 'cpu'
        )

        self.dataset = 'Gear'
        # Fashion / Cifar10 / Cifar100  / Gear

        self.method = 'FOL_Gear_0.5'

        # ----------------------------------------------------
        # FOL hyperparameters
        # ----------------------------------------------------

        # Eq.(3): local pre-training
        self.pretrain_epochs = 20

        # Model collection rounds E
        self.collection_rounds = 1

        # Maximum collected neighbor models Q
        self.collection_size = 5

        # Top-K selected models
        self.top_k = 3

        # Fine-tuning Eq.(4)
        self.finetune_epochs = 1
        self.finetune_lr = 0.001

        # Alignment-aware pruning Eq.(5)
        self.prune_ratio = 0.20
        self.lambda_align = 0.10

        # Post-pruning fine-tuning Eq.(6)
        self.post_finetune_epochs = 1

        # Ensemble optimization Eq.(11)
        self.ensemble_epochs = 3
        self.ensemble_lr = 0.05

        # Knowledge distillation Eq.(12)
        self.distill_epochs = 1
        self.distill_lr = 0.001
        self.temperature = 3.0
        self.lambda_kd = 1e-7

        # Supervised CE stabilization
        self.distill_ce_weight = 0.5

        # Optimizer
        self.lr = 0.001
        self.weight_decay = 5e-4

        # Compatibility with original FedAvg.py
        self.local_epochs = 10


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':

    args = Args()

    begin_time = time.time()

    # ========================================================
    # Data processing
    #
    # Keep exactly the same data-loading format as FedAvg.py.
    # You can directly replace this part with other data loaders.
    # ========================================================
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

    # ========================================================
    # Initialize clients
    # ========================================================
    clients = [
        Client(
            train_loaders[i],
            test_loaders[i],
            args,
            i
        )
        for i in range(args.all_clients)
    ]

    # ========================================================
    # Run FOL
    # ========================================================
    fol = FOL(
        clients,
        args
    )

    fol.train()

    end_time = time.time()

    run_time = end_time - begin_time

    print(
        f"\nRunning time: "
        f"{run_time:.2f} seconds"
    )
