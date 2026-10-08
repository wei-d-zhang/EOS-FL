# -*- coding: utf-8 -*-

import os
import csv
import copy
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset

from torchvision import datasets, transforms

from model import *
from data_loader1 import *
from overlap_data import create_client_dataloaders_partial_ood1


# ============================================================
# 1. Reproducibility
# ============================================================

torch.manual_seed(643)
np.random.seed(600)
random.seed(600)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(643)


# ============================================================
# 2. Utility functions
# ============================================================

def get_num_classes(dataset):
    if dataset == "Fashion":
        return 10
    elif dataset == "Cifar10":
        return 10
    elif dataset == "Cifar100":
        return 100
    elif dataset == "Gear":
        # Gear is loaded with ImageFolder; the class count is determined
        # by the directory structure rather than hard-coded here.
        root = "./data/Gear"
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"Gear dataset directory not found: {root}"
            )
        return len(datasets.ImageFolder(root=root).classes)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def build_model(dataset, device):
    model_mapping = {
        "Fashion": Fashion_ResNet18,
        "Cifar10": cifar10_ResNet18,
        "Cifar100": cifar100_ResNet18,
    }

    if dataset == "Gear":
        # Prefer a Gear-specific model if it is already defined in model.py.
        if "Gear_ResNet18" in globals():
            return Gear_ResNet18().to(device)
        if "gear_ResNet18" in globals():
            return gear_ResNet18().to(device)

        # If model.py does not define a Gear-specific network, a CIFAR-style
        # ResNet18 is appropriate for 3-channel 32x32 Gear images.
        return cifar10_ResNet18(
            num_classes=get_num_classes("Gear")
        ).to(device)

    if dataset not in model_mapping:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return model_mapping[dataset]().to(device)


def get_observed_classes(loader):
    """
    Obtain the set C_i of classes observed by a client.

    Works with common torchvision Dataset / Subset / custom Dataset
    implementations. As a fallback, it scans the dataloader once.
    """
    dataset = loader.dataset

    # torchvision datasets
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy().tolist()
        return sorted(set(int(y) for y in targets))

    # torch.utils.data.Subset
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        parent = dataset.dataset
        if hasattr(parent, "targets"):
            targets = parent.targets
            if torch.is_tensor(targets):
                targets = targets.cpu().numpy().tolist()
            return sorted(set(int(targets[i]) for i in dataset.indices))

    # Generic fallback
    classes = set()
    for _, labels in loader:
        if torch.is_tensor(labels):
            classes.update(labels.cpu().numpy().astype(int).tolist())
        else:
            classes.update(int(x) for x in labels)

    return sorted(classes)


def clone_state_dict(state_dict):
    return {k: v.detach().clone() for k, v in state_dict.items()}


def weighted_average_state_dict(state_dicts, weights=None):
    """
    FedAvg-style weighted aggregation.

    Floating tensors are averaged numerically.
    Integer buffers (e.g., BatchNorm counters) are copied from the first
    client to avoid invalid floating-point conversion.
    """
    if len(state_dicts) == 0:
        raise ValueError("state_dicts is empty")

    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    else:
        total = float(sum(weights))
        weights = [float(w) / total for w in weights]

    result = {}

    for key in state_dicts[0].keys():
        first = state_dicts[0][key]

        if torch.is_floating_point(first) or torch.is_complex(first):
            value = torch.zeros_like(first, dtype=first.dtype)
            for sd, w in zip(state_dicts, weights):
                value += sd[key].to(value.device, dtype=value.dtype) * w
            result[key] = value
        else:
            # For integer buffers such as num_batches_tracked.
            result[key] = first.clone()

    return result


def state_num_samples(loader):
    try:
        return len(loader.dataset)
    except Exception:
        return 1


# ============================================================
# 3. Client
# ============================================================

class Client(object):

    def __init__(self, local_trainloader, local_testloader, args, client_id):
        self.trainloader = local_trainloader
        self.testloader = local_testloader
        self.args = args
        self.client_id = client_id

        self.net = build_model(args.dataset, args.device)

        self.criterion = nn.CrossEntropyLoss()

        self.observed_classes = get_observed_classes(self.trainloader)
        self.num_samples = state_num_samples(self.trainloader)

    def train(self, net):
        """
        Local adaptation:
            theta_i <- theta_g
            theta_i <- SGD on D_i

        Corresponds to Section 4.2.1.
        """
        net.train()

        # Re-create optimizer because the model is synchronized from the
        # current global model at the beginning of each communication round.
        optimizer = optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        scheduler = None
        if self.args.use_lr_scheduler:
            milestones = [
                max(1, int(self.args.local_epochs * 0.5)),
                max(1, int(self.args.local_epochs * 0.75)),
            ]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=0.1,
            )

        for epoch in range(self.args.local_epochs):
            correct = 0
            total = 0
            loss_sum = 0.0

            for inputs, labels in self.trainloader:
                inputs = inputs.to(self.args.device)
                labels = labels.to(self.args.device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                loss_sum += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            if scheduler is not None:
                scheduler.step()

            train_acc = 100.0 * correct / max(total, 1)
            avg_loss = loss_sum / max(total, 1)

            if self.args.verbose_local:
                print(
                    f"Client {self.client_id} | "
                    f"Epoch {epoch + 1}/{self.args.local_epochs} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Train Acc: {train_acc:.2f}%"
                )

        return clone_state_dict(net.state_dict())

    @torch.no_grad()
    def compute_class_prototypes(self):
        """
        Compute input-space class prototypes from private client data.

        For each locally observed class k,

            p_i^k = 1 / |D_i^k| * sum_{(x,y) in D_i^k} x.

        Only the class prototype and its sample count are returned. Raw
        training samples remain on the client. The prototype is computed
        from the same tensors delivered by the local DataLoader, so its
        preprocessing is automatically consistent with local training.
        """
        class_sum = {}
        class_count = {}

        for inputs, labels in self.trainloader:
            inputs = inputs.detach().cpu().float()
            labels = labels.detach().cpu().long()

            for cls in torch.unique(labels).tolist():
                cls = int(cls)
                mask = labels == cls
                cls_inputs = inputs[mask]

                if cls not in class_sum:
                    class_sum[cls] = torch.zeros(
                        cls_inputs.shape[1:],
                        dtype=torch.float32
                    )
                    class_count[cls] = 0

                class_sum[cls] += cls_inputs.sum(dim=0)
                class_count[cls] += int(mask.sum().item())

        prototypes = {}
        for cls in sorted(class_sum.keys()):
            prototypes[cls] = {
                "prototype": class_sum[cls] / max(class_count[cls], 1),
                "count": class_count[cls]
            }

        return prototypes

    def fine_tune(self, net):
        """
        Optional post-grafting local refinement.

        This corresponds to the optional fine-tuning extension described
        after Algorithm 1. It introduces no additional communication.
        """
        if self.args.finetune_epochs <= 0:
            return

        net.train()

        optimizer = optim.SGD(
            net.parameters(),
            lr=self.args.finetune_lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        for epoch in range(self.args.finetune_epochs):
            for inputs, labels in self.trainloader:
                inputs = inputs.to(self.args.device)
                labels = labels.to(self.args.device)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

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

        return 100.0 * correct / max(total, 1)


# ============================================================
# 4. EOS-FL
# ============================================================

class EOSFL(object):

    def __init__(self, clients, args, prototype_loader=None):
        self.clients = clients
        self.args = args

        self.global_model = build_model(args.dataset, args.device)

        self.num_classes = get_num_classes(args.dataset)

        # Prototype-based auxiliary data D^proto.
        self.prototype_loader = prototype_loader

        # S_k[j] for k = 0,...,K-1.
        # Stored as flattened parameter vectors.
        self.sensitivity = None

        # Per-client accuracy history.
        self.clients_acc = [[] for _ in range(args.num_clients)]
        self.global_acc = []

    # --------------------------------------------------------
    # FedAvg aggregation
    # --------------------------------------------------------

    def avg_weights(self, local_weights, sample_counts=None):
        if sample_counts is None:
            return weighted_average_state_dict(local_weights)

        return weighted_average_state_dict(
            local_weights,
            weights=sample_counts
        )

    # --------------------------------------------------------
    # Public proxy sensitivity
    # --------------------------------------------------------

    def compute_classwise_sensitivity(self):
        """
        Equation (4):

              1
        S_k[j] = -------- sum_x | d f_k(theta_g; x) / d theta[j] |
              |X_k|

        The implementation accumulates the absolute gradient of the
        class-k logit over samples belonging to class k.

        The gradient is computed on the SERVER only.
        """

        if self.prototype_loader is None:
            raise RuntimeError(
                "prototype_loader is required for EOS-FL sensitivity analysis."
            )

        model = self.global_model
        model.eval()

        # Parameter metadata.
        trainable_names = []
        trainable_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_names.append(name)
                trainable_params.append(param)

        sensitivity = [
            torch.zeros(
                sum(p.numel() for p in trainable_params),
                device=self.args.device,
                dtype=torch.float32
            )
            for _ in range(self.num_classes)
        ]

        class_counts = torch.zeros(
            self.num_classes,
            device=self.args.device,
            dtype=torch.long
        )

        # The prototype set contains one sample per class; we still divide by the actual
        # number of samples per class for robustness.
        for inputs, labels in self.prototype_loader:
            inputs = inputs.to(self.args.device)
            labels = labels.to(self.args.device)

            batch_size = labels.size(0)

            for b in range(batch_size):
                x = inputs[b:b + 1]
                true_class = int(labels[b].item())

                model.zero_grad(set_to_none=True)

                logits = model(x)
                target_logit = logits[0, true_class]

                grads = torch.autograd.grad(
                    target_logit,
                    trainable_params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True
                )

                flat_grad = []

                for p, g in zip(trainable_params, grads):
                    if g is None:
                        flat_grad.append(torch.zeros_like(p).reshape(-1))
                    else:
                        flat_grad.append(g.detach().abs().reshape(-1))

                flat_grad = torch.cat(flat_grad)

                sensitivity[true_class] += flat_grad
                class_counts[true_class] += 1

        for k in range(self.num_classes):
            if class_counts[k] > 0:
                sensitivity[k] /= class_counts[k].float()

        self.sensitivity = sensitivity

        print("\n[Server] Class-wise sensitivity estimation finished.")
        print(
            "[Server] Samples per class:",
            class_counts.detach().cpu().tolist()
        )

        return sensitivity

    # --------------------------------------------------------
    # Flatten / restore parameter vectors
    # --------------------------------------------------------

    def _flatten_parameters(self, model):
        vectors = []

        for p in model.parameters():
            if p.requires_grad:
                vectors.append(p.detach().reshape(-1))

        if len(vectors) == 0:
            return torch.empty(0, device=self.args.device)

        return torch.cat(vectors)

    def _graft_state_dict(
        self,
        local_state,
        global_state,
        mask,
        alpha
    ):
        """
        Equation (7):

        theta_i^p =
            theta_i^local
            + alpha * M_i * (theta_g - theta_i^local)

        Only floating-point trainable parameters are grafted.
        Non-floating buffers are kept from the local model.
        """

        personalized = clone_state_dict(local_state)

        offset = 0

        # model parameters
        for name, p in self.global_model.named_parameters():
            if not p.requires_grad:
                continue

            numel = p.numel()

            m = mask[offset: offset + numel].view_as(p)
            offset += numel

            local_param = local_state[name]
            global_param = global_state[name]

            personalized[name] = (
                local_param
                + alpha * m.to(local_param.dtype)
                * (global_param - local_param)
            )

        # buffers remain local.
        return personalized

    # --------------------------------------------------------
    # Parameter grafting
    # --------------------------------------------------------

    def parameter_grafting(self, local_weights):
        """
        Equations (5)-(7).

        For each client:
          C_i^miss = C \\ C_i
          I_i      = mean_{k in C_i^miss} S_k
          M_i      = top-rho(I_i)
          theta_i^p = theta_i^local +
                      alpha M_i (theta_g - theta_i^local)
        """

        if self.sensitivity is None:
            raise RuntimeError(
                "Run compute_classwise_sensitivity() before grafting."
            )

        global_state = clone_state_dict(self.global_model.state_dict())

        personalized_weights = []

        all_classes = set(range(self.num_classes))

        for client_id, client in enumerate(self.clients):

            observed = set(client.observed_classes)
            missing = sorted(all_classes - observed)

            if len(missing) == 0:
                # No missing classes -> no class-specific correction.
                personalized_weights.append(
                    clone_state_dict(local_weights[client_id])
                )
                print(
                    f"[Server] Client {client_id}: "
                    f"no missing classes, skip grafting."
                )
                continue

            importance = torch.stack(
                [self.sensitivity[k] for k in missing],
                dim=0
            ).mean(dim=0)

            # Equation (6): select top rho fraction.
            rho = self.args.rho
            keep_num = max(1, int(np.ceil(importance.numel() * rho)))

            # More efficient and numerically stable than computing a full
            # quantile for very large ResNet parameter vectors.
            threshold = torch.topk(
                importance,
                k=keep_num,
                largest=True,
                sorted=False
            ).values.min()

            mask = (importance >= threshold).float()

            personalized = self._graft_state_dict(
                local_state=local_weights[client_id],
                global_state=global_state,
                mask=mask,
                alpha=self.args.alpha
            )

            personalized_weights.append(personalized)

            print(
                f"[Server] Client {client_id} | "
                f"Observed classes: {len(observed)} | "
                f"Missing classes: {len(missing)} | "
                f"Grafted parameters: "
                f"{int(mask.sum().item())}/{mask.numel()} "
                f"({100.0 * mask.mean().item():.4f}%)"
            )

        return personalized_weights

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    def evaluate(self, round_id):
        local_accs = []
        global_accs = []
        personalized_accs = []

        for client_id, client in enumerate(self.clients):

            local_acc = client.test(client.net)
            global_acc = client.test(self.global_model)

            local_accs.append(local_acc)
            global_accs.append(global_acc)

            self.clients_acc[client_id].append(local_acc)

        mean_local = float(np.mean(local_accs))
        mean_global = float(np.mean(global_accs))

        self.global_acc.append(mean_global)

        print(
            f"[Round {round_id}] "
            f"Global Test Acc: {mean_global:.2f}% | "
            f"Client Test Acc: {mean_local:.2f}%"
        )

        return mean_local, mean_global

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    def save_results(self):
        result_dir = os.path.join(
            "./results",
            self.args.dataset
        )
        os.makedirs(result_dir, exist_ok=True)

        summary_file = os.path.join(
            result_dir,
            f"{self.args.method}.csv"
        )

        with open(
            summary_file,
            "w",
            encoding="utf-8",
            newline=""
        ) as f:
            writer = csv.writer(f)

            for client_id in range(len(self.clients)):
                writer.writerow(self.clients_acc[client_id])

        print(f"[Result] Saved to: {summary_file}")

    # --------------------------------------------------------
    # Main training
    # --------------------------------------------------------

    def train(self):
        """
        Strict one-shot EOS-FL training.

        There is exactly one client-server communication round:
          1) server broadcasts the initial model;
          2) clients perform local training and upload local models/classes;
          3) server aggregates the uploaded models, computes class-wise
             sensitivity, and performs personalized parameter grafting;
          4) server sends the personalized models back to clients.

        The manuscript's T=100 statement is treated as a writing error and
        is intentionally not implemented.
        """

        print("\n" + "=" * 70)
        print("EOS-FL STRICT ONE-SHOT MODE")
        print("Communication rounds: 1")
        print("=" * 70)

        # ------------------------------------------------
        # Step 1. Broadcast initial global model
        # ------------------------------------------------
        initial_global = clone_state_dict(
            self.global_model.state_dict()
        )

        for client in self.clients:
            client.net.load_state_dict(initial_global)

        # ------------------------------------------------
        # Step 2. Local training
        # ------------------------------------------------
        local_weights = []
        sample_counts = []

        print("\n[Stage 1] Local training")

        for client in self.clients:
            print(
                f"\nClient {client.client_id} | "
                f"Observed classes: {client.observed_classes}"
            )

            local_w = client.train(client.net)
            local_weights.append(local_w)
            sample_counts.append(client.num_samples)

        # ------------------------------------------------
        # Step 3. Server aggregation
        # ------------------------------------------------
        print("\n[Stage 2] Server FedAvg aggregation")

        global_weights = self.avg_weights(
            local_weights,
            sample_counts
        )

        self.global_model.load_state_dict(global_weights)

        # ------------------------------------------------
        # Step 3.5. Prototype-based auxiliary data construction
        # ------------------------------------------------
        print("\n[Stage 3] Prototype-based auxiliary data construction")

        self.prototype_loader = build_prototype_proxy(
            self.clients,
            self.args
        )

        # ------------------------------------------------
        # Step 4. Server-side class sensitivity
        # ------------------------------------------------
        print("\n[Stage 4] Class-wise sensitivity estimation")

        self.compute_classwise_sensitivity()

        # ------------------------------------------------
        # Step 5. Personalized parameter grafting
        # ------------------------------------------------
        print("\n[Stage 5] Personalized parameter grafting")

        personalized_weights = self.parameter_grafting(
            local_weights
        )

        # ------------------------------------------------
        # Step 6. Download personalized model
        # ------------------------------------------------
        print("\n[Stage 6] Personalized model deployment")

        for client_id, client in enumerate(self.clients):
            client.net.load_state_dict(
                personalized_weights[client_id]
            )

        # Optional local refinement. This does NOT introduce another
        # communication round because the refinement remains client-side.
        if self.args.finetune_epochs > 0:
            print(
                "\n[Stage 7] Optional local fine-tuning "
                f"({self.args.finetune_epochs} epochs)"
            )

            for client in self.clients:
                client.fine_tune(client.net)

        # ------------------------------------------------
        # Step 7. Evaluation
        # ------------------------------------------------
        print("\n[Stage 8] Evaluation")

        local_accs = []
        global_accs = []

        for client_id, client in enumerate(self.clients):
            local_acc = client.test(client.net)
            global_acc = client.test(self.global_model)

            self.clients_acc[client_id].append(local_acc)

            local_accs.append(local_acc)
            global_accs.append(global_acc)

            print(
                f"Client {client_id} | "
                f"Personalized Acc: {local_acc:.2f}% | "
                f"Global Acc: {global_acc:.2f}%"
            )

        print(
            f"\nEOS-FL Personalized Mean Accuracy: "
            f"{np.mean(local_accs):.2f}%"
        )

        self.save_results()


# ============================================================
# 5. Prototype-based auxiliary data
# ============================================================

def build_prototype_proxy(clients, args):
    """
    Construct D^proto from client-side class prototypes.

    Each client computes one input-space prototype for every locally
    observed class. The server aggregates prototypes of the same class
    using the corresponding local class sample counts. No raw samples are
    transmitted to the server.

    Because the experimental setting assumes that every global class is
    observed by at least one client, every class must have at least one
    aggregated prototype. Otherwise the sensitivity of that class cannot
    be estimated from prototypes alone.
    """

    prototype_sum = {}
    prototype_count = {}

    print("\n[Prototype] Collecting local class prototypes...")

    for client in clients:
        local_prototypes = client.compute_class_prototypes()

        print(
            f"Client {client.client_id}: "
            f"{len(local_prototypes)} local prototypes -> "
            f"{sorted(local_prototypes.keys())}"
        )

        for cls, item in local_prototypes.items():
            proto = item["prototype"].float()
            count = int(item["count"])

            if cls not in prototype_sum:
                prototype_sum[cls] = torch.zeros_like(proto)
                prototype_count[cls] = 0

            prototype_sum[cls] += proto * count
            prototype_count[cls] += count

    missing_classes = [
        cls for cls in range(args.num_classes)
        if cls not in prototype_sum
    ]

    if missing_classes:
        raise RuntimeError(
            "Prototype coverage is incomplete. No client provides "
            f"samples for classes: {missing_classes}. "
            "Prototype-based EOS-FL requires every global class to "
            "appear on at least one client."
        )

    prototypes = []
    labels = []

    for cls in range(args.num_classes):
        prototype = (
            prototype_sum[cls] / max(prototype_count[cls], 1)
        ).float()

        prototypes.append(prototype)
        labels.append(cls)

    prototypes = torch.stack(prototypes, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    prototype_dataset = TensorDataset(
        prototypes,
        labels
    )

    prototype_loader = DataLoader(
        prototype_dataset,
        batch_size=args.prototype_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print(
        f"[Prototype] D^proto constructed: "
        f"{len(prototype_dataset)} class prototypes"
    )
    print(
        "[Prototype] Class coverage: "
        f"{len(prototypes)}/{args.num_classes}"
    )

    return prototype_loader


# ============================================================
# 6. Arguments
# ============================================================

class Args:

    def __init__(self):

        # ----------------------------
        # Dataset
        # ----------------------------
        self.dataset = "Fashion"   ####Cifar10/Cifar100/Fashion/Gear
        self.data_root = "./data"

        self.num_classes = get_num_classes(self.dataset)

        # ----------------------------
        # Federated setting
        # ----------------------------
        self.all_clients = 10
        self.num_clients = 10

        # EOS-FL is strictly one-shot: exactly one client-server
        # communication round. The previous T=100 setting was a writing
        # error in the manuscript and is not used in this implementation.
        self.communication_rounds = 1

        # ----------------------------
        # Local optimization
        # ----------------------------
        self.local_epochs = 100
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.batch_size = 64

        self.use_lr_scheduler = True

        # ----------------------------
        # EOS-FL
        # ----------------------------
        self.rho = 0.01
        self.alpha = 0.2

        # Optional fine-tuning after grafting.
        # Core EOS-FL = 0.
        self.finetune_epochs = 1
        self.finetune_lr = 0.001

        # ----------------------------
        # Prototype-based auxiliary data
        # ----------------------------
        # One input-space prototype is generated per class.
        self.prototype_batch_size = 32

        # ----------------------------
        # Missing-label setting
        # ----------------------------
        # 0.10 / 0.20 / 0.30 / 0.50
        self.missing_ratio = 0.5

        # ----------------------------
        # Runtime
        # ----------------------------
        self.device = (
            "cuda:1"
            if torch.cuda.is_available() and torch.cuda.device_count() > 1
            else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        self.num_workers = 0
        self.verbose_local = True

        self.method = (
            f"EOSFL_{self.dataset}_"
            f"missing{int(self.missing_ratio * 100)}"
        )


# ============================================================
# 7. Main
# ============================================================

if __name__ == "__main__":

    args = Args()

    begin_time = time.time()

    print("=" * 70)
    print("EOS-FL Reproduction")
    print("=" * 70)

    print(f"Dataset       : {args.dataset}")
    print(f"Clients       : {args.num_clients}")
    print(f"Missing ratio : {args.missing_ratio:.2f}")
    print(f"Communication : {args.communication_rounds} round (one-shot)")
    print(f"Local epochs  : {args.local_epochs}")
    print(f"rho           : {args.rho}")
    print(f"alpha         : {args.alpha}")
    print(f"Device        : {args.device}")

    # --------------------------------------------------------
    # Class-missing configuration
    # --------------------------------------------------------

    num_missing_classes = int(
        round(args.num_classes * args.missing_ratio)
    )

    num_train_classes = args.num_classes - num_missing_classes

    print(
        f"Nominal missing-class ratio: {args.missing_ratio:.2f} | "
        f"Missing classes per nominal setting: {num_missing_classes}"
    )

    # The supplied FedAvg code already uses this loader to construct
    # partially-overlapping class distributions. We keep that interface
    # unchanged so EOS-FL can be dropped into the same project.
    train_loaders, test_loaders, info = (
        create_client_dataloaders_partial_ood1(
            dataname=args.dataset,
            num_clients=args.num_clients,
            num_train_classes=num_train_classes,
            num_overlap_classes=num_train_classes,
            num_test_only_classes=num_missing_classes,
            batch_size=args.batch_size
        )
    )

    print("\n[Data information]")
    print(info)

    # --------------------------------------------------------
    # Clients
    # --------------------------------------------------------

    clients = [
        Client(
            train_loaders[i],
            test_loaders[i],
            args,
            i
        )
        for i in range(args.all_clients)
    ]

    for client in clients:
        print(
            f"Client {client.client_id}: "
            f"{len(client.observed_classes)} observed classes -> "
            f"{client.observed_classes}"
        )

    # --------------------------------------------------------
    # EOS-FL
    # --------------------------------------------------------

    # Prototypes are extracted after local training and therefore remain
    # inside the same one-shot client-server exchange.
    eos_fl = EOSFL(
        clients=clients,
        args=args,
        prototype_loader=None
    )

    eos_fl.train()

    end_time = time.time()

    run_time = end_time - begin_time

    print(
        f"\nTotal running time: "
        f"{run_time:.2f} seconds"
    )
