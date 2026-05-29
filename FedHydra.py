#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6+

import argparse
import copy
import os
import csv
import sys
import warnings
import datetime
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from helpers.datasets import partition_data1
from helpers.synthesizers import SASynthesizer
from helpers.utils import (
    average_weights,
    DatasetSplit,
    KLCom,
    setup_seed,
    test
)
from models.generator import Generator
from models.nets import *

warnings.filterwarnings("ignore")

# =======================
# Device
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =======================
# Logger
# =======================
def get_logger(logpath):
    logger = logging.getLogger(logpath)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)

    fh = logging.FileHandler(logpath)
    fh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


# =======================
# Args
# =======================
def args_parser():
    parser = argparse.ArgumentParser()

    # Federated
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_users", type=int, default=10)
    parser.add_argument("--local_ep", type=int, default=100)
    parser.add_argument("--local_bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)

    # Dataset
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--partition", type=str, default="partial_ood")
    parser.add_argument("--beta", type=float, default=0.9)

    # Model
    parser.add_argument("--model", type=str, default="cifar10_resnet")
    parser.add_argument("--type", type=str, default="pretrain")

    # Distillation
    parser.add_argument("--T", type=float, default=1)
    parser.add_argument("--kl_alpha1", type=float, default=0)
    parser.add_argument("--kl_alpha2", type=float, default=1)
    parser.add_argument("--g_steps", type=int, default=20)
    parser.add_argument("--nz", type=int, default=256)

    # Misc
    parser.add_argument("--seed", type=int, default=600)
    parser.add_argument("--timestr", type=str, default="")
    parser.add_argument("--other", type=str, default="")

    return parser.parse_args()


# =======================
# Local Training
# =======================
class LocalUpdate:
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.train_loader = DataLoader(
            DatasetSplit(dataset, idxs),
            batch_size=args.local_bs,
            shuffle=True,
            num_workers=4,
        )

    def update_weights(self, model, client_id):
        model.train()
        model.to(device)

        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.args.lr, momentum=0.9
        )

        local_acc = []

        for ep in range(self.args.local_ep):
            losses = []
            pbar = tqdm(
                self.train_loader,
                desc=f"[Client {client_id}] Epoch {ep+1}/{self.args.local_ep}",
                leave=False,
            )

            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                output = model(images)
                loss = F.cross_entropy(output, labels)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            acc, _ = test(model, test_loader)
            local_acc.append(acc)

            args.flogger.info(
                f"[Client {client_id}] "
                f"Epoch {ep+1} | "
                f"TrainLoss={np.mean(losses):.4f} "
                f"TestAcc={acc:.2f}%"
            )

        return model.state_dict(), np.array(local_acc)


# =======================
# Models
# =======================
def get_model(args):
    if args.model == "fmnist_cnn":
        return CNNMnist().to(device)
    elif args.model == "fmnist_resnet":
        return Fashion_ResNet18().to(device)
    elif args.model == "cifar10_resnet":
        return cifar10_ResNet18().to(device)
    elif args.model == "cifar100_resnet":
        return cifar100_ResNet18().to(device)
    else:
        raise ValueError("Unknown model")


# =======================
# Main
# =======================
if __name__ == "__main__":

    args = args_parser()
    args.timestr = args.timestr or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/{args.timestr}_{args.dataset}_{args.model}.log"
    args.flogger = get_logger(log_path)

    setup_seed(args.seed)

    args.flogger.info("=" * 60)
    args.flogger.info("Start Experiment")
    args.flogger.info("=" * 60)

    # Dataset
    train_dataset, test_dataset, user_groups, _ = partition_data1(
        args.dataset,
        args.partition,
        beta=args.beta,
        num_users=args.num_users,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=4
    )

    global_model = get_model(args)

    # =======================
    # Pretrain
    # =======================
    if args.type == "pretrain":
        local_weights = []

        for cid in range(args.num_users):
            args.flogger.info(f"Training client {cid}/{args.num_users - 1}")
            local = LocalUpdate(args, train_dataset, user_groups[cid])
            w, _ = local.update_weights(copy.deepcopy(global_model), cid)
            local_weights.append(w)

        os.makedirs("pkl", exist_ok=True)
        torch.save(
            local_weights,
            f"pkl/{args.timestr}_{args.dataset}_{args.num_users}_{args.beta}.pkl",
        )

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        acc, loss = test(global_model, test_loader)
        args.flogger.info(
            f"[Global Model] Final TestAcc={acc:.2f}% Loss={loss:.4f}"
        )

    args.flogger.info("Experiment Finished")
