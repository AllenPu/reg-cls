import torch
import numpy as np
import random
import time
import argparse
import logging
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from scipy.stats import gmean

import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
# 
from loss_elr import ELR_reg

# from tensorboard_logger import Logger

from utils import (
    AverageMeter,
    ProgressMeter,
    save_checkpoint,
    adjust_learning_rate,
    prepare_folders,
    accuracy,
    cls_num_from_filename,
)

from resnet import resnet50
from loss import *
from datasets import IMDBWIKI

import os

os.environ["KMP_WARNINGS"] = "FALSE"
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--seed", type=int, default=0, help="seed")

# --- Added classification options -----------------
parser.add_argument(
    "--testset",
    type=str,
    default="test",
    choices=["test", "val"],
    help="Which set to report final scores on.",
)
parser.add_argument(
    "--erlambda", type=float, default=1.0, help="Regression loss weight."
)
####################
# newly added
####################
parser.add_argument(
    "--ce", type=bool, default=False, help="use ce loss weight or not."
)
parser.add_argument(
    "--celambda", type=float, default=0, help="weight of the ce loss weight."
)
#####################
parser.add_argument(
    "--losstype",
    type=str,
    default="mse",
    choices=["msecls", "mse"],
    help="Loss type",
)
parser.add_argument(
    "--cls_num", type=int, default=0, help="Number classes for classification."
)
parser.add_argument(
    "--cls_equalize",
    action="store_true",
    help="If we should equalize the classes, per sample.",
)
parser.add_argument(
    "--balance_data",
    action="store_true",
    help="Rebalance data",
)
parser.add_argument(
    "--batch_acu",
    default=1,
    type=int,
    help="Batch accumulation steps for accumulated gradients.",
)  # 1 GPU - 8
# --- Added classification options -----------------

# imbalanced related
# LDS
parser.add_argument(
    "--lds", action="store_true", default=False, help="whether to enable LDS"
)
parser.add_argument(
    "--lds_kernel",
    type=str,
    default="gaussian",
    choices=["gaussian", "triang", "laplace"],
    help="LDS kernel type",
)
parser.add_argument(
    "--lds_ks", type=int, default=5, help="LDS kernel size: should be odd number"
)
parser.add_argument(
    "--lds_sigma", type=float, default=1, help="LDS gaussian/laplace kernel sigma"
)
# FDS
parser.add_argument(
    "--fds", action="store_true", default=False, help="whether to enable FDS"
)
parser.add_argument(
    "--fds_kernel",
    type=str,
    default="gaussian",
    choices=["gaussian", "triang", "laplace"],
    help="FDS kernel type",
)
parser.add_argument(
    "--fds_ks", type=int, default=5, help="FDS kernel size: should be odd number"
)
parser.add_argument(
    "--fds_sigma", type=float, default=1, help="FDS gaussian/laplace kernel sigma"
)
parser.add_argument(
    "--start_update", type=int, default=0, help="which epoch to start FDS updating"
)
parser.add_argument(
    "--start_smooth",
    type=int,
    default=1,
    help="which epoch to start using FDS to smooth features",
)
parser.add_argument(
    "--bucket_num", type=int, default=100, help="maximum bucket considered for FDS"
)
parser.add_argument(
    "--bucket_start",
    type=int,
    default=0,
    choices=[0, 3],
    help="minimum(starting) bucket for FDS, 0 for IMDBWIKI, 3 for AgeDB",
)
parser.add_argument("--fds_mmt", type=float, default=0.9, help="FDS momentum")

# re-weighting: SQRT_INV / INV
parser.add_argument(
    "--reweight",
    type=str,
    default="none",
    choices=["none", "sqrt_inv", "inverse"],
    help="cost-sensitive reweighting scheme",
)
# two-stage training: RRT
parser.add_argument(
    "--retrain_fc",
    action="store_true",
    default=False,
    help="whether to retrain last regression layer (regressor)",
)

# training/optimization related
parser.add_argument(
    "--dataset",
    type=str,
    default="imdb_wiki",
    choices=["imdb_wiki", "agedb"],
    help="dataset name",
)
parser.add_argument("--data_dir", type=str, default="./data", help="data directory")
parser.add_argument("--model", type=str, default="resnet50", help="model name")
parser.add_argument(
    "--store_root",
    type=str,
    default="checkpoint",
    help="root path for storing checkpoints, logs",
)
parser.add_argument("--store_name", type=str, default="", help="experiment store name")
parser.add_argument("--gpu", type=int, default=None)
parser.add_argument(
    "--optimizer",
    type=str,
    default="sgd",
    choices=["adam", "sgd"],
    help="optimizer type",
)
parser.add_argument(
    "--loss",
    type=str,
    default="l1",
    choices=["mse", "l1", "focal_l1", "focal_mse", "huber"],
    help="training loss type",
)
parser.add_argument("--lr", type=float, default=1e-2, help="initial learning rate")
parser.add_argument("--epoch", type=int, default=90, help="number of epochs to train")
parser.add_argument("--momentum", type=float, default=0.9, help="optimizer momentum")
parser.add_argument(
    "--weight_decay", type=float, default=1e-4, help="optimizer weight decay"
)
parser.add_argument(
    "--schedule",
    type=int,
    nargs="*",
    default=[60, 80],
    help="lr schedule (when to drop lr by 10x)",
)
parser.add_argument("--batch_size", type=int, default=16, help="batch size")  # 256 / 8
parser.add_argument("--print_freq", type=int, default=100, help="logging frequency")
parser.add_argument(
    "--img_size", type=int, default=224, help="image size used in training"
)
parser.add_argument(
    "--workers", type=int, default=4, help="number of workers used in data loading"
)
# checkpoints
parser.add_argument(
    "--resume", type=str, default="", help="checkpoint file path to resume training"
)
parser.add_argument(
    "--pretrained",
    type=str,
    default="",
    help="checkpoint file path to load backbone weights",
)
parser.add_argument("--evaluate", action="store_true", help="evaluate only flag")

parser.set_defaults(augment=True)
args, unknown = parser.parse_known_args()
args.start_epoch, args.best_loss = 0, 1e5


# Extend the model name with the parse arguments
if len(args.store_name):
    args.store_name = f"_{args.store_name}"
if not args.lds and args.reweight != "none":
    args.store_name += f"_{args.reweight}"
if args.lds:
    args.store_name += f"_lds_{args.lds_kernel[:3]}_{args.lds_ks}"
    if args.lds_kernel in ["gaussian", "laplace"]:
        args.store_name += f"_{args.lds_sigma}"
if args.fds:
    args.store_name += f"_fds_{args.fds_kernel[:3]}_{args.fds_ks}"
    if args.fds_kernel in ["gaussian", "laplace"]:
        args.store_name += f"_{args.fds_sigma}"
    args.store_name += f"_{args.start_update}_{args.start_smooth}_{args.fds_mmt}"
if args.retrain_fc:
    args.store_name += f"_retrain_fc"
args.store_name = f"{args.dataset}_{args.model}{args.store_name}_{args.optimizer}_{args.loss}_{args.lr}_{args.batch_size}"
args.store_name += (
    "_seed" + str(args.seed) + "_lambda" + str(args.erlambda) + args.losstype
)
if args.losstype.startswith("msecls"):
    args.store_name += (
        "__cls"
        + str(args.cls_num)
        + ("_equ" if args.cls_equalize is True else "_noequ")
    )
prepare_folders(args)
print(f"Args: {args}")
print(f"Store name: {args.store_name}")


def set_seed(seed):
    """Define the seed for all the random packages."""
    seed = int(seed)
    # make everything deterministic -> seed setup
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def main():
    set_seed(args.seed)

    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    # Data
    print("=====> Preparing data...")
    print(f"File (.csv): {args.dataset}.csv")
    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))
    df_train, df_val, df_test = (
        df[df["split"] == "train"],
        df[df["split"] == "val"],
        df[df["split"] == "test"],
    )
    train_labels = df_train["age"]

    # Define the datasets to be used
    train_dataset = IMDBWIKI(
        args,
        df=df_train,
        cls_num=args.cls_num,
        split="train",
    )
    val_dataset = IMDBWIKI(
        args,
        df=df_val,
        cls_num=args.cls_num,  # To estimate accuracy, if wanted
        split="val",
    )
    test_dataset = IMDBWIKI(
        args,
        df=df_test,
        cls_num=args.cls_num,  # To estimate accuracy
        split="test",
    )
    #
    total_len_train = len(train_dataset)
    #
    num_classes = train_dataset.nclasses
    #
    # Define the data loaders over the datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    # If the code should run a validation round (for parameter searches) or a test round.
    if args.testset.endswith("test"):
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )
    else:
        test_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")

    # Define the deep model to be used
    model = define_model(args, train_loader.dataset.nclasses)
    print(f"Args: {args}")

    # Evaluate only the model without training
    if args.evaluate:
        resume_name = os.path.join(args.store_root, args.store_name, args.resume)
        nclasses = cls_num_from_filename(resume_name)
        # Then redefine the model cause the classes may have changed
        model = define_model(args, nclasses)
        assert args.resume, "Specify a trained model using [args.resume]"
        checkpoint = torch.load(resume_name)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print(
            f"===> Checkpoint '{args.resume}' loaded (epoch [{checkpoint['epoch']}]), testing..."
        )

        # Call the evaluation script
        validate(
            test_loader,
            model,
            train_labels=train_labels,
            prefix="Test",
            losstype=args.losstype,
        )
        return

    if args.retrain_fc:
        assert args.reweight != "none" and args.pretrained
        print("===> Retrain last regression layer only!")
        for name, param in model.named_parameters():
            if "fc" not in name and "linear" not in name:
                param.requires_grad = False

    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            if "linear" not in k and "fc" not in k:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        print(
            f"===> Pretrained weights found in total: [{len(list(new_state_dict.keys()))}]"
        )
        print(f"===> Pre-trained model loaded: {args.pretrained}")

    # Define the loss and the optimizer
    if not args.retrain_fc:
        optimizer = (
            torch.optim.Adam(model.parameters(), lr=args.lr)
            if args.optimizer == "adam"
            else torch.optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        )
    else:
        # optimize only the last linear layer
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        names = list(
            filter(
                lambda k: k is not None,
                [
                    k if v.requires_grad else None
                    for k, v in model.module.named_parameters()
                ],
            )
        )
        assert 1 <= len(parameters) <= 2  # fc.weight, fc.bias
        print(f"===> Only optimize parameters: {names}")
        optimizer = (
            torch.optim.Adam(parameters, lr=args.lr)
            if args.optimizer == "adam"
            else torch.optim.SGD(
                parameters,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        )

    # If the model should be resumed
    if args.resume:
        if os.path.isfile(args.resume):
            resume_name = os.path.join(args.store_root, args.store_name, args.resume)
            nclasses = cls_num_from_filename(resume_name)
            # Then redefine the model cause the classes may have changed
            model = define_model(args, nclasses)

            print(f"===> Loading checkpoint '{args.resume}'")
            checkpoint = (
                torch.load(resume_name)
                if args.gpu is None
                else torch.load(
                    resume_name, map_location=torch.device(f"cuda:{str(args.gpu)}")
                )
            )
            args.start_epoch = checkpoint["epoch"]
            args.best_loss = checkpoint["best_loss"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                f"===> Loaded checkpoint '{args.resume}' (Epoch [{checkpoint['epoch']}])"
            )
        else:
            print(f"===> No checkpoint found at '{args.resume}'")
    cudnn.benchmark = True

    # The loop over the training epochs ----------------------------------------------
    for epoch in range(args.start_epoch, args.epoch):
        adjust_learning_rate(optimizer, epoch, args)

        # One pass over the data
        train_loss = train(
            train_loader,
            model,
            optimizer,
            epoch,
            losstype=args.losstype,
            cls_num=args.cls_num,
            erlambda=args.erlambda,
            total_len = total_len_train
        )

        # Evaluate the model on the validation set
        val_loss_mse, val_loss_l1, val_loss_gmean = validate(
            val_loader,
            model,
            train_labels=train_labels,
            losstype=args.losstype,
            cls_num=args.cls_num,
        )

        # Pick the best model on the validation set across epochs
        loss_metric = val_loss_mse
        is_best = loss_metric < args.best_loss
        args.best_loss = min(loss_metric, args.best_loss)
        print(f"Best L1 Loss: {args.best_loss:.3f}")
        save_checkpoint(
            args,
            {
                "epoch": epoch,
                "model": args.model,
                "best_loss": args.best_loss,
                "state_dict": model.state_dict(),
                "cls_num": args.cls_num,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
        )
        print(
            f"Epoch #{epoch}: Train loss [{train_loss:.4f}]; "
            f"Val loss: MSE [{val_loss_mse:.4f}], L1 [{val_loss_l1:.4f}], G-Mean [{val_loss_gmean:.4f}]"
        )
    # ---------------------------------------------------------------

    # Evaluate the best checkpoint on the test set
    print("=" * 120)
    print("Test best model on testset...")
    checkpoint = torch.load(f"{args.store_root}/{args.store_name}/ckpt.best.pth.tar")
    model.load_state_dict(checkpoint["state_dict"])
    print(
        f"Loaded best model, epoch {checkpoint['epoch']}, best val loss {checkpoint['best_loss']:.4f}"
    )
    test_loss_mse, test_loss_l1, test_loss_gmean = validate(
        test_loader,
        model,
        train_labels=train_labels,
        losstype=args.losstype,
        prefix="Test",
        cls_num=args.cls_num,
    )
    print(
        f"Test loss: MSE [{test_loss_mse:.4f}], L1 [{test_loss_l1:.4f}], G-Mean [{test_loss_gmean:.4f}]\nDone"
    )


def train(train_loader, model, optimizer, epoch, losstype, cls_num, erlambda, total_len, args = 0):
    """The main training loop, passing 1x over the training data.
    Args:
        - train_loader: the dataloader of the training set
        - model: the model to be trained
        - optimizer: the optimizer to be used
        - epoch: the current epoch number
        - losstype: 'mse' or 'msecls' (if we use a classification head)
        - cls_num: the number of classes in the classification head
        - erlambda: the weight for the loss in the reg+cls
        - total_len : total length of the train dataset
    """

    batch_time = AverageMeter("Time", ":6.2f")
    data_time = AverageMeter("Data", ":6.4f")
    losses = AverageMeter(f"Loss ({args.loss.upper()})", ":.3f")

    # Define the classification loss
    cls_loss_criterion = None
    if losstype.startswith("msecls") and cls_num > 0:
        #cls_loss_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        cls_loss_criterion = ELR_reg(total_len, cls_num)
        accuracies = AverageMeter("Accuracy", ":.3f")
        losses_cls = AverageMeter("Loss-cls", ":.3f")
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, losses_cls, accuracies],
            prefix="Epoch: [{}]".format(epoch),
        )
    else:
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses],
            prefix="Epoch: [{}]".format(epoch),
        )

    model.train()
    end = time.time()

    # Loop over all training samples ------------------------------------------------------
    for idx, (inputs, targets, labels, weights, index) in enumerate(train_loader):
        data_time.update(time.time() - end)
        inputs, targets, labels, weights = (
            inputs.cuda(non_blocking=True),
            targets.cuda(non_blocking=True),
            labels.cuda(non_blocking=True),
            weights.cuda(non_blocking=True),
        )

        # Do the forward pass
        if args.fds:
            outputs_reg, outputs_cls, _ = model(inputs, targets, epoch)
        else:
            outputs_reg, outputs_cls = model(inputs, targets, epoch)

        # Hack to balanced the regression as well during training
        if args.balance_data:
            targets = targets.view(labels.shape)[labels != -1]
            outputs_reg = outputs_reg.view(labels.shape)[labels != -1]
            weights = weights.view(labels.shape)[labels != -1]
            keep_size = targets.size()[0]
            if keep_size < 1:
                continue

        # Compute the regression loss
        loss = globals()[f"weighted_{args.loss}_loss"](outputs_reg, targets, weights)
        assert not (
            np.isnan(loss.item()) or loss.item() > 1e6
        ), f"Loss explosion: {loss.item()}"
        losses.update(loss.item(), inputs.size(0))

        # Add a classification loss
        if cls_loss_criterion is not None:
            labels = labels.reshape(
                -1,
            ).contiguous()

            outputs_cls = outputs_cls[labels != -1]
            labels = labels[labels != -1]

            loss_cls = cls_loss_criterion(index = index, inputs=outputs_cls, targets=labels)

            keep_size = labels[labels != -1].size()[0]
            if keep_size > 0:
                batch_accuracy = accuracy(outputs_cls, labels)
                accuracies.update(batch_accuracy[0], keep_size)
                losses_cls.update(loss_cls.item(), keep_size)

            # Add the cls-loss to the reg-loss
            if erlambda > 1:
                totalloss = loss * erlambda + loss_cls
            else:
                totalloss = loss + loss_cls / erlambda
        else:
            totalloss = loss

        ###########################
        #
        ###########################
        if args.ce:
            ce_loss = torch.nn.functional.cross_entropy(outputs_cls, labels, ignore_index=-1)
            totalloss = args.celambda * ce_loss

        # Normalize the loss to accumulate gradients
        totalloss = totalloss / float(args.batch_acu)
        totalloss.backward()

        # Do the backward pass every <batch_acu> steps
        if ((idx + 1) % args.batch_acu == 0) or (idx + 1 == len(train_loader)):
            # Update Optimizer
            optimizer.step()
            # Zero grads at the end
            optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()
        if idx % args.print_freq == 0:
            progress.display(idx)
    # -----------------------------------------------------------------------------

    if args.fds and epoch >= args.start_update:
        print(f"Create Epoch [{epoch}] features of all training data...")
        encodings, labels = [], []
        with torch.no_grad():
            for (inputs, targets, _) in tqdm(train_loader):
                inputs = inputs.cuda(non_blocking=True)
                outputs, _, feature = model(inputs, targets, epoch)
                encodings.extend(feature.data.squeeze().cpu().numpy())
                labels.extend(targets.data.squeeze().cpu().numpy())

        encodings, labels = (
            torch.from_numpy(np.vstack(encodings)).cuda(),
            torch.from_numpy(np.hstack(labels)).cuda(),
        )
        model.module.FDS.update_last_epoch_stats(epoch)
        model.module.FDS.update_running_stats(encodings, labels, epoch)
    return losses.avg


def validate(val_loader, model, losstype, cls_num, train_labels=None, prefix="Val"):
    """Evaluated the model on the val/test sets.
    Args:
        - val_loader: the val/test dataloader
        - model: the model to be evaluated
        - cls_num: the number of classes
        - train_labels: needed in to show the metrics
        - prefix: for logging purposes
    """
    batch_time = AverageMeter("Time", ":6.3f")
    losses_mse = AverageMeter("Loss (MSE)", ":.3f")
    losses_l1 = AverageMeter("Loss (L1)", ":.3f")

    # Define the classification loss
    accuracies = None
    if losstype.startswith("msecls") and cls_num > 0:
        accuracies = AverageMeter("Accuracy", ":.3f")
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses_mse, losses_l1, accuracies],
            prefix=f"{prefix}: ",
        )
    else:
        progress = ProgressMeter(
            len(val_loader), [batch_time, losses_mse, losses_l1], prefix=f"{prefix}: "
        )

    # Define the regression metrics
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_gmean = nn.L1Loss(reduction="none")

    model.eval()
    losses_all = []
    preds, labels = [], []

    with torch.no_grad():
        end = time.time()

        # Loop over the test samples ------------------------------------------------------
        for idx, (inputs, targets, cls_labels, _, _) in enumerate(val_loader):
            inputs, targets, cls_labels = (
                inputs.cuda(non_blocking=True),
                targets.cuda(non_blocking=True),
                cls_labels.cuda(non_blocking=True),
            )

            # Get the model predictions
            outputs_reg, outputs_cls = model(inputs)

            # Store the predictions and the losses
            preds.extend(outputs_reg.data.cpu().numpy())
            labels.extend(targets.data.cpu().numpy())

            loss_mse = criterion_mse(outputs_reg, targets)
            loss_l1 = criterion_l1(outputs_reg, targets)
            loss_all = criterion_gmean(outputs_reg, targets)
            losses_all.extend(loss_all.cpu().numpy())

            losses_mse.update(loss_mse.item(), inputs.size(0))
            losses_l1.update(loss_l1.item(), inputs.size(0))

            # Checks class accuracy as well
            if accuracies is not None:
                cls_labels = cls_labels.reshape(
                    -1,
                ).contiguous()
                keep_size = cls_labels[cls_labels != -1].size()[0]

                if keep_size > 0:
                    batch_accuracy = accuracy(
                        outputs_cls[cls_labels != -1],
                        cls_labels[cls_labels != -1],
                    )
                    accuracies.update(batch_accuracy[0], keep_size)
            batch_time.update(time.time() - end)
            end = time.time()
            if idx % args.print_freq == 0:
                progress.display(idx)
        # ---------------------------------------------------------------------

        if accuracies is not None:
            print(
                "Test accuracy {acc.val:.4f} % ({acc.avg:.4f}) %\t".format(
                    acc=accuracies
                )
            )

        shot_dict = shot_metrics(np.hstack(preds), np.hstack(labels), train_labels)
        loss_gmean = gmean(np.hstack(losses_all), axis=None).astype(float)
        print(
            f" * Overall: MSE {losses_mse.avg:.3f}\tL1 {losses_l1.avg:.3f}\tG-Mean {loss_gmean:.3f}"
        )
        print(
            f" * Many: MSE {shot_dict['many']['mse']:.3f}\t"
            f"L1 {shot_dict['many']['l1']:.3f}\tG-Mean {shot_dict['many']['gmean']:.3f}"
        )
        print(
            f" * Median: MSE {shot_dict['median']['mse']:.3f}\t"
            f"L1 {shot_dict['median']['l1']:.3f}\tG-Mean {shot_dict['median']['gmean']:.3f}"
        )
        print(
            f" * Low: MSE {shot_dict['low']['mse']:.3f}\t"
            f"L1 {shot_dict['low']['l1']:.3f}\tG-Mean {shot_dict['low']['gmean']:.3f}"
        )
    return losses_mse.avg, losses_l1.avg, loss_gmean


def shot_metrics(preds, labels, train_labels, many_shot_thr=100, low_shot_thr=20):
    """Evaluates the predictions against the labels."""

    train_labels = np.array(train_labels).astype(int)
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(f"Type ({type(preds)}) of predictions not supported")

    train_class_count, test_class_count = [], []
    mse_per_class, l1_per_class, l1_all_per_class = [], [], []
    for l in np.unique(labels):
        train_class_count.append(len(train_labels[train_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        mse_per_class.append(np.sum((preds[labels == l] - labels[labels == l]) ** 2))
        l1_per_class.append(np.sum(np.abs(preds[labels == l] - labels[labels == l])))
        l1_all_per_class.append(np.abs(preds[labels == l] - labels[labels == l]))

    many_shot_mse, median_shot_mse, low_shot_mse = [], [], []
    many_shot_l1, median_shot_l1, low_shot_l1 = [], [], []
    many_shot_gmean, median_shot_gmean, low_shot_gmean = [], [], []
    many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot_mse.append(mse_per_class[i])
            many_shot_l1.append(l1_per_class[i])
            many_shot_gmean += list(l1_all_per_class[i])
            many_shot_cnt.append(test_class_count[i])
        elif train_class_count[i] < low_shot_thr:
            low_shot_mse.append(mse_per_class[i])
            low_shot_l1.append(l1_per_class[i])
            low_shot_gmean += list(l1_all_per_class[i])
            low_shot_cnt.append(test_class_count[i])
        else:
            median_shot_mse.append(mse_per_class[i])
            median_shot_l1.append(l1_per_class[i])
            median_shot_gmean += list(l1_all_per_class[i])
            median_shot_cnt.append(test_class_count[i])

    shot_dict = defaultdict(dict)
    shot_dict["many"]["mse"] = np.sum(many_shot_mse) / np.sum(many_shot_cnt)
    shot_dict["many"]["l1"] = np.sum(many_shot_l1) / np.sum(many_shot_cnt)
    shot_dict["many"]["gmean"] = gmean(np.hstack(many_shot_gmean), axis=None).astype(
        float
    )
    shot_dict["median"]["mse"] = np.sum(median_shot_mse) / np.sum(median_shot_cnt)
    shot_dict["median"]["l1"] = np.sum(median_shot_l1) / np.sum(median_shot_cnt)
    shot_dict["median"]["gmean"] = gmean(
        np.hstack(median_shot_gmean), axis=None
    ).astype(float)
    shot_dict["low"]["mse"] = np.sum(low_shot_mse) / np.sum(low_shot_cnt)
    shot_dict["low"]["l1"] = np.sum(low_shot_l1) / np.sum(low_shot_cnt)
    shot_dict["low"]["gmean"] = gmean(np.hstack(low_shot_gmean), axis=None).astype(
        float
    )

    return shot_dict


def define_model(args, nclasses):
    """Define the deep model:
    Args:
        - nclasses: the number of classes to use in the classification head.
    """
    # Model
    print("=====> Building model...")
    model = resnet50(
        fds=args.fds,
        bucket_num=args.bucket_num,
        bucket_start=args.bucket_start,
        start_update=args.start_update,
        start_smooth=args.start_smooth,
        kernel=args.fds_kernel,
        ks=args.fds_ks,
        sigma=args.fds_sigma,
        momentum=args.fds_mmt,
        cls_num=nclasses,
    )
    model = torch.nn.DataParallel(model).cuda()
    return model


if __name__ == "__main__":
    main()
