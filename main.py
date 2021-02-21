import sys
import os
import time
import datetime

import random
import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from preprocess import preprocess_pipeline
from datasets import SensorDataset
from models import create
from utils.utils import paint, Logger, AverageMeter
from utils.utils_plot import plot_confusion
from utils.utils_pytorch import (
    get_info_params,
    get_info_layers,
    init_weights_orthogonal,
)
from utils.utils_mixup import mixup_data, MixUpLoss
from utils.utils_centerloss import compute_center_loss, get_center_delta

from settings import get_args

import warnings

warnings.filterwarnings("ignore")


def model_train(model, dataset, dataset_val, args):
    print(paint("[STEP 4] Running HAR training loop ..."))

    logger = SummaryWriter(log_dir=os.path.join(model.path_logs, "train"))
    logger_val = SummaryWriter(log_dir=os.path.join(model.path_logs, "val"))

    if args.weighted_sampler:
        print(paint("[-] Using weighted sampler (balanced batch)..."))
        sampler = WeightedRandomSampler(
            dataset.weight_samples, len(dataset.weight_samples)
        )
        loader = DataLoader(dataset, args.batch_size, sampler=sampler, pin_memory=True)
    else:
        loader = DataLoader(dataset, args.batch_size, True, pin_memory=True)
    loader_val = DataLoader(dataset_val, args.batch_size, False, pin_memory=True)

    criterion = nn.CrossEntropyLoss(reduction="mean").cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())

    if args.optimizer == "Adam":
        optimizer = optim.Adam(params, lr=args.lr)
    elif args.optimizer == "RMSprop":
        optimizer = optim.RMSprop(params, lr=args.lr)

    if args.lr_step > 0:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step, gamma=args.lr_decay
        )

    if args.init_weights == "orthogonal":
        print(paint("[-] Initializing weights (orthogonal)..."))
        model.apply(init_weights_orthogonal)

    metric_best = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        print("--" * 50)
        print("[-] Learning rate: ", optimizer.param_groups[0]["lr"])
        train_one_epoch(model, loader, criterion, optimizer, epoch, args)
        loss, acc, fm, fw = eval_one_epoch(
            model, loader, criterion, epoch, logger, args
        )
        loss_val, acc_val, fm_val, fw_val = eval_one_epoch(
            model, loader_val, criterion, epoch, logger_val, args
        )

        print(
            paint(
                f"[-] Epoch {epoch}/{args.epochs}"
                f"\tTrain loss: {loss:.2f} \tacc: {acc:.2f}(%)\tfm: {fm:.2f}(%)\tfw: {fw:.2f}(%)"
            )
        )

        print(
            paint(
                f"[-] Epoch {epoch}/{args.epochs}"
                f"\tVal loss: {loss_val:.2f} \tacc: {acc_val:.2f}(%)\tfm: {fm_val:.2f}(%)\tfw: {fw_val:.2f}(%)"
            )
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "random_rnd_state": random.getstate(),
            "numpy_rnd_state": np.random.get_state(),
            "torch_rnd_state": torch.get_rng_state(),
        }

        metric = fm_val
        if metric >= metric_best:
            print(paint(f"[*] Saving checkpoint... ({metric_best}->{metric})", "blue"))
            metric_best = metric
            torch.save(
                checkpoint, os.path.join(model.path_checkpoints, "checkpoint_best.pth")
            )

        if epoch % 5 == 0:
            torch.save(
                checkpoint,
                os.path.join(model.path_checkpoints, f"checkpoint_{epoch}.pth"),
            )

        if args.lr_step > 0:
            scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))

    print(paint(f"[STEP 4] Finished HAR training loop (h:m:s): {elapsed}"))
    print(paint("--" * 50, "blue"))


def train_one_epoch(model, loader, criterion, optimizer, epoch, args):

    losses = AverageMeter("Loss")
    model.train()

    for batch_idx, (data, target, idx) in enumerate(loader):
        data = data.cuda()
        target = target.view(-1).cuda()

        centers = model.centers

        if args.mixup:
            data, y_a_y_b_lam = mixup_data(data, target, args.alpha)

        z, logits = model(data)

        if args.mixup:
            criterion = MixUpLoss(criterion)
            loss = criterion(logits, y_a_y_b_lam)
        else:
            loss = criterion(logits, target)

        center_loss = compute_center_loss(z, centers, target)
        loss = loss + args.beta * center_loss

        losses.update(loss.item(), data.shape[0])

        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        center_deltas = get_center_delta(z.data, centers, target, args.lr_cent)
        model.centers = centers - center_deltas

        if batch_idx % args.print_freq == 0:
            print(f"[-] Batch {batch_idx}/{len(loader)}\t Loss: {str(losses)}")

        if args.mixup:
            criterion = criterion.get_old()


def eval_one_epoch(model, loader, criterion, epoch, logger, args):

    losses = AverageMeter("Loss")
    y_true, y_pred = [], []
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target, idx) in enumerate(loader):
            data = data.cuda()
            target = target.cuda()

            z, logits = model(data)
            loss = criterion(logits, target.view(-1))
            losses.update(loss.item(), data.shape[0])

            probabilities = nn.Softmax(dim=1)(logits)
            _, predictions = torch.max(probabilities, 1)

            y_pred.append(predictions.cpu().numpy().reshape(-1))
            y_true.append(target.cpu().numpy().reshape(-1))

    # append invalid samples at the beginning of the test sequence
    if loader.dataset.prefix == "test":
        ws = data.shape[1] - 1
        samples_invalid = [y_true[0][0]] * ws
        y_true.append(samples_invalid)
        y_pred.append(samples_invalid)

    y_true = np.concatenate(y_true, 0)
    y_pred = np.concatenate(y_pred, 0)

    acc = 100.0 * metrics.accuracy_score(y_true, y_pred)
    fm = 100.0 * metrics.f1_score(y_true, y_pred, average="macro")
    fw = 100.0 * metrics.f1_score(y_true, y_pred, average="weighted")

    if logger:
        logger.add_scalars("Loss", {"CrossEntropy": losses.avg}, epoch)
        logger.add_scalar("Acc", acc, epoch)
        logger.add_scalar("Fm", fm, epoch)
        logger.add_scalar("Fw", fw, epoch)

    if epoch % 50 == 0 or not args.train_mode:
        plot_confusion(
            y_true,
            y_pred,
            os.path.join(model.path_visuals, f"cm/{loader.dataset.prefix}"),
            epoch,
            class_map=args.class_map,
        )

    return losses.avg, acc, fm, fw


def model_eval(model, dataset_test, args):
    print(paint("[STEP 5] Running HAR evaluation loop ..."))

    loader_test = DataLoader(dataset_test, args.batch_size, False, pin_memory=True)

    criterion = nn.CrossEntropyLoss(reduction="mean").cuda()

    print("[-] Loading checkpoint ...")
    if args.train_mode:
        path_checkpoint = os.path.join(model.path_checkpoints, "checkpoint_best.pth")
    else:
        path_checkpoint = os.path.join(f"./weights/checkpoint_{args.dataset}.pth")

    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion.load_state_dict(checkpoint["criterion_state_dict"])

    start_time = time.time()

    loss_test, acc_test, fm_test, fw_test = eval_one_epoch(
        model, loader_test, criterion, -1, logger=None, args=args
    )

    print(
        paint(
            f"[-] Test loss: {loss_test:.2f}"
            f"\tacc: {acc_test:.2f}(%)\tfm: {fm_test:.2f}(%)\tfw: {fw_test:.2f}(%)"
        )
    )

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print(paint(f"[STEP 5] Finished HAR evaluation loop (h:m:s): {elapsed}"))


def main():

    # get experiment arguments
    args, config_dataset, config_model = get_args()

    # [STEP 0 and 1] load the .mat files (sample-level) and partition the datasets (segment-level)
    preprocess_pipeline(args)

    if args.train_mode:

        # [STEP 2] create HAR datasets
        dataset = SensorDataset(**config_dataset, prefix="train")
        dataset_val = SensorDataset(**config_dataset, prefix="val")

        # [STEP 3] create HAR models
        if torch.cuda.is_available():
            model = create(args.model, config_model).cuda()
            torch.backends.cudnn.benchmark = True
            sys.stdout = Logger(
                os.path.join(model.path_logs, f"log_main_{args.experiment}.txt")
            )

        # show args
        print("##" * 50)
        print(paint(f"Experiment: {model.experiment}", "blue"))
        print(
            paint(
                f"[-] Using {torch.cuda.device_count()} GPU: {torch.cuda.is_available()}"
            )
        )
        print(args)
        get_info_params(model)
        get_info_layers(model)
        print("##" * 50)

        # [STEP 4] train HAR models
        model_train(model, dataset, dataset_val, args)

    # [STEP 5] evaluate HAR models
    dataset_test = SensorDataset(**config_dataset, prefix="test")
    if not args.train_mode:
        config_model["experiment"] = "inference"
        model = create(args.model, config_model).cuda()
    model_eval(model, dataset_test, args)


if __name__ == "__main__":
    main()
