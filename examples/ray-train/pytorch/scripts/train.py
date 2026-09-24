"""Distributed data-parallel PyTorch training on a Ray cluster inside a SageMaker
training job.

This is an ordinary single-node training loop that Ray Train fans out across every
GPU in the cluster. ``ray.train.torch.TorchTrainer`` launches one worker per GPU,
``prepare_model`` wraps the model in DistributedDataParallel (DDP), and
``prepare_data_loader`` shards the dataset so each worker sees a disjoint shard.
The rank-0 worker reports metrics and writes a checkpoint that Ray Train persists to
the shared ``storage_path`` (Amazon S3), so the run survives a worker or job restart.

The launcher (``launcher.py``) stands up the Ray head/worker topology and then runs
this script's ``__main__`` block, where Ray is already initialized.
"""

from argparse import ArgumentParser
import logging
import os
import sys
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import ray
import ray.train
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from model import build_resnet


def setup_logging():
    logger = logging.getLogger(__name__)
    if logger.handlers or hasattr(logger, "_configured"):
        return logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger._configured = True
    logger.propagate = False
    return logger


logger = setup_logging()

# SageMaker syncs anything written under this path to the CheckpointConfig S3 URI.
CHECKPOINT_DIR = os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")


def read_params():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Ray Train workers. 0 = auto-size from the cluster's GPU count.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.info(f"Ignoring unknown arguments: {unknown}")
    return args


def build_dataloaders(data_dir, batch_size):
    """CIFAR-10 train/val loaders. Downloaded once per worker node if absent."""
    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    )
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_tf = transforms.Compose([transforms.ToTensor(), normalize])

    train_ds = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf
    )
    val_ds = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=val_tf
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def train_func(config):
    """Runs on every Ray Train worker. This is the body Ray fans out."""
    epochs = config["epochs"]
    lr = config["learning_rate"]
    batch_size = config["batch_size"]
    data_dir = config["data_dir"]

    # DDP: wrap the model and move it to this worker's device.
    model = build_resnet(num_classes=10)
    model = ray.train.torch.prepare_model(model)

    # Shard the data so each worker trains on a disjoint slice.
    train_loader, val_loader = build_dataloaders(data_dir, batch_size)
    train_loader = ray.train.torch.prepare_data_loader(train_loader)
    val_loader = ray.train.torch.prepare_data_loader(val_loader)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Keeps shards different across epochs for a sharded (DistributedSampler) loader.
        if ray.train.get_context().get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)

        model.train()
        running = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()  # NCCL all-reduce of gradients happens here (over EFA on GPU nodes)
            running += loss.item()

        # Validation on rank-0's shard is enough for a running signal.
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / max(total, 1)
        avg_loss = running / max(len(train_loader), 1)

        # Only rank 0 writes the checkpoint; all workers must call report().
        metrics = {"epoch": epoch, "loss": avg_loss, "val_accuracy": val_acc}
        if ray.train.get_context().get_world_rank() == 0:
            with tempfile.TemporaryDirectory() as tmp:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.module.state_dict()
                        if hasattr(model, "module")
                        else model.state_dict(),
                    },
                    os.path.join(tmp, "checkpoint.pt"),
                )
                ray.train.report(metrics, checkpoint=Checkpoint.from_directory(tmp))
        else:
            ray.train.report(metrics)

        logger.info(
            f"epoch={epoch} loss={avg_loss:.4f} val_acc={val_acc:.4f}"
        )


def resolve_num_workers(requested):
    """Auto-size Ray Train workers to the cluster's GPU count (fallback: CPUs)."""
    if requested and requested > 0:
        return requested, bool(int(ray.cluster_resources().get("GPU", 0)))
    gpus = int(ray.cluster_resources().get("GPU", 0))
    if gpus > 0:
        return gpus, True
    cpus = int(ray.cluster_resources().get("CPU", 1))
    return max(1, cpus - 1), False


if __name__ == "__main__":
    args = read_params()
    logger.info(f"Training arguments: {args}")
    logger.info(f"Ray cluster resources: {ray.cluster_resources()}")

    num_workers, use_gpu = resolve_num_workers(args.num_workers)
    logger.info(f"Ray Train: num_workers={num_workers} use_gpu={use_gpu}")

    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "data_dir": args.data_dir,
        },
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        # storage_path is where Ray Train persists checkpoints. SageMaker syncs
        # CHECKPOINT_DIR to the CheckpointConfig S3 URI set in the notebook.
        run_config=RunConfig(
            name="ray-train-cifar10",
            storage_path=CHECKPOINT_DIR,
        ),
    )

    result = trainer.fit()
    logger.info(f"Training complete. Best metrics: {result.metrics}")
    if result.checkpoint:
        logger.info(f"Final checkpoint at: {result.checkpoint.path}")
