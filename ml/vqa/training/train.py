"""
Training script for the custom PyTorch VQAModel.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.vqa.config import ModelConfig, TrainConfig
from ml.vqa.model import VQAModel, VQALoss
from ml.vqa.training.dataset import VQADataset

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: VQAModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: VQALoss,
    device: str,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> float:
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        optimizer.zero_grad()

        images = batch["image"].to(device)
        question_ids = batch["question_ids"].to(device)
        targets = {"answer_label": batch["answer_label"].to(device)}

        if scaler is not None:
            with torch.cuda.amp.autocast():
                preds = model(image=images, question_ids=question_ids)
                loss_dict = criterion(preds, targets)
                loss = loss_dict["loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(image=images, question_ids=question_ids)
            loss_dict = criterion(preds, targets)
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


def validate(
    model: VQAModel,
    dataloader: DataLoader,
    device: str,
) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch["image"].to(device)
            question_ids = batch["question_ids"].to(device)
            labels = batch["answer_label"].to(device)

            preds = model(image=images, question_ids=question_ids)
            logits = preds["answer_logits"]
            predicted = logits.argmax(dim=-1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total if total > 0 else 0.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/rsvqa-lr")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    
    train_config = TrainConfig(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
    )
    model_config = ModelConfig()

    device = train_config.resolve_device()
    logger.info(f"Using device: {device}")

    # Datasets
    train_ds = VQADataset(data_root=train_config.data_root, split=train_config.split_train, config=train_config)
    val_ds = VQADataset(data_root=train_config.data_root, split=train_config.split_val, config=train_config)

    train_loader = DataLoader(
        train_ds, batch_size=train_config.batch_size, shuffle=True,
        num_workers=train_config.num_workers, pin_memory=train_config.pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_config.batch_size, shuffle=False,
        num_workers=train_config.num_workers, pin_memory=train_config.pin_memory
    )

    model = VQAModel(model_config).to(device)
    if device == "cuda" and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs via DataParallel!")
        model = torch.nn.DataParallel(model)
        
    criterion = VQALoss().to(device)

    # Optimizer
    # Separate learning rate for backbone
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": train_config.backbone_lr},
        {"params": head_params, "lr": train_config.lr},
    ], weight_decay=train_config.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if train_config.use_amp and device == "cuda" else None

    # Training Loop
    best_acc = 0.0
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)

    for epoch in range(1, train_config.epochs + 1):
        logger.info(f"Epoch {epoch}/{train_config.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        
        logger.info(f"Train Loss: {train_loss:.4f}")

        if epoch % train_config.eval_every == 0:
            val_acc = validate(model, val_loader, device)
            logger.info(f"Validation Accuracy: {val_acc * 100:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                ckpt_path = Path(train_config.checkpoint_dir) / "best.pt"
                
                # Unwrap DataParallel if necessary
                state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": model_config.__dict__,
                }, ckpt_path)
                logger.info(f"Saved best model to {ckpt_path}")
                
    return 0

if __name__ == "__main__":
    sys.exit(main())
