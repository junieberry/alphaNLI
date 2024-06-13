import os
from pathlib import Path

import torch
from torch import nn, var

from dataset import AnliTrainDataset
from utils import *
from transformers import AutoConfig, AutoModel, AutoTokenizer, DataCollatorWithPadding, \
    AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import wandb

from tqdm import tqdm


def train_model(model, data, optimizer, scheduler):
    model.train()
    losses = []
    for batch in tqdm(data, desc="Training.."):
        for k, v in batch.items():
            batch[k] = v.to(args.device)
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        wandb.log({"train_loss": loss.item()})
        losses.append(loss.item())

    wandb.log({"epoch_loss": sum(losses) / len(losses)})


def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=1).unsqueeze(1)


def evaluate_model(model, data, mode="valid", save_logits=False):
    model.eval()
    predictions, labels, losses = [], [], []
    logits = []
    for batch in tqdm(data, desc="Evaluating..."):
        for k, v in batch.items():
            batch[k] = v.to(args.device)
        outputs = model(**batch)
        predictions += outputs.logits.argmax(dim=-1).tolist()
        if save_logits:
            logits += softmax(outputs.logits).tolist()

        labels += batch["labels"].tolist()

        if mode == "valid":
            loss = outputs.loss
            losses.append(loss.item())

    acc = accuracy_score(labels, predictions)
    print(f"Acc: {acc}")
    wandb.log({f"{mode}_accuracy": acc})
    if mode == "valid":
        wandb.log({f"{mode}_loss": sum(losses) / len(losses)})
    if save_logits:
        return acc, logits
    return acc


def finetune(args):
    wandb.init(
        project="alphaNLI",
        entity="junieberry",
        tags=[f"shuffle_{args.shuffle_type}"],
        config=vars(args),
    )

    data_path = Path(args.data_dir)
    checkpoint_path = Path(args.output_dir) / wandb.run.name
    os.makedirs(checkpoint_path, exist_ok=True)

    print(f"Loading Data...")
    train_data, valid_data, dev_data, train_labels, valid_labels, dev_labels = load_data(data_path)
    print(f"Loading Model...")
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = len(set(train_labels))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)
    model.to(args.device)

    train_dataset = AnliTrainDataset(train_data, train_labels, tokenizer, args.shuffle_type, mode="train")
    valid_dataset = AnliTrainDataset(valid_data, valid_labels, tokenizer, args.shuffle_type, mode="valid")
    test_dataset = AnliTrainDataset(dev_data, dev_labels, tokenizer, args.shuffle_type, mode="test")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(
        train_dataloader) * args.epochs)

    best_score = 0
    try:
        print(f"Training Model...")
        for epoch in range(args.epochs):
            print(f"\n\n{epoch} Epoch...")
            train_model(model, train_dataloader, optimizer, scheduler)
            val_loss = evaluate_model(model, valid_dataloader, mode="valid")
            if val_loss > best_score:
                best_score = val_loss
                torch.save(model.state_dict(), checkpoint_path / "best_model.pt")

        print("Loading the best model...")
        model.load_state_dict(torch.load(checkpoint_path / "best_model.pt"))
        acc, logits = evaluate_model(model, test_dataloader, mode="test", save_logits=True)
        with open(data_path / f"predictions_shuffle_{args.shuffle_type}.json", "w") as f:
            json.dump(logits, f)

    # if keyboard interrupt evaluate the model on test set
    except KeyboardInterrupt:
        print("Loading the best model...")
        model.load_state_dict(torch.load(checkpoint_path / "best_model.pt"))
        evaluate_model(model, test_dataloader, mode="test")


if __name__ == '__main__':
    args = get_args()
    finetune(args)
