import argparse
import json
import random


def accuracy_score(y_true, y_pred):
    correct = sum([1 for true, pred in zip(y_true, y_pred) if true == pred])
    return correct / len(y_true)


def load_data(data_dir):
    with open(data_dir / 'train.jsonl') as f:
        train_data = [json.loads(line) for line in f]
    with open(data_dir / 'dev.jsonl') as f:
        dev_data = [json.loads(line) for line in f]
    with open(data_dir / 'train-labels.lst') as f:
        train_labels = [int(line.strip()) for line in f]
    with open(data_dir / 'dev-labels.lst') as f:
        dev_labels = [int(line.strip()) for line in f]

    train_len = int(0.8 * len(train_data))

    valid_data = train_data[train_len:]
    valid_labels = train_labels[train_len:]
    train_data = train_data[:train_len]
    train_labels = train_labels[:train_len]

    return train_data, valid_data, dev_data, train_labels, valid_labels, dev_labels


def get_args():
    parser = argparse.ArgumentParser(description='alphaNLI Training')
    parser.add_argument('--data_dir', type=str, default='data/alphanli-train-dev', help='data directory')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='output directory')
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-base', help='model name')
    # train
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')

    return parser.parse_args()
