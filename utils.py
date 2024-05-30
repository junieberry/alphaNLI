import argparse
import json
import random


def accuracy_score(y_true, y_pred):
    correct = sum([1 for true, pred in zip(y_true, y_pred) if true == pred])
    return correct / len(y_true)


def load_data(data_dir):
    with open(data_dir / 'train_train.jsonl') as f:
        train_data = [json.loads(line) for line in f]
    with open(data_dir / 'train_val.jsonl') as f:
        valid_data = [json.loads(line) for line in f]
    with open(data_dir / 'dev.jsonl') as f:
        dev_data = [json.loads(line) for line in f]
    with open(data_dir / 'train_train-labels.lst') as f:
        train_labels = [int(line.strip()) for line in f]
    with open(data_dir / 'train_val-labels.lst') as f:
        valid_labels = [int(line.strip()) for line in f]
    with open(data_dir / 'dev-labels.lst') as f:
        dev_labels = [int(line.strip()) for line in f]

    return train_data, valid_data, dev_data, train_labels, valid_labels, dev_labels


def get_args():
    parser = argparse.ArgumentParser(description='alphaNLI Training')
    parser.add_argument('--data_dir', type=str, default='data/alphanli-train-dev', help='data directory')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='output directory')
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-base', help='model name')
    parser.add_argument("--shuffle_type", type=str, default=['hyp', 'hyp_obs'], help='shuffle type')
    # train
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument("--tags", nargs='+', help='wandb tags')

    return parser.parse_args()
