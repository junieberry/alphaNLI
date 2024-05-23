from torch.utils.data import Dataset
import random


class AnliTrainDataset(Dataset):
    def __init__(self, data, label, tokenizer):
        self.data = data
        self.label = label
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        l = self.label[idx] - 1

        tokenized = self.tokenizer(self.tokenizer.sep_token.join([d['obs1'], d['hyp1'], d['obs2'], d['obs1'],
                                                                  d['hyp2'], d['obs2']]),
                                   padding='max_length', truncation=True)

        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'label': l
        }
