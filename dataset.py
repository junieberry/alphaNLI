from torch.utils.data import Dataset
import random


class AnliTrainDataset(Dataset):
    def __init__(self, data, label, tokenizer, shuffle_type, mode=None):
        self.data = data
        self.label = label
        self.tokenizer = tokenizer
        self.shuffle_type = shuffle_type
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        if self.shuffle_type == 'none':
            label_idx = self.label[idx] - 1

            tokenized = self.tokenizer([(
                f"Observation 1: {d['obs1']} Hypothesis 1: {d['hyp1']} Observation 2: {d['obs2']}",
                f"Observation 1: {d['obs1']} Hypothesis 2: {d['hyp2']} Observation 2: {d['obs2']}",
            )], truncation=True, return_tensors='pt')

        elif self.shuffle_type == 'hyp':
            if self.mode == 'train':
                label_idx = self.label[idx]
                hyp_list = [1, 2]
                random.shuffle(hyp_list)
                label_idx = hyp_list.index(label_idx)
            else:
                label_idx = self.label[idx] - 1
                hyp_list = [1, 2]
            tokenized = self.tokenizer([(
                f"Observation 1: {d['obs1']} Hypothesis {hyp_list[0]}: {d[f'hyp{hyp_list[0]}']} Observation 2: {d['obs2']}",
                f"Observation 1: {d['obs1']} Hypothesis {hyp_list[1]}: {d[f'hyp{hyp_list[1]}']} Observation 2: {d['obs2']}"
            )], truncation=True, return_tensors='pt')

        elif int(self.shuffle_type) in range(6):
            if self.mode == 'train':
                label_idx = self.label[idx]
                hyp_list = [1, 2]
                random.shuffle(hyp_list)
                label_idx = hyp_list.index(label_idx)
            else:
                label_idx = self.label[idx] - 1
                hyp_list = [1, 2]
            tokenized = self.tokenizer([(map_prompt_type(int(self.shuffle_type), d['obs1'], d['obs2'],
                                                         d[f'hyp{hyp_list[0]}'], d[f'hyp{hyp_list[1]}']))],
                                       truncation=True, return_tensors='pt')
        else:
            raise ValueError("Invalid shuffle type")

        return {
            'input_ids': tokenized['input_ids'][0],
            'attention_mask': tokenized['attention_mask'][0],
            'label': label_idx
        }


def shuffle_hyp_obs(o1, o2, h1, h2):
    prompt_type = random.randint(0, 5)
    return map_prompt_type(prompt_type, o1, o2, h1, h2)


def map_prompt_type(prompt_type, o1, o2, h1, h2):
    if prompt_type == 0:
        return (f"Observation 1: {o1} Hypothesis 1: {h1} Observation 2: {o2}",
                f"Observation 1: {o1} Hypothesis 2: {h2} Observation 2: {o2}")
    elif prompt_type == 1:
        return (f"Observation 1: {o1} Observation 2: {o2} Hypothesis 1: {h1}",
                f"Observation 1: {o1} Observation 2: {o2} Hypothesis 2: {h2}")
    elif prompt_type == 2:
        return (f"Hypothesis 1: {h1} Observation 1: {o1} Observation 2: {o2}",
                f"Hypothesis 2: {h2} Observation 1: {o1} Observation 2: {o2}")
    elif prompt_type == 3:
        return (f"Hypothesis 1: {h1} Observation 2: {o2} Observation 1: {o1}",
                f"Hypothesis 2: {h2} Observation 2: {o2} Observation 1: {o1}")
    elif prompt_type == 4:
        return (f"Observation 2: {o2} Hypothesis 1: {h1} Observation 1: {o1}",
                f"Observation 2: {o2} Hypothesis 2: {h2} Observation 1: {o1}")
    elif prompt_type == 5:
        return (f"Observation 2: {o2} Observation 1: {o1} Hypothesis 1: {h1}",
                f"Observation 2: {o2} Observation 1: {o1} Hypothesis 2: {h2}")
    else:
        raise ValueError(f"Invalid prompt type{prompt_type}")
