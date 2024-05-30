from torch.utils.data import Dataset
import random


class AnliTrainDataset(Dataset):
    def __init__(self, data, label, tokenizer, shuffle_type):
        self.data = data
        self.label = label
        self.tokenizer = tokenizer
        self.shuffle_type = shuffle_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        label_idx = self.label[idx]
        hyp_list = [1, 2]
        random.shuffle(hyp_list)
        label_idx = hyp_list.index(label_idx)

        if self.shuffle_type == 'hyp':
            tokenized = self.tokenizer([(
                f"Observation 1: {d['obs1']} Hypothesis {hyp_list[0]}: {d[f'hyp{hyp_list[0]}']} Observation 2: {d['obs2']}",
                f"Observation 1: {d['obs1']} Hypothesis {hyp_list[1]}: {d[f'hyp{hyp_list[1]}']} Observation 2: {d['obs2']}"
            )], truncation=True, return_tensors='pt')
        elif self.shuffle_type == 'hyp_obs':
            tokenized = self.tokenizer(
                [shuffle_hyp_obs(d['obs1'], d['obs2'], d[f'hyp{hyp_list[0]}'], d[f'hyp{hyp_list[1]}'])],
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
