from torch.utils.data import Dataset
from noising.noiser import noise


class Seq2SeqDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_length=48):
        if inputs and targets:
            assert len(inputs) == len(targets)
        self.tokenizer = tokenizer
        self.data_size = len(targets)
        self.max_length = max_length
        self.inputs = self.tokenizer(
            inputs, truncation=True, max_length=self.max_length)
        self.targets = self.tokenizer(
            targets, truncation=True, max_length=self.max_length)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.inputs.items()}

        item['labels'] = [
            -100 if token == self.tokenizer.pad_token_id else
            token for token in self.targets["input_ids"][idx]]
        return item

    def __len__(self):
        return self.data_size


class DynamicTypoDataset(Dataset):
    def __init__(self, targets, tokenizer, max_length=48, typo_probs=1.0):
        self.max_length = max_length
        self.typo_probs = typo_probs
        self.tokenizer = tokenizer
        self.data_size = len(targets)
        self.targets = targets
        self.targets_encodings = self.tokenizer(
            targets, truncation=True, max_length=self.max_length)

    def __getitem__(self, idx):
        inputs = noise([self.targets[idx]], typo_probs=self.typo_probs)[0]
        item = self.tokenizer(inputs, truncation=True,
                              max_length=self.max_length)

        item['labels'] = [
            -100 if token == self.tokenizer.pad_token_id else
            token for token in self.targets_encodings["input_ids"][idx]]
        return item

    def __len__(self):
        return self.data_size
