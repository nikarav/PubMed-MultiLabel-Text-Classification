from torch.utils.data import Dataset
import pandas as pd
import torch

# from ..util import preprocess


class TextDataset(Dataset):

    def __init__(
        self,
        data_path,
        tokenizer,
        features_cols,
        label_cols,
        *args,
        **kwargs,
    ):

        self._data_path = data_path
        self._tokenizer = tokenizer
        self._features_cols = features_cols
        self._label_cols = label_cols
        self._data = pd.read_csv(self._data_path)

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        row = self._data.loc[idx]
        features = row[self._features_cols].to_numpy()[0]
        labels = row[self._label_cols].to_numpy()[0].strip("'[]")
        labels = list(map(int, labels.split()))

        features = self._tokenizer(features)

        ids = torch.tensor(features["input_ids"], dtype=torch.long)
        masks = torch.tensor(features["attention_mask"], dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        # single sentence classification task, the token type IDs are not necessary.
        return ids, masks, labels
