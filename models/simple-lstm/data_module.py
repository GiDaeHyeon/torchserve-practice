from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
try:
    from data.dataset import IMDB
except ModuleNotFoundError:
    import sys
    sys.path.append('./')
    from data.dataset import IMDB
from utils import SimpleTokenizer


class IMDBDataset(Dataset):
    def __init__(self, mode: str, tokenizer: Optional[SimpleTokenizer] = None, max_length: int = 256) -> None:
        super().__init__()
        self.data = IMDB(mode=mode)
        self.datas, self.labels = IMDB(mode=mode).get_data()
        self.max_length = max_length
        if tokenizer is None:
            self.tokenizer = SimpleTokenizer()
            self.tokenizer.fit(corpus=self.datas)
            self.num_words = self.tokenizer.get_words_num()
        else:
            self.tokenizer = tokenizer
            self.num_words = self.tokenizer.get_words_num()

    def __len__(self) -> int:
        return len(self.datas)

    def __getitem__(self, idx: int) -> tuple:
        tokens = self.tokenizer(self.datas[idx])

        if len(tokens) < self.max_length:
            for i in range(self.max_length - len(tokens)):
                tokens += [2]  # PAD
        elif len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        elif len(tokens) == self.max_length:
            pass

        return torch.Tensor(tokens).long(), torch.Tensor([self.labels[idx]]).int()


class SimpleDataModule(LightningDataModule):
    def __init__(self, batch_size: int, max_length: int) -> None:
        super().__init__()
        self.train_dataset = IMDBDataset(mode='train', max_length=max_length)
        self.val_dataset = IMDBDataset(mode='test', tokenizer=self.train_dataset.tokenizer, max_length=max_length)
        self.batch_size = batch_size
        self.num_embeddings = self.train_dataset.num_words

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=0,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.batch_size * 2,
                          num_workers=0,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=False)
