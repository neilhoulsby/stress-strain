import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

MAX_LEN = 1001
TEST_SIZE = 32
DATA_PATH = "data.npz"

class DictDataset(Dataset):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __getitem__(self, index):
        return {key: values[index] for key, values in self.dictionary.items()}

    def __len__(self):
        return len(next(iter(self.dictionary.values())))

def get_datasets(
    batch_size,
    train_size,
    max_len=MAX_LEN,
    test_size=TEST_SIZE,
):
    with open(DATA_PATH, "rb") as f:
        data = np.load(f)
        RAW_DATA = data["arr_0"]

    data = RAW_DATA[:, :max_len, :]
    train_x = torch.FloatTensor(data[:train_size, :max_len, 0][..., None])
    train_y = torch.FloatTensor(data[:train_size, :max_len, 1][..., None])
    test_x = torch.FloatTensor(data[-test_size:, :max_len, 0][..., None])
    test_y = torch.FloatTensor(data[-test_size:, :max_len, 1][..., None])

    train_ds = DictDataset({"x": train_x, "y": train_y})
    test_ds = DictDataset({"x": test_x, "y": test_y})

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
