from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class MelDataset(Dataset):

  def __init__(self, root: Path, listfp: Path, train: bool = True, split_ratio: float = 0.1):
    self.mels_dir  = root / "mels"
    self.units_dir = root / "units"

    with open(listfp) as fh:
      fns = [line.strip() for line in fh.read().split('\n') if line.strip()]

    cp = int(len(fns) * split_ratio)
    if train: fns = fns[cp:]
    else:     fns = fns[:cp]

    self.metadata = fns

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, index):
    path = self.metadata[index]
    mel_path   = self.mels_dir / path
    units_path = self.units_dir / path

    mel   = np.load(mel_path  .with_suffix(".npy")).T
    units = np.load(units_path.with_suffix(".npy"))

    length = 2 * units.shape[0]

    mel = torch.from_numpy(mel[:length, :])
    mel = F.pad(mel, (0, 0, 1, 0))
    units = torch.from_numpy(units)
    return mel, units

  def pad_collate(self, batch):
    mels, units = zip(*batch)
    mels, units = list(mels), list(units)

    mels_lengths  = torch.tensor([x.size(0) - 1 for x in mels])
    units_lengths = torch.tensor([x.size(0) for x in units])

    mels  = pad_sequence(mels, batch_first=True)
    units = pad_sequence(units, batch_first=True, padding_value=0)

    return mels, mels_lengths, units, units_lengths
