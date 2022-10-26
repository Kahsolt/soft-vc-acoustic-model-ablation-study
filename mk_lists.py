#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/15 

import os
from argparse import ArgumentParser

import numpy as np
np.random.seed(114514)

SAMPLE_RATE = 16000
HOP_LENGTH = 160
TIME_CUTS = {
  '8h': 8 * 60 * 60,
  '4h': 4 * 60 * 60,
  '2h': 2 * 60 * 60,
  '1h': 1 * 60 * 60,
  '30min':  30 * 60,
  '10min':  10 * 60,
}

def make_lists(args):
  in_dp = os.path.join(args.data_path, args.dataset)
  mel_dp = os.path.join(in_dp, 'mels')
  mel_fns = os.listdir(mel_dp)
  os.makedirs(args.out_dir, exist_ok=True)

  # make full list
  np.random.shuffle(mel_fns)
  fp = os.path.join(args.out_dir, f'{args.dataset}-full.txt')
  with open(fp, 'w', encoding='utf-8') as fh:
    for fn in mel_fns:
      fh.write(fn)
      fh.write('\n')

  dur_cache = { }   # {'fn': time(float)}
  n_examples = { }  # {'list_name': len(selected_fns)}
  # make partial lists
  for list_name, time_cut in TIME_CUTS.items():
    np.random.shuffle(mel_fns)
    selected_fns = []
    for fn in mel_fns:
      selected_fns.append(fn)
      if fn not in dur_cache:
        mel = np.load(os.path.join(mel_dp, fn))
        dur_cache[fn] = mel.shape[1] / (SAMPLE_RATE / HOP_LENGTH)
      time_cut -= dur_cache[fn]
      if time_cut <= 0: break

    n_examples[list_name] = len(selected_fns)
    fp = os.path.join(args.out_dir, f'{args.dataset}-{list_name}.txt')
    with open(fp, 'w', encoding='utf-8') as fh:
      for fn in selected_fns:
        fh.write(fn)
        fh.write('\n')

  fp = os.path.join(args.out_dir, f'lists-{args.dataset}.txt')
  with open(fp, 'w', encoding='utf-8') as fh:
    fh.write('[n_exmaples]\n')
    for k, v in n_examples.items():
      fh.write(f'{k}: {v}')
      fh.write('\n')
    fh.write(f'full: {len(mel_fns)}')


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("dataset", metavar='dataset', help="dataset name")
  parser.add_argument("--data_path", default='data', help="data base path")
  parser.add_argument("--out_dir", default='lists', help="output dirname")
  args = parser.parse_args()

  make_lists(args)
