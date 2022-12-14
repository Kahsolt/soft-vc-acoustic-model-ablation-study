#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/12 

import os
from argparse import ArgumentParser
from traceback import print_exc

import torch
import torchaudio
from scipy.io import wavfile

from train import MODELS, device


def convert(args):
  hubert   = torch.hub.load("bshall/hubert:main",          "hubert_soft").to(device)
  hifigan  = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_soft").to(device)
  acoustic = MODELS[args.model]().to(device)

  ckpt = torch.load(f'log/{args.name}/model-best.pt', map_location=device)
  acoustic.load_state_dict(ckpt["acoustic-model"])

  if os.path.isfile(args.input):
    wav_fps = [args.input]
  else:
    wav_fps = [os.path.join(args.input, fn) for fn in os.listdir(args.input)]
  os.makedirs(args.out_dp, exist_ok=True)
  
  SAMPLE_RATE = 16000
  with torch.inference_mode():
    for wav_fp in wav_fps:
      try:
        source, sr = torchaudio.load(wav_fp)
        source = torchaudio.functional.resample(source, sr, SAMPLE_RATE)
        source = source.unsqueeze(0).to(device)
        
        units = hubert.units(source)
        mel = acoustic.generate(units).transpose(1, 2)
        target = hifigan(mel)

        y_hat = target.squeeze().cpu().numpy()
        name, ext = os.path.splitext(os.path.basename(wav_fp))
        save_fp = os.path.join(args.out_dp, f'{name}_{args.name}{ext}')
        wavfile.write(save_fp, SAMPLE_RATE, y_hat)
        print(f'>> {save_fp}')
      except Exception as e:
        print_exc()
        #print(f'<< [Error] {e}')
        print(f'<< ignore file {wav_fp}')


if __name__ == '__main__':
  NAMES = os.listdir('log')   # where ckpt locates

  parser = ArgumentParser()
  parser.add_argument("name", metavar='name', choices=NAMES, help='experiment names in `log` folder')
  parser.add_argument("input", metavar='input', help='input file or folder for conversion')
  parser.add_argument("--out_dp", default='gen', help='output folder for converted wavfiles')
  args = parser.parse_args()

  try:
    model, list = args.name.split('_')
  except:
    cp = args.name.rindex('_')
    model, list = args.name[:cp], args.name[cp+1:]

  args.model = model
  args.list = list

  convert(args)
