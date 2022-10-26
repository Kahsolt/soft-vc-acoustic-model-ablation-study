import os
import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from acoustic.dataset import MelDataset
from acoustic.model import *
from acoustic.utils import Metric, save_checkpoint, load_checkpoint, plot_spectrogram

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True


# Hyperparams
BATCH_SIZE          = 32
LEARNING_RATE       = 3e-4
BETAS               = (0.8, 0.99)
WEIGHT_DECAY        = 1e-5
STEPS               = 36000
LOG_INTERVAL        = 10
VALIDATION_INTERVAL = 1000
CHECKPOINT_INTERVAL = 3000

MODELS = {
  'baseline':     AcousticModel,
  'no_dropout':   AcousticModel_no_dropout,
  'no_IN':        AcousticModel_no_IN,
  'single_LSTM':  AcousticModel_single_LSTM,
  'only_Encoder': AcousticModel_only_Encoder,
  'only_Decoder': AcousticModel_only_Decoder,
  'tiny':         AcousticModel_tiny,
  'tiny_half':    AcousticModel_tiny_half,
}


def train(args):
  '''Logger'''
  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger(__name__)
  log_dir = args.log_path / "logs"
  log_dir.mkdir(exist_ok=True, parents=True)
  logger.setLevel(logging.INFO)
  handler = logging.FileHandler(log_dir / f"{args.log_path.stem}.log")
  handler.setLevel(logging.INFO)
  formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S")
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  writer = SummaryWriter(log_dir)

  ''' Model & Optimizer '''
  acoustic = MODELS[args.model]().to(device)
  param_cnt = sum(p.numel() for p in acoustic.parameters())

  optimizer = optim.AdamW(
    acoustic.parameters(),
    lr=LEARNING_RATE,
    betas=BETAS,
    weight_decay=WEIGHT_DECAY,
  )

  ''' Data '''
  train_dataset = MelDataset(root=args.data_path, listfp=args.listfp, train=True, split_ratio=args.split_ratio)
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=train_dataset.pad_collate,
                            num_workers=8, pin_memory=True, shuffle=True, drop_last=True)
  
  validation_dataset = MelDataset(root=args.data_path, listfp=args.listfp, train=False, split_ratio=args.split_ratio)
  validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False,
                                 num_workers=4, pin_memory=True)

  ''' Ckpt '''
  if args.resume is not None:
    global_step, best_loss = load_checkpoint(
      load_path=args.resume,
      acoustic=acoustic,
      optimizer=optimizer,
      device=device,
      logger=logger,
    )
  else:
    global_step, best_loss = 0, float("inf")

  ''' Bookkeeper '''
  n_epochs = STEPS // len(train_loader) + 1
  start_epoch = global_step // len(train_loader) + 1
  average_loss  = Metric()
  epoch_loss    = Metric()
  validation_loss = Metric()

  logger.info("**" * 40)
  logger.info(f"PyTorch version: {torch.__version__}")
  if torch.cuda.is_available():
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
    logger.info(f"CUDNN enabled: {torch.backends.cudnn.enabled}")
    logger.info(f"CUDNN deterministic: {torch.backends.cudnn.deterministic}")
    logger.info(f"CUDNN benchmark: {torch.backends.cudnn.benchmark}")
    logger.info(f"# of GPUS: {torch.cuda.device_count()}")
  logger.info(f"batch size: {BATCH_SIZE}")
  logger.info(f"iterations per epoch: {len(train_loader)}")
  logger.info(f"# of epochs: {n_epochs}")
  logger.info(f"started at epoch: {start_epoch}")
  logger.info(f"")
  logger.info(f'BATCH_SIZE: {BATCH_SIZE}')
  logger.info(f'LEARNING_RATE: {LEARNING_RATE}')
  logger.info(f'BETAS: {BETAS}')
  logger.info(f'WEIGHT_DECAY: {WEIGHT_DECAY}')
  logger.info(f'STEPS: {STEPS}')
  logger.info(f'LOG_INTERVAL: {LOG_INTERVAL}')
  logger.info(f'VALIDATION_INTERVAL: {VALIDATION_INTERVAL}')
  logger.info(f'CHECKPOINT_INTERVAL: {CHECKPOINT_INTERVAL}')
  logger.info(f"")
  logger.info(f"Model: {args.model}")
  logger.info(f'   param_cnt: {param_cnt}')
  logger.info("**" * 40 + "\n")

  ''' Train '''
  acoustic.train()
  for epoch in range(start_epoch, n_epochs + 1):
    epoch_loss.reset()

    for mels, mels_lengths, units, units_lengths in train_loader:
      mels,  mels_lengths  = mels .to(device), mels_lengths .to(device)
      units, units_lengths = units.to(device), units_lengths.to(device)

      optimizer.zero_grad()

      mels_ = acoustic(units, mels[:, :-1, :])

      loss = F.l1_loss(mels_, mels[:, 1:, :], reduction="none")
      loss = torch.sum(loss, dim=(1, 2)) / (mels_.size(-1) * mels_lengths)
      loss = torch.mean(loss)

      loss.backward()
      optimizer.step()

      global_step += 1

      average_loss.update(loss.item())
      epoch_loss  .update(loss.item())

      if global_step % LOG_INTERVAL == 0:
        logger.info(f">> [Step {global_step}] loss: {average_loss.value}")
        writer.add_scalar("train/loss", average_loss.value, global_step)
        average_loss.reset()

      if global_step % VALIDATION_INTERVAL == 0:
        acoustic.eval()
        validation_loss.reset()

        with torch.no_grad():
          for i, (mels, units) in enumerate(validation_loader, 1):
            mels, units = mels.to(device), units.to(device)

            mels_ = acoustic(units, mels[:, :-1, :])
            loss = F.l1_loss(mels_, mels[:, 1:, :])

            validation_loss.update(loss.item())

            if i < 4:   # display first three samples
              if global_step == VALIDATION_INTERVAL:  # if the first time
                writer.add_figure(f"original/mel_{i}",  plot_spectrogram(mels .squeeze().transpose(0, 1).cpu().numpy()), global_step)
              writer.add_figure(f"generated/mel_{i}", plot_spectrogram(mels_.squeeze().transpose(0, 1).cpu().numpy()), global_step)

        acoustic.train()

        writer.add_scalar("validation/loss", validation_loss.value, global_step)
        logger.info(f"valid -- epoch: {epoch}, loss: {validation_loss.value:.4f}")

        new_best = best_loss > validation_loss.value
        if new_best or CHECKPOINT_INTERVAL and global_step % CHECKPOINT_INTERVAL:
          if new_best:
            logger.info("-------- new best model found!")
            best_loss = validation_loss.value

        save_checkpoint(
            checkpoint_dir=args.log_path,
            acoustic=acoustic,
            optimizer=optimizer,
            step=global_step,
            loss=validation_loss.value,
            best=new_best,
            logger=logger,
        )

    logger.info(f"train -- epoch: {epoch}, loss: {epoch_loss.value:.4f}")


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("dataset", metavar='dataset', help="dataset name")
  parser.add_argument("--listfp", default='lists\databaker-full.txt', help="training list file path")
  parser.add_argument("--model", default='baseline', choices=MODELS.keys(), help="model architecture")
  parser.add_argument("--resume", help="checkpoint file path to resume from.", type=Path)
  parser.add_argument("--split_ratio", default=0.1, help="dataset valid/train split ratio.")
  args = parser.parse_args()

  list_name = os.path.splitext(os.path.basename(args.listfp))[0]
  args.data_path = Path('data') / args.dataset
  args.log_path  = Path('log')  / f'{args.model}_{list_name}'

  train(args)
