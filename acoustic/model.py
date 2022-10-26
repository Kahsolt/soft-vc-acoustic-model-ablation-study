import torch
import torch.nn as nn


''' Module '''
class PreNet(nn.Module):
  def __init__(self, input_size: int, hidden_size: int, output_size: int, 
               dropout: float = 0.5, use_drop: bool = True):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(input_size, hidden_size),
      nn.ReLU(),
      nn.Dropout(dropout) if use_drop else nn.Identity(),
      nn.Linear(hidden_size, output_size),
      nn.ReLU(),
      nn.Dropout(dropout) if use_drop else nn.Identity(),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)


class Encoder(nn.Module):
  def __init__(self, use_in: bool = True, use_drop: bool = True):
    super().__init__()
    self.prenet = PreNet(256, 256, 256, use_drop=use_drop)
    self.convs = nn.Sequential(
      nn.Conv1d(256, 512, 5, 1, 2),
      nn.ReLU(),
      nn.InstanceNorm1d(512) if use_in else nn.Identity(),
      nn.ConvTranspose1d(512, 512, 4, 2, 1),
      nn.Conv1d(512, 512, 5, 1, 2),
      nn.ReLU(),
      nn.InstanceNorm1d(512) if use_in else nn.Identity(),
      nn.Conv1d(512, 512, 5, 1, 2),
      nn.ReLU(),
      nn.InstanceNorm1d(512) if use_in else nn.Identity(),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.prenet(x)
    x = self.convs(x.transpose(1, 2))
    return x.transpose(1, 2)


class Decoder(nn.Module):
  def __init__(self, use_drop: bool = True):
    super().__init__()
    self.prenet = PreNet(128, 256, 256, use_drop=use_drop)
    self.lstm1 = nn.LSTM(512 + 256, 768, batch_first=True)
    self.lstm2 = nn.LSTM(768, 768, batch_first=True)
    self.lstm3 = nn.LSTM(768, 768, batch_first=True)
    self.proj = nn.Linear(768, 128, bias=False)

  def forward(self, x: torch.Tensor, mels: torch.Tensor) -> torch.Tensor:
    mels = self.prenet(mels)
    x, _ = self.lstm1(torch.cat((x, mels), dim=-1))
    res = x
    x, _ = self.lstm2(x)
    x = res + x
    res = x
    x, _ = self.lstm3(x)
    x = res + x
    return self.proj(x)

  @torch.inference_mode()
  def generate(self, xs: torch.Tensor) -> torch.Tensor:
    m = torch.zeros(xs.size(0), 128, device=xs.device)
    h1 = torch.zeros(1, xs.size(0), 768, device=xs.device)
    c1 = torch.zeros(1, xs.size(0), 768, device=xs.device)
    h2 = torch.zeros(1, xs.size(0), 768, device=xs.device)
    c2 = torch.zeros(1, xs.size(0), 768, device=xs.device)
    h3 = torch.zeros(1, xs.size(0), 768, device=xs.device)
    c3 = torch.zeros(1, xs.size(0), 768, device=xs.device)

    mel = []
    for x in torch.unbind(xs, dim=1):
      m = self.prenet(m)
      x = torch.cat((x, m), dim=1).unsqueeze(1)
      x1, (h1, c1) = self.lstm1(x, (h1, c1))
      x2, (h2, c2) = self.lstm2(x1, (h2, c2))
      x = x1 + x2
      x3, (h3, c3) = self.lstm3(x, (h3, c3))
      x = x + x3
      m = self.proj(x).squeeze(1)
      mel.append(m)
    return torch.stack(mel, dim=1)


class Decoder_single_LSTM(nn.Module):
  def __init__(self, use_drop: bool = True):
    super().__init__()
    self.prenet = PreNet(128, 256, 256, use_drop=use_drop)
    self.lstm = nn.LSTM(512 + 256, 768, batch_first=True)
    self.proj = nn.Linear(768, 128, bias=False)

  def forward(self, x: torch.Tensor, mels: torch.Tensor) -> torch.Tensor:
    mels = self.prenet(mels)
    x, _ = self.lstm(torch.cat((x, mels), dim=-1))
    return self.proj(x)

  @torch.inference_mode()
  def generate(self, xs: torch.Tensor) -> torch.Tensor:
    m = torch.zeros(xs.size(0), 128, device=xs.device)
    h = torch.zeros(1, xs.size(0), 768, device=xs.device)
    c = torch.zeros(1, xs.size(0), 768, device=xs.device)

    mel = []
    for x in torch.unbind(xs, dim=1):
      m = self.prenet(m)
      x = torch.cat((x, m), dim=1).unsqueeze(1)
      x, (h, c) = self.lstm(x, (h, c))
      m = self.proj(x).squeeze(1)
      mel.append(m)
    return torch.stack(mel, dim=1)


''' Model '''
class AcousticModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()

  def forward(self, x: torch.Tensor, mels: torch.Tensor) -> torch.Tensor:
    x = self.encoder(x)
    return self.decoder(x, mels)

  @torch.inference_mode()
  def generate(self, x: torch.Tensor) -> torch.Tensor:
    x = self.encoder(x)
    return self.decoder.generate(x)


class AcousticModel_no_dropout(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = Encoder(use_drop=False)
    self.decoder = Decoder(use_drop=False)

  def forward(self, x: torch.Tensor, mels: torch.Tensor) -> torch.Tensor:
    x = self.encoder(x)
    return self.decoder(x, mels)

  @torch.inference_mode()
  def generate(self, x: torch.Tensor) -> torch.Tensor:
    x = self.encoder(x)
    return self.decoder.generate(x)


class AcousticModel_no_IN(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = Encoder(use_in=False)
    self.decoder = Decoder()

  def forward(self, x: torch.Tensor, mels: torch.Tensor) -> torch.Tensor:
    x = self.encoder(x)
    return self.decoder(x, mels)

  @torch.inference_mode()
  def generate(self, x: torch.Tensor) -> torch.Tensor:
    x = self.encoder(x)
    return self.decoder.generate(x)


class AcousticModel_single_LSTM(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = Encoder()
    self.decoder = Decoder_single_LSTM()

  def forward(self, x: torch.Tensor, mels: torch.Tensor) -> torch.Tensor:
    x = self.encoder(x)
    return self.decoder(x, mels)

  @torch.inference_mode()
  def generate(self, x: torch.Tensor) -> torch.Tensor:
    x = self.encoder(x)
    return self.decoder.generate(x)


class AcousticModel_only_Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = Encoder(use_in=False)
    self.proj = nn.Linear(512, 128, bias=False)

  def forward(self, x: torch.Tensor, mels: torch.Tensor) -> torch.Tensor:
    x = self.encoder(x)
    return self.proj(x)

  @torch.inference_mode()
  def generate(self, x: torch.Tensor) -> torch.Tensor:
    x = self.encoder(x)
    return self.proj(x)


class AcousticModel_only_Decoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.upsample = nn.ConvTranspose1d(256, 512, 4, 2, 1)
    self.decoder = Decoder()

  def forward(self, x: torch.Tensor, mels: torch.Tensor) -> torch.Tensor:
    x = self.upsample(x.transpose(1, 2)).transpose(1, 2)
    x = self.decoder(x, mels)
    return x

  @torch.inference_mode()
  def generate(self, x: torch.Tensor) -> torch.Tensor:
    x = self.upsample(x.transpose(1, 2)).transpose(1, 2)
    x = self.decoder.generate(x)
    return x


class AcousticModel_tiny(nn.Module):
  def __init__(self):
    super().__init__()
    
    ''' Encoder '''
    self.encoder_prenet = nn.Sequential(
      nn.Linear(256, 256),
      nn.ReLU(),
    )
    self.convs = nn.Sequential(
      nn.Conv1d(256, 512, 9, 1, 4),  # if to keep same receptio field: (256, 512, 13, 1, 6)
      nn.ReLU(),
      nn.InstanceNorm1d(512),
      nn.ConvTranspose1d(512, 512, 4, 2, 1),  # must upsample x2 due to units-mels length match in dataloader
      nn.ReLU(),
      nn.InstanceNorm1d(512),
    )
    ''' Decoder '''
    self.decoder_prenet = nn.Sequential(
      nn.Linear(128, 256),
      nn.ReLU(),
    )
    self.lstm = nn.LSTM(512 + 256, 768, batch_first=True)
    self.proj = nn.Linear(768, 128, bias=False)

  def forward(self, x: torch.Tensor, mels: torch.Tensor) -> torch.Tensor:
    # encoder
    x = self.encoder_prenet(x)
    x = self.convs(x.transpose(1, 2))
    x = x.transpose(1, 2)
    # encoder
    mels = self.decoder_prenet(mels)
    x, _ = self.lstm(torch.cat((x, mels), dim=-1))
    x = self.proj(x)
    return x

  @torch.inference_mode()
  def generate(self, x: torch.Tensor) -> torch.Tensor:
    # encoder
    x = self.encoder_prenet(x)
    x = self.convs(x.transpose(1, 2))
    x = x.transpose(1, 2)

    # encoder
    xs = x
    m = torch.zeros(xs.size(0), 128, device=xs.device)
    h = torch.zeros(1, xs.size(0), 768, device=xs.device)
    c = torch.zeros(1, xs.size(0), 768, device=xs.device)

    mel = []
    for x in torch.unbind(xs, dim=1):
      m = self.decoder_prenet(m)
      x = torch.cat((x, m), dim=1).unsqueeze(1)
      x, (h, c) = self.lstm(x, (h, c))
      m = self.proj(x).squeeze(1)
      mel.append(m)
    return torch.stack(mel, dim=1)


class AcousticModel_tiny_half(nn.Module):
  def __init__(self):
    super().__init__()
    
    ''' Encoder '''
    self.encoder_prenet = nn.Sequential(
      nn.Linear(256, 128),
      nn.ReLU(),
    )
    self.convs = nn.Sequential(
      nn.Conv1d(128, 256, 9, 1, 4),  # if to keep same receptio field: (256, 512, 13, 1, 6)
      nn.ReLU(),
      nn.InstanceNorm1d(256),
      nn.ConvTranspose1d(256, 256, 4, 2, 1),  # must upsample x2 due to units-mels length match in dataloader
      nn.ReLU(),
      nn.InstanceNorm1d(256),
    )
    ''' Decoder '''
    self.decoder_prenet = nn.Sequential(
      nn.Linear(128, 256),
      nn.ReLU(),
    )
    self.lstm = nn.LSTM(256 + 256, 384, batch_first=True)
    self.proj = nn.Linear(384, 128, bias=False)

  def forward(self, x: torch.Tensor, mels: torch.Tensor) -> torch.Tensor:
    # encoder
    x = self.encoder_prenet(x)
    x = self.convs(x.transpose(1, 2))
    x = x.transpose(1, 2)
    # encoder
    mels = self.decoder_prenet(mels)
    x, _ = self.lstm(torch.cat((x, mels), dim=-1))
    x = self.proj(x)
    return x

  @torch.inference_mode()
  def generate(self, x: torch.Tensor) -> torch.Tensor:
    # encoder
    x = self.encoder_prenet(x)
    x = self.convs(x.transpose(1, 2))
    x = x.transpose(1, 2)

    # encoder
    xs = x
    m = torch.zeros(xs.size(0), 128, device=xs.device)
    h = torch.zeros(1, xs.size(0), 384, device=xs.device)
    c = torch.zeros(1, xs.size(0), 384, device=xs.device)

    mel = []
    for x in torch.unbind(xs, dim=1):
      m = self.decoder_prenet(m)
      x = torch.cat((x, m), dim=1).unsqueeze(1)
      x, (h, c) = self.lstm(x, (h, c))
      m = self.proj(x).squeeze(1)
      mel.append(m)
    return torch.stack(mel, dim=1)
