
import soundfile
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
import torch
from torch import nn
import numpy as np
import tensorflow as tf

asrModel = "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"

d = ModelDownloader()
speech2text = Speech2Text(
    **d.download_and_unpack(asrModel),
    # Decoding parameters are not included in the model file
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=20,
    ctc_weight=0.3,
    lm_weight=0.5,
    penalty=0.0,
    nbest=1
)


class GANCritic(nn.Module):
    def __init__(self):
        super(GANCritic, self).__init__()
        self.frontend = speech2text.asr_model.frontend
        self.embed = speech2text.asr_model.encoder.embed
        self.encoder = speech2text.asr_model.encoder.encoders[:3]
        self.lstm1 = nn.LSTM(512, 100, 1, bidirectional=False, batch_first=True)
        self.lstm2 = nn.LSTM(100, 20, 1, bidirectional=False, batch_first=True)
        self.linear = nn.Linear(20, 1)
        self.sn = torch.nn.utils.spectral_norm(self.linear)
        #m = torch.nn.utils.parametrizations.spectral_norm(self.linear)
        #self.sn = torch.nn.utils.parametrizations.spectral_norm(self.lstm2)

    def forward(self, audio, fakeAudio, ilen, audioType):
      with torch.no_grad():
        if audioType == 'waveform':
          audio = self.frontend(audio, ilen)[0]

        z = self.embed(audio, x_mask=None)
        f = self.embed(fakeAudio, x_mask=None)
        z = self.encoder(z[0], z[1])
        f = self.encoder(f[0], f[1])
      z = self.lstm1(z[0])[0]
      f = self.lstm1(f[0])[0]
      z = self.lstm2(z)[0][:,-1,:]
      f = self.lstm2(f)[0][:,-1,:]
      z = self.linear(z)
      f = self.linear(f)
      z = nn.ReLU()(z)
      f = nn.ReLU()(f)
      return z, f

def GanLoss(enc, opt_enc, real, fake, batch_size):

    #torch.autograd.set_detect_anomaly(True)
    #torch.nn.utils.parametrizations.spectral_norm(enc.linear)
    real2 = torch.tensor(real.numpy()).double().to('cuda')
    fake2 = torch.tensor(fake.numpy()).double().to('cuda')

    #encReal = enc(real2, torch.tensor([real.shape[1]]*batch_size).double(), audioType='spec')
    #encFake = enc(fake2, torch.tensor([fake.shape[1]]*batch_size).double(), audioType='spec')
    encReal, encFake = enc(real2, fake2, torch.tensor([real.shape[1]]*batch_size).double(), audioType='spec')

    enc.zero_grad()
    output = -(torch.mean(encReal) - torch.mean(encFake))
    output.backward()
    genScore = torch.mean(encFake)
    opt_enc.step()
    return output.detach().cpu().numpy(), tf.convert_to_tensor(genScore.detach().cpu().numpy(), dtype=tf.float32)
