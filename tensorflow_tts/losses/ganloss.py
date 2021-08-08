
import soundfile
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
import torch
from torch import nn
import numpy as np

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
        m = torch.nn.utils.parametrizations.spectral_norm(self.linear)
        #self.sn = torch.nn.utils.parametrizations.spectral_norm(self.lstm2)

    def forward(self, audio, ilen, audioType):
      with torch.no_grad():
        if audioType == 'waveform':
          audio = self.frontend(audio, ilen)[0]

        z = self.embed(audio, x_mask=None)
        z = self.encoder(z[0], z[1])
      z = self.lstm1(z[0])[0]
      z = self.lstm2(z)[0][:,-1,:]
      z = self.linear(z)
      z = nn.ReLU()(z)
      return z

def GanLoss(enc, opt_enc, real, fake, batch_size):

    real = torch.tensor(real.numpy())
    fake = torch.tensor(fake.numpy())

    encReal = enc(real.double(), torch.tensor([real.shape[1]]*batch_size).double(), audioType='spec')
    encFake = enc(fake.double(), torch.tensor([fake.shape[1]]*batch_size).double(), audioType='spec')

    enc.zero_grad()
    output = -(torch.mean(encReal) - torch.mean(encFake))
    output.backward()
    opt_enc.step()
    return output
