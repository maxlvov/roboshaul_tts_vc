# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write

import torch

import params
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN


import torchaudio


HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'


def main(file,checkpoint,timesteps,speaker_id,temperature,length_scale,hifigan) :

    if torch.cuda.is_available():
        device = 'cuda'
    else :
        device = 'cpu'

    HIFIGAN_CHECKPT = hifigan
    if not isinstance(speaker_id, type(None)):
        assert params.n_spks > 1, "Ensure you set right number of speakers in `params.py`."
        spk = torch.LongTensor([speaker_id]).to(device)
    else:
        spk = None
    
    print('Initializing Grad-TTS...')
    generator = GradTTS(len(symbols)+1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    generator.load_state_dict(torch.load(checkpoint, map_location=lambda loc, storage: loc)['model'])
    _ = generator.to(device).eval()
    print(f'Number of parameters: {generator.nparams}')
    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.to(device).eval()
    vocoder.remove_weight_norm()
    
    with open(file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    cmu = cmudict.CMUDict('./resources/cmu_dictionary')
    
    with torch.no_grad():
        for i, text in enumerate(texts):
            print(f'Synthesizing {i} text...', end=' ')
            x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).to(device)[None]
            x_lengths = torch.LongTensor([x.shape[-1]]).to(device)
            
            t = dt.datetime.now()
            y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=timesteps, temperature=temperature,
                                                   stoc=False, spk=spk, length_scale=length_scale)
            t = (dt.datetime.now() - t).total_seconds()
            print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')
            
            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            write(f'./out/sample_{i}.dec.wav', 22050, audio)
            audio = (vocoder.forward(y_enc).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            write(f'./out/sample_{i}.enc.wav', 22050, audio)




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help='path to a file with texts to synthesize')
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to a checkpoint of Grad-TTS')
    parser.add_argument('-t', '--timesteps', type=int, required=False, default=10, help='number of timesteps of reverse diffusion')
    parser.add_argument('-s', '--speaker_id', type=int, required=False, default=None, help='speaker id for multispeaker model')
    parser.add_argument('-temp', '--temperature', type=float, required=False, default=1.5, help='temperature for diffusion model')
    parser.add_argument('-l', '--length_scale', type=float, required=False, default=0.91, help='To slow down, increase this number')
    parser.add_argument('-hf', '--hifigan', type=str, required=False, default=HIFIGAN_CHECKPT, help='Path to a checkpoint of HiFi-GAN')
    args = parser.parse_args()
    main(args.file,args.checkpoint,args.timesteps,args.speaker_id,args.temperature,args.length_scale,args.hifigan)



    #print('Done. Check out `out` folder for samples.')
