# -*- coding: utf-8 -*-
"""
This is a demo for RoboShaul Voice Conversion system, based on soft-vc.

First, git clone the repo to the current computer.
"""

"""Import all modules"""

import os
import sys
import torch, torchaudio
sys.path.append('vc-acoustic-model-main/acoustic-model-main')
from acoustic.model import AcousticModel
sys.path.append('hifigan/hifigan-main')
from hifigan.generator import HifiganGenerator
import scipy
import argparse
parser = argparse.ArgumentParser(description='Process some inputs.')

# Add arguments to the parser
parser.add_argument('path_to_models_dirs', type=str, help='Path to the directory containing the models.')
parser.add_argument('path_to_input_wave_file', type=str, help='Path to the input wave file to be voice-converted.')
parser.add_argument('path_to_output_wave_file', type=str, help='Path to the output wave file.')


args = parser.parse_args()
path_to_models_dirs = args.path_to_models_dirs
path_to_input_wave_file = args.path_to_input_wave_file
path_to_output_wave_file = args.path_to_output_wave_file


path_to_acoustic_model_vc = os.path.join(path_to_models_dirs,'VC','model-best.pt')
path_to_hifigan = os.path.join(path_to_models_dirs,'HifiGAN','hifigan-hubert-discrete-bbad3043.pt')

for fil in [path_to_acoustic_model_vc, path_to_hifigan] :
    if not os.path.exists(fil) :
        print('File',fil,'does not exist!')
        sys.exit(1)
        
if torch.cuda.is_available() :
  device = 'cuda'
else :
  device = 'cpu'

"""Load the models"""

#Load hubert
hubert = torch.hub.load("bshall/hubert:main", "hubert_soft").to(device)

#Load acoustic model
acoustic = AcousticModel()
state_dic_am = torch.load(path_to_acoustic_model_vc, map_location=torch.device('cpu'))
acoustic.load_state_dict(state_dic_am['acoustic-model'])
acoustic.to(device)

#Load HiFi-GAN model for voice conversion
hifigan = torch.load(path_to_hifigan, map_location=torch.device(device))

#Load audio
source, sr = torchaudio.load(path_to_input_wave_file)
source = torchaudio.functional.resample(source, sr, 16000).to(device)
source = source.unsqueeze(0)

#Convert to high quality audio
with torch.inference_mode():
    # Extract speech units
    units = hubert.units(source)
    # Generate target spectrogram
    mel = acoustic.generate(units).transpose(1, 2)
    # Generate audio waveform
    target = hifigan(mel)


"""Save the audio"""

scipy.io.wavfile.write(path_to_output_wave_file, 16000, target.squeeze().numpy())
print('The example file is saved to:',path_to_output_wave_file)
