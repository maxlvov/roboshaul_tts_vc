# -*- coding: utf-8 -*-
"""
This is a demo for RoboShaul TTS system.
It is based on:
1. Diacritization using DICTA API
2. Grapheme to phoneme conversion using an Open-NMT model or manual lexicon.
3. TTS using a Grad-TTS model.

First, git clone the repo to the current computer.
"""
import os
import sys
import lex_utils.utils as lex_utils
import torch, torchaudio
import scipy
import argparse
import subprocess
import shutil
import importlib  
sys.path.append("Grad-TTS")
sys.path.append('Grad-TTS/hifi-gan/')
Grad_TTS = importlib.import_module("Grad-TTS.inference")

import argparse
parser = argparse.ArgumentParser(description='Process some inputs.')

# Add arguments to the parser
parser.add_argument('--temp_speed', type=float, default=0.95, help='Argument for tempo speed of audio. Higher means slower speech tempo.')
parser.add_argument('--timesteps', type=int, default=50, help='Number of timesteps in reverse diffusion.')
parser.add_argument('--temperature', type=float, default=1.5, help='Temperature for diffusion model.')
parser.add_argument('path_to_models_dirs', type=str, help='Path to the directory containing the models.')
parser.add_argument('path_to_textual_file', type=str, help='Path to the input textual file with a text for TTS.')
parser.add_argument('path_to_output_wave_file', type=str, help='Path to the output wave file.')


args = parser.parse_args()
temp_speed = args.temp_speed
timesteps = args.timesteps
temperature = args.temperature
path_to_models_dirs = args.path_to_models_dirs
path_to_textual_file = args.path_to_textual_file
path_to_output_wave_file = args.path_to_output_wave_file



path_to_g2p_model = os.path.join(path_to_models_dirs,'G2P','model_step_10000.pt')
path_to_tts_model = os.path.join(path_to_models_dirs,'TTS','grad_tts.pt')
path_to_hifigan_tts = os.path.join(path_to_models_dirs,'HiFiGAN_for_GradTTS','hifigan.pt')

for fil in [path_to_g2p_model, path_to_tts_model, path_to_hifigan_tts] :
    if not os.path.exists(fil) :
        print('File',fil,'does not exist!')
        sys.exit(1)
        
if torch.cuda.is_available() :
  device = 'cuda'
else :
  device = 'cpu'

"""Read the textual file."""
with open(path_to_textual_file, encoding='utf-8') as f:
    text = ' '.join(f.read().splitlines()).strip()

words = text.strip().split()
punctuations_set = {",",".","?","!"}
punctuations_of_words = [(w[-1] if w[-1] in punctuations_set else '') for w in words]
words_without_punctuations = [(w[:-1] if w[-1] in punctuations_set else w) for w in words]

"""Diacritize words using the Dicta API. If the API does not work, make sure you have full word coverage in the manual lexicon, located in roboshaul/lex_utils/lex_for_tts.txt"""

diac_text, oov = lex_utils.diac_text_with_dicta(text)
if (diac_text is None) or len(words)!=len(diac_text) :
  dicta = False
  print('Warning! Dicta diacritization API does not work. To continue, you can use manual lexicon. Make sure all words appear in the lexicon, in path: lex_utils/lex_for_tts.txt')
else :
  dicta = True

"""Take the first nbest from the Dicta API"""

if dicta :
  diac_text_best = [nbest[0] for nbest in diac_text]
  diac_words_without_punctuations = [(w[:-1] if w[-1] in punctuations_set else w) for w in diac_text_best]
  print('Diacritized text:')
  print(' '.join(diac_text_best))

"""Prepare words for G2P translation"""

if dicta :
  wordlist_for_g2p, punctuations_of_words = lex_utils.prepare_for_g2p(diac_text_best)
  os.makedirs('temp',exist_ok=True)
  path_to_words_for_g2p = os.path.join('temp','words_for_g2p.txt')
  fout = open(path_to_words_for_g2p,'w',encoding='utf-8')
  for w in wordlist_for_g2p :
    print(w,file=fout)
  fout.close()

"""Convert words to phonemes using G2P"""

#Convert to phonems
path_to_output_phones_file = os.path.join("temp","words_for_g2p.phones.txt")
subprocess.run(["onmt_translate", "-model", path_to_g2p_model, "-src", path_to_words_for_g2p, "-output", path_to_output_phones_file])
#!onmt_translate -model $path_to_g2p_model -src $path_to_words_for_g2p -output $path_to_output_phones_file

"""Load manual lexicon"""

#Load manual lexicon
lexicon = lex_utils.load_lex(os.path.join('lex_utils','lex_for_tts.txt'))
for word in lexicon :
  if len(lexicon[word])>1 :
    print('Warning! More than one phonetic pronunciation for the word '+word+'. Taking only the first:')
    for pron in lexicon[word] :
      print(pron)
  lexicon[word] = lexicon[word][0]

"""Prepare text for TTS"""

if dicta :
  with open(path_to_output_phones_file,encoding='utf-8') as f:
      phones = f.read().splitlines()
  assert(len(phones)==len(diac_text_best))
else :
  assert(all([(w in lexicon) for w in words_without_punctuations]))
  phones = [lexicon[w] for w in words_without_punctuations]
  diac_words_without_punctuations = words_without_punctuations
text_for_tts = lex_utils.prepare_input_to_tts(words_without_punctuations, diac_words_without_punctuations, phones, punctuations_of_words, lexicon)
path_to_text_for_tts = os.path.join("temp","text_for_tts.txt")
fout = open(path_to_text_for_tts,'w',encoding='utf-8')
print(text_for_tts,file=fout)
fout.close()

"""Run TTS"""

#Create initial audio
os.chdir('Grad-TTS')
path_to_output_text=os.path.join("..",path_to_text_for_tts)
os.makedirs('out',exist_ok=True)
path_to_checkpoint = os.path.join('..',path_to_tts_model)
Grad_TTS.main(path_to_output_text, path_to_checkpoint,timesteps,None,temperature,temp_speed,path_to_hifigan_tts)
os.chdir('..')

path_to_tts_output_wave = os.path.join('Grad-TTS','out','sample_0.dec.wav')
shutil.copy(path_to_tts_output_wave,path_to_output_wave_file)
print('The output file is saved to:',path_to_output_wave_file)
#Audio(path_to_tts_output_wave,rate=22050)
