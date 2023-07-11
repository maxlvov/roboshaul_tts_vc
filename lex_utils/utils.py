import re
import requests
import json
from tqdm import notebook as tqdm
import time
from typing import List, Optional
heb_letters = "אבגדהוזחטיכךלמםנןסעפףצץקרשת"

def delete_diac(word: str) :
    return ''.join([l for l in word if (l in heb_letters or l in ['"',"'",'.',',','!','?'])])

def is_heb_word(word: str) :
    for l in word :
        if l not in heb_letters and l not in ['"',"'",'.',',','!','?'] :
            return False
    return True

def diac_text_with_dicta(text: str, return_with_punctuation: bool = True) :
    '''
    This function uses dicta API to diac a text. It will return a list of lists, each sublist contains all possible nbests for the current word.
    '''
    url = 'https://nakdan-2-0.loadbalancer.dicta.org.il/api'
    words = text.strip().split()
    punct = ". , ? !".split()
    has_punct = [w[-1] in punct  for w in words]
    payload = {
        "task": "nakdan",
        "genre": "modern",
        "data": text,
        "addmorph": True,
        "keepqq": False,
        "nodageshdefmem": False,
        "patachma": False,
        "keepmetagim": True,
    }
    headers = {
        'content-type': 'text/plain;charset=UTF-8'
    }
    try :
        r = requests.post(url, json=payload, headers=headers)
    except :
        return None, False
    all_transcribed = True
    if r.status_code == 200 :
        diac = [[w['options'][j][0] for j in range(len(w['options']))] for w in r.json() if 'options' in w and len(w['options'])>0]
        diac_wordset = set()
        if(len(diac)!=len(words)) :
            #print('Warning! number of diac words does not equal to the number of original words')
            all_transcribed = False
        elif return_with_punctuation :
            for j in range(len(words)) :
                if has_punct[j] :
                    #print('word',j,'has punct:',words[j][-1])
                    for k in range(len(diac[j])) :
                        diac[j][k]+=words[j][-1]
        return diac, all_transcribed
    else : #Problem
        return None, False

def prepare_diac_lexicon(wordset: set, sleep_interval_sec: float = 0.1, retry_times: int = 3, max_chunk: int = 1000, diac_nonheb_words: bool = False) :
    '''
    This function gets a word set, then using Dicta API gives a list of nbests for every word in the set.
    The output is a dictionary which maps every word to a list of its diac versions.
    '''

    heb_words = set([w for w in wordset if is_heb_word(w)])
    dic_word2diac = {}
    words_to_diac = (wordset if diac_nonheb_words else heb_words)
    oov = set()
    for tries in range(retry_times) :
        if all([w in dic_word2diac for w in words_to_diac]) :
            oov = set(wordset).difference(dic_word2diac)
            print('Done!')
            return (dic_word2diac, oov)
        print("Try",tries,"/",retry_times,sep=' ')
        chunk = []
        oov = sorted(words_to_diac.difference(dic_word2diac))
        for i,w in tqdm.tqdm(enumerate(oov), total=len(oov)) :
            if w in dic_word2diac and i<(len(oov)-1) :
                continue
            if w not in dic_word2diac :
                chunk.append(w)
            if len(chunk) % max_chunk == 0 or i==(len(oov)-1):
                txt = '\n'.join(chunk)
                time.sleep(sleep_interval_sec)
                diac, _ = diac_text_woth_dicta(txt)
                if diac is not None :
                    for nbests in diac :
                        if len(nbests)==0 :
                            continue
                        w_no_diac = delete_diac(nbests[0])
                        dic_word2diac[w_no_diac] = nbests
    oov = wordset.difference(dic_word2diac)
    print('Done!')
    return dic_word2diac, oov

        
        
def load_lex(path_to_lex: str) :
    dic_lex = dict()
    with open(path_to_lex,encoding='utf-8') as f :
        for line in f.read().splitlines() :
            word,pron = line.split('\t')
            if word not in dic_lex :
                dic_lex[word]=[pron]
            else :
                if pron not in dic_lex[word] :
                    dic_lex[word].append(pron)
    return dic_lex

#Find phonetic transcript for every word
import re
heb_letters = "אבגדהוזחטיכךלמםנןסעפףצץקשרת"
re_gimel = re.compile("ג([^"+heb_letters+"]*)'")
re_zain = re.compile("ז([^"+heb_letters+"]*)'")
re_tsadik = re.compile("[צץ]([^"+heb_letters+"]*)'")
re_tsadik_sofit = re.compile("ץ")
re_haf = re.compile("ך")
re_mem = re.compile("ם")
re_nun = re.compile("ן")
re_pey = re.compile("ף")
def transform_word(word: str) :
  '''
  This function is used for transforming diac words before inserting them into the G2P model
  '''
  word = re_gimel.sub("1"+r"\1",word)
  word = re_zain.sub("2"+r"\1",word)
  word = re_tsadik.sub("3"+r"\1",word)
  word = re_haf.sub("כ",word)
  word = re_mem.sub("מ",word)
  word = re_nun.sub("נ",word)
  word = re_pey.sub("פ",word)
  word = re_tsadik_sofit.sub("צ",word)
  word = list(word)
  return ' '.join(word)

punctutations_set = {",",".","?","!"}

def prepare_for_g2p(diac_words: List[str]) :
    punctutations_of_nbest = [(w[-1] if w[-1] in punctutations_set else '') for w in diac_words]
    words_without_punctuations = [(w[:-1] if w[-1] in punctutations_set else w) for w in diac_words]
    wordset_for_g2p = [transform_word(w) for w in words_without_punctuations]
    return wordset_for_g2p, punctutations_of_nbest


def prepare_input_to_tts(words: List[str],diac_words: List[str],g2p_prons: List[str], punctuations_of_words: List[str], manual_lex: dict) :
    assert(len(words)==len(diac_words))
    assert(len(words)==len(g2p_prons))
    assert(len(words)==len(punctuations_of_words))
    
    for i in range(len(words)) :
        if words[i] in manual_lex :
            g2p_prons[i] = manual_lex[words[i]]
        elif diac_words[i] in manual_lex :
            g2p_prons[i] = manual_lex[diac_words[i]]
        else :
            if '|' in diac_words[i] and len(g2p_prons[i])>3 and g2p_prons[i][:3]=='h a' :
                g2p_prons[i] = g2p_prons[i][2:] #Delete the first "h" when it is a prefix. This is from my ampirical observations, otherwise it gets a stress.
    #Add SIL after every punctuation, and before the utterance
    text_for_generator = '{SIL} '+' '.join(['{'+trans+'}'+punct+(' {SIL}' if punct!='' else '') for trans, punct in zip(g2p_prons, punctuations_of_words)])
    return text_for_generator