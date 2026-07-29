import random
import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

import re
chars_to_ignore_regex = r'[\,\?\.\!\-\;\:\"]'

def remove_special_characters(text):
    text = re.sub(chars_to_ignore_regex, '', text).upper()
    return text

class IEMOCAPLoader():
    def __init__(self, root_dir, fourclass=False, binary=None, speaker=False, speaker_id=False, avd=False):
        self.root_dir = root_dir

        if fourclass:
            self.label_list = {'neu': 0, 'hap': 1, 'ang': 2, 'sad': 3}
        else:
            self.label_list = {'neu': 0, 'hap': 1, 'ang': 2, 'sad': 3, 'fru': 4, 'exc': 5, 'sur': 6}
        
        self.num_labels = len(self.label_list)
        if avd: self.num_labels = 3
        if binary is not None:
            if binary == 'a':
                self.label_list = {'neu': 0, 'hap': 1, 'ang': 1, 'sad': 0}
            elif binary == 'v':
                self.label_list = {'neu': 0, 'hap': 0, 'ang': 1, 'sad': 1}
            else:
                assert False
            self.num_labels = 2

        
        # Get a list of all the session directories
        session_dirs = sorted(glob.glob(os.path.join(root_dir, 'Session*')))
        self.data = []
        
        # Collect every EmoEvaluation file (sorted, so the parse order — and with
        # it the seeded CV fold assignment below — is filesystem-independent).
        eval_files = []
        for session_dir in session_dirs:
            emo_eval_path = os.path.join(session_dir, 'dialog', 'EmoEvaluation')
            for emo_eval_file in sorted(glob.glob(os.path.join(emo_eval_path, '*.txt'))):
                eval_files.append((session_dir, emo_eval_file))

        # Loop over each EmoEvaluation file to get the utterance information
        for session_dir, emo_eval_file in tqdm(eval_files, desc='parsing IEMOCAP',
                                               unit='file', dynamic_ncols=True):
            transcript_path = os.path.join(session_dir, 'dialog', 'transcriptions')
            df = pd.read_csv(emo_eval_file, header=None, skiprows=1, delimiter='\t').dropna()
            t_df = pd.read_csv(os.path.join(transcript_path, emo_eval_file.split('/')[-1]), sep=':', header=None)
            t_df['ID'] = t_df.iloc[:,0].apply(lambda x: str(x).split(' ')[0])
            t_df['txt'] = t_df.iloc[:,1].apply(lambda x: str(x).strip())
            t_df = t_df[['ID', 'txt']].set_index('ID')

            for j, row in df.iterrows():
                # Get the utterance ID
                utterance_id = row[1]

                # Get the emotion label from the dataframe
                if row[2] in list(self.label_list):
                    emotion = row[2]
                elif row[2] == 'exc' and fourclass:
                    emotion = 'hap'
                else:
                    continue
                if avd:
                    emotion = (torch.tensor(eval(row[3]))-1)/4

                # Get the path to the waveform file
                wav_file = os.path.join(session_dir, 'sentences', 'wav', emo_eval_file.split('/')[-1][:-4], utterance_id + '.wav')
                transcript = remove_special_characters(t_df.loc[utterance_id][0])

                if avd:
                    emotion_label = (torch.tensor(emotion) - 3)/2
                else:
                    emotion_label = self.label_list.get(emotion, 0)
                    emotion_label = F.one_hot(torch.tensor([emotion_label]), num_classes=self.num_labels).float()

                # Append the data to the list
                self.data.append((wav_file, transcript, emotion_label))
        data_lst = self.data

        # data_lst = reduce(lambda acc, cur: acc + [cur + (aug,) for aug in aug_lst], data_lst, [])
        
        
        # data_lst = self.data
        np.random.seed(230)
        np.random.shuffle(data_lst)
        if speaker:
            spkr_dic = {}
            for row in data_lst:
                w,t,e = row
                spkr = w.split('/')[-1][3:5] + w.split('/')[-1].split('_')[-1][0:1]
                if spkr in spkr_dic.keys():
                    spkr_dic[spkr].append(row)
                else:
                    spkr_dic[spkr]=[row]
            self.splits = list(spkr_dic.values())
            if speaker_id:
                self.spkr_id = list(spkr_dic.keys())
        else:
            split_idx = [int(i*0.1 * len(data_lst)) for i in range(11)]
            self.splits = [data_lst[sp_idx:split_idx[i+1]] for i, sp_idx in enumerate(split_idx) if i != 10]
        self.data = data_lst
        



class MSPIMPROVLoader():
    def __init__(self, root_dir, binary=None, speaker=False, avd=False):
        self.root_dir = root_dir

        self.label_list = {'N': 0, 'H': 1, 'A': 2, 'S': 3}
        self.num_labels = len(self.label_list)
        if avd: self.num_labels = 3
        if binary is not None:
            if binary == 'a':
                self.label_list = {'N': 0, 'H': 1, 'A': 1, 'S': 0}
            elif binary == 'v':
                self.label_list = {'N': 0, 'H': 0, 'A': 1, 'S': 1}
            else:
                assert False
            self.num_labels = 2

        self.data = []
        
        emo_eval_path = os.path.join(root_dir, 'Evalution.txt')
        transcript_path =  os.path.join(root_dir, 'All_human_transcriptions')

        with open(emo_eval_path, 'r') as f:
            eval_file = f.readlines()
            eval_dic = {}
            for el in eval_file:
                if el.split(';')[0].endswith('.avi'):
                    eval_idx = el.split(';')[0].split('.')[0]
                    eval_idx = eval_idx.replace('UTD', 'MSP')
                    eval_dic[eval_idx] = el
        
        for eval_file in tqdm(eval_dic, desc='parsing MSP-IMPROV',
                              unit='utt', dynamic_ncols=True):

            #text
            with open(os.path.join(transcript_path, eval_file+'.txt')) as f:
                txt = f.read().rstrip('\n')
            transcript = remove_special_characters(txt)

            #label
            eval_str = eval_dic[eval_file]
            if avd:
                emotion = list(map(lambda x: eval(x.split(':')[1]), eval_str.split(';')[2:5]))
                emotion_label = (torch.tensor(emotion) - 1)/4
            else:
                label = eval_str.split(';')[1].strip()

                emotion_label = self.label_list.get(label, 0)
                emotion_label = F.one_hot(torch.tensor([emotion_label]), num_classes=self.num_labels).float()

            #wav
            tgt,spkr, scnr = eval_file.split('-')[2:5]
            wav_file = os.path.join(root_dir, 'Audio', f'session{spkr[-1]}', tgt, scnr, eval_file + '.wav')

            self.data.append((wav_file, transcript, emotion_label))




        data_lst = self.data

        np.random.seed(230)
        np.random.shuffle(data_lst)
        if speaker:
            spkr_dic = {}
            for row in data_lst:
                w,t,e = row
                spkr = w.split('/')[-1].split('-')[3]
                if spkr in spkr_dic.keys():
                    spkr_dic[spkr].append(row)
                else:
                    spkr_dic[spkr]=[row]
            self.splits = list(spkr_dic.values())
        else:
            split_idx = [int(i*0.1 * len(data_lst)) for i in range(11)]
            self.splits = [data_lst[sp_idx:split_idx[i+1]] for i, sp_idx in enumerate(split_idx) if i != 10]
        self.data = data_lst

     

class MSPPODCASTLoader():
    def __init__(self, root_dir, full=False ,binary=False, t=False, avd=False, min=None, max=None, multi=False, lim=None, bal=False):
        self.root_dir = root_dir

        self.label_list = {'N': 0, 'H': 1, 'A': 2, 'S': 3}
        if full:
            self.label_list = {'N': 0, 'H': 1, 'A': 2, 'S': 3, 'U': 4,'F': 5,'D': 6,'C': 7,}
        self.num_labels = len(self.label_list)
        if avd: self.num_labels = 3
        if binary:
            self.label_list = {'N': 0, 'H': 0, 'A': 1, 'S': 1}
            self.num_labels = 2

        self.data = []
        self.dev = []
        self.test = []
        self.train = []
        if bal: d = {0:[],1:[],2:[],3:[]};
        
        emo_eval_path = os.path.join(root_dir, 'Labels/labels.txt')
        split_path = os.path.join(root_dir, 'Partitions.txt')
        transcript_path =  os.path.join(root_dir, 'Transcripts')

        with open(emo_eval_path, 'r') as f:
            eval_file = f.readlines()
            eval_dic = {}
            for el in eval_file:
                if el.split(';')[0].endswith('.wav'):
                    eval_idx = el.split(';')[0].split('.')[0]
                    # eval_idx = eval_idx.replace('UTD', 'MSP')
                    eval_dic[eval_idx] = el

        with open(split_path, 'r') as f:
            split_file = f.readlines()
            split_dic = {}
            for sp in split_file:
                if ';' in sp:
                    if sp.split(';')[1].strip().endswith('.wav'):
                        split_idx = sp.split(';')[1].strip().split('.')[0]
                        # split_idx = split_idx.replace('UTD', 'MSP')
                        split_dic[split_idx] = sp.split(';')[0]
        chk = 0
        for eval_file in tqdm(eval_dic, desc='parsing MSP-PODCAST',
                              unit='utt', dynamic_ncols=True):

            #label (checked before the per-utterance transcript read, so skipped
            #utterances cost no file I/O)
            eval_str = eval_dic[eval_file]
            if avd:
                emotion = list(map(lambda x: eval(x.split(':')[1]), eval_str.split(';')[2:5]))
                emotion_label = (torch.tensor(emotion) - 1)/6
                if multi:
                    self.label_list = {'N': 0, 'H': 1, 'A': 2, 'S': 3, 'U': 4,'F': 5,'D': 6,'C': 7,}
                    label = eval_str.split(';')[1].strip()
                    if label in list(self.label_list):
                        pass
                    else:
                        continue
                    el = self.label_list.get(label, -1)
                    if el==-1: continue
                    el = F.one_hot(torch.tensor(el), num_classes=8).float()
                    emotion_label = torch.concat((emotion_label,el), dim=0)


            else:
                label = eval_str.split(';')[1].strip()

                if label in list(self.label_list):
                    pass
                else:
                    continue
                emotion_label = self.label_list.get(label, -1)
                if emotion_label==-1: continue
                if bal:
                    el = emotion_label
                    emotion_label = F.one_hot(torch.tensor([emotion_label]), num_classes=self.num_labels).float()
                else:
                    emotion_label = F.one_hot(torch.tensor([emotion_label]), num_classes=self.num_labels).float()

            #text
            with open(os.path.join(transcript_path, eval_file+'.txt')) as f:
                txt = f.read().rstrip('\n')
            transcript = remove_special_characters(txt)

            #wav
            # tgt,spkr, scnr = eval_file.split('-')[2:5]
            wav_file = os.path.join(root_dir, 'Audio', eval_file + '.wav')

            self.data.append((wav_file, transcript, emotion_label))

            if split_dic.get(eval_file, '') == 'Test1':
                self.test.append((wav_file, transcript, emotion_label))
            if split_dic.get(eval_file, '') == 'Development':
                self.dev.append((wav_file, transcript, emotion_label))
            if split_dic.get(eval_file, '') == 'Train':
                if min is not None:
                    if len(transcript.split(' ')) <= min:
                        continue
                if max is not None:
                    if len(transcript.split(' ')) >= max:
                        continue
                if bal:
                    d[el].append((wav_file, transcript, emotion_label))
                else:
                    self.train.append((wav_file, transcript, emotion_label))
                    chk+=1
                    assert len(self.train) == chk
        if lim is not None:
            self.train =  self.train[:lim]

        if bal:
            llll = 1e+9
            for e in d:
                lllllllll = len(d[e])
                if llll > lllllllll:
                    llll = lllllllll
            self.train = d[0][:llll] + d[1][:llll] + d[2][:llll] + d[3][:llll]
            np.random.shuffle(self.train) 
  