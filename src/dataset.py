import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.data import Dataset


class Dataset_aug(Dataset):
    """Audio + (optional) text dataset.

    Each example is a ``(wav_file, transcript, emotion_label)`` tuple. The waveform
    is loaded and resampled to 16 kHz; the transcript is tokenized when a tokenizer
    has been set. Audio feature extraction/padding is handled by the collator.
    """

    def __init__(self, data, max_len_t=200):
        self.data = data
        self.max_len_t = max_len_t
        self.tokenizer = None

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        wav_file, transcript, emotion_label = self.data[index]

        waveform, sample_rate = torchaudio.load(wav_file)
        if sample_rate != 16000:
            waveform = F.resample(waveform, sample_rate, 16000)
            sample_rate = 16000
        assert waveform.shape[0] == 1
        waveform = waveform.squeeze(0)

        if self.tokenizer is not None:
            inputs = self.tokenizer.encode_plus(
                transcript,
                None,
                add_special_tokens=True,
                max_length=self.max_len_t,
                pad_to_max_length=True,
                return_token_type_ids=True,
                truncation=True,
            )
            transcripts = {
                'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
                'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
                'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            }
        else:
            transcripts = None

        return {"input_values": waveform, 'texts': transcripts, "labels": emotion_label}


class TXTDataset_aug(Dataset):
    """Text-only dataset.

    Each example is a ``(wav_file, transcript, emotion_label)`` tuple; the waveform
    path is ignored and only the tokenized transcript is returned.
    """

    def __init__(self, data, max_len_t=200):
        self.data = data
        self.max_len_t = max_len_t
        self.tokenizer = None

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        wav_file, transcript, emotion_label = self.data[index]

        inputs = self.tokenizer.encode_plus(
            transcript,
            None,
            add_special_tokens=True,
            max_length=self.max_len_t,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
        )
        transcript = {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
        }

        return {"input_values": None, 'texts': transcript, "labels": emotion_label}
