from dataclasses import dataclass
from typing import Optional, Union

import torch
from transformers import Wav2Vec2Processor


def _collate_text(features):
    """Stack the tokenized text fields of a batch (or return None if absent)."""
    if features[0]['texts'] is None:
        return None
    return {
        'ids': torch.stack([f['texts']['ids'] for f in features], dim=0),
        'mask': torch.stack([f['texts']['mask'] for f in features], dim=0),
        'token_type_ids': torch.stack([f['texts']['token_type_ids'] for f in features], dim=0),
    }


@dataclass
class DataCollatorCTCWithPadding:
    """Pad audio ``input_values`` with the processor and stack labels / text."""

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = torch.stack([f["labels"] for f in features], dim=0)
        text = _collate_text(features)
        if text is not None:
            batch["text"] = text
        return batch


@dataclass
class TXTDataCollatorWithPadding:
    """Text-only collator: stacks labels and tokenized text (no audio)."""

    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        batch = {"input_values": None}
        batch["labels"] = torch.stack([f["labels"] for f in features], dim=0)
        text = _collate_text(features)
        if text is not None:
            batch["text"] = text
        return batch
