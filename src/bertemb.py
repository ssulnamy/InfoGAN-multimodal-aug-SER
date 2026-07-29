import torch.nn as nn


class ClassificationHead(nn.Module):
    r"""Linear classification head over the pooled text embedding."""

    def __init__(self, config):
        super().__init__()
        self.out_proj = nn.Linear(1024, config.num_labels)

    def forward(self, features, **kwargs):
        return self.out_proj(features)


class BERTEnc(nn.Module):
    """Text encoder: a pretrained language model + a classification head.

    ``forward`` returns ``(embedding, logits)``. The embedding is a projection of
    the final layer's pooled hidden state — the last token for causal LMs
    (``pool="last"``) or the [CLS] token for bidirectional encoders (``pool="cls"``).
    """

    def __init__(self, model_t, config_t, freeze=False, pool="last"):
        super().__init__()
        if freeze:
            model_t.eval()
        self.model = model_t
        self.cls = ClassificationHead(config=config_t)
        self.num_labels = config_t.num_labels
        self.pool = pool
        self.prof = nn.Linear(config_t.hidden_size, 1024)

    def forward(self, data, device):
        text_input = data['text']
        ids = text_input['ids'].to(device)
        mask = text_input['mask'].to(device)
        token_type_ids = text_input['token_type_ids'].to(device)

        out = self.model(ids, attention_mask=mask, token_type_ids=token_type_ids,
                         output_hidden_states=True, use_cache=False)

        hidden = out.hidden_states[-1]
        pooled = hidden[:, 0, :] if self.pool == "cls" else hidden[:, -1, :]
        emb = self.prof(pooled)
        yhat = self.cls(emb)
        return emb, yhat
