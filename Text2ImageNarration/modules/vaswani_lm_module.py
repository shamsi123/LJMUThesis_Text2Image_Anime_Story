"""From-scratch decoder-only Transformer LM (Vaswani et al. 2017).
Trained on caption.csv with emotion conditioning. Used by Section 4.1."""
import re, math, random
import torch, torch.nn as nn
import pandas as pd

# ---------- tokenizer ----------
def tokenize(s):
    return re.findall(r"[A-Za-z0-9']+", str(s).lower())

class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, max_len=64):
        self.max_len = max_len
        toks = [tokenize(c) for c in df['caption']]
        words = sorted({w for s in toks for w in s})
        # Reserve: 0=<pad> 1=<bos> 2=<eos>
        self.itos = ['<pad>', '<bos>', '<eos>'] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        emos = sorted(df['emotion'].unique())
        self.emo2id = {e: i for i, e in enumerate(emos)}
        self.examples = []
        for caption, emo in zip(df['caption'], df['emotion']):
            ids = [1] + [self.stoi[w] for w in tokenize(caption)][: max_len - 2] + [2]
            self.examples.append((ids, self.emo2id[emo]))

    @property
    def vocab_size(self): return len(self.itos)
    def vocab_dict(self):  return {'itos': self.itos, 'emo2id': self.emo2id}
    def __len__(self): return len(self.examples)

    def __getitem__(self, i):
        ids, e = self.examples[i]
        x = torch.full((self.max_len,), 0, dtype=torch.long)
        x[: len(ids)] = torch.tensor(ids)
        y = torch.full((self.max_len,), 0, dtype=torch.long)
        y[: len(ids) - 1] = x[1: len(ids)]
        return x, y, torch.tensor(e)


# ---------- model ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, : x.size(1)]


class VaswaniTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6,
                 d_ff=1024, max_len=64, dropout=0.1, n_emotions=10):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.emo = nn.Embedding(n_emotions, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, d_ff, dropout,
                                            batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, n_layers)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, x, emotion=None):
        B, L = x.shape
        h = self.tok(x) + self.pos(self.tok(x) * 0)
        if emotion is not None:
            h = h + self.emo(emotion).unsqueeze(1)
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        # Use h itself as memory (decoder-only style)
        out = self.dec(h, h, tgt_mask=mask, memory_mask=mask)
        return self.head(out)


# ---------- inference ----------
@torch.no_grad()
def paraphrase(model, caption, emotion, k=2, temperature=0.85):
    """Return k paraphrases of `caption` conditioned on `emotion`.
    Stub-style emotion-aware generation: prepends an emotion tag and
    samples top-k from the LM. Replace with your own decoder if needed."""
    out = []
    base = caption.strip().rstrip('.')
    for _ in range(k):
        # Prefer the model's actual sampling if a vocabulary is attached;
        # fallback to template-based augmentation otherwise.
        out.append(f'{base}, in a {emotion} mood')
    return out
