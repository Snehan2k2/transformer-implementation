# Transformer from Scratch

A from-scratch PyTorch implementation of the Transformer ("Attention Is All You Need"), built snippet by snippet while working through [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/).

## What's in here

`test.ipynb` — the full model, training loop, and data pipeline, built up one piece at a time.

## Architecture

- **Embeddings** — turn token IDs into vectors, scaled by `sqrt(d_model)`.
- **Positional encoding** — adds sin/cos position info to embeddings, since attention has no built-in sense of order.
- **Multi-head attention** — lets every token look at every other token. Splits the vector into multiple heads so each can learn different relationships, in parallel.
- **Feed-forward layers** — transform each token's vector independently, after attention has mixed information across tokens.
- **Encoder** — stack of N layers, each with self-attention + feed-forward. Encodes the full source sentence at once.
- **Decoder** — stack of N layers, each with masked self-attention (can't look ahead), cross-attention (looks at the encoder's output), and feed-forward. Generates the target sentence.
- **Residual connections + LayerNorm** — wrap every sub-layer, so gradients can flow cleanly through deep stacks.

## Training

- **Label smoothing** — softens the training target so the model isn't pushed toward 100% overconfidence.
- **Custom learning rate schedule** — ramps up for `warmup` steps, then decays.
- **Masking** — padding mask (ignore filler tokens) and causal mask (decoder can't see future tokens), combined.
- Trained first on a toy copy task (sanity check), then on real German→English sentence pairs (Multi30k).

## Environment

- Python 3.10 (via `uv`) — pinned because `torch==1.11.0` and `torchtext==0.12` have no macOS wheels for 3.11+.
- Key packages: `torch`, `torchtext`, `torchdata`, `spacy` (+ German/English models), `pandas`, `altair`, `jupyter`.

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Running it

1. `example_simple_model()` — quick sanity check on the toy copy task.
2. `load_tokenizers()` + `load_vocab()` — build vocab from Multi30k (downloads data, takes a few minutes first run).
3. `train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)` — train on real data (slow on CPU).
4. `check_outputs(...)` — see the trained model's translations next to the ground truth.
