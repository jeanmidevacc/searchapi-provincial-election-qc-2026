"""Embed ChatGPT answers with a multilingual sentence-embedding model, cached on disk.

Model: ibm-granite/granite-embedding-97m-multilingual-r2 -- 97M params, EN+FR,
32K-token context. Picked over MiniLM (128-token limit truncated 99% of these
~550-token answers), EmbeddingGemma-300m and granite-311m on a CPU benchmark of
the day-1 answers: a (party, issue)'s EN answer finds its own FR answer as
nearest neighbour among all 208 FR answers 92% of the time (MiniLM 66%,
EmbeddingGemma 83%, granite-311m 95% at 4x the CPU time).

Embedding ~400 answers takes a few minutes on CPU, so vectors are cached in
data/cache/, keyed by a hash of the cleaned text -- a rebuild only embeds
answers it hasn't seen.
"""
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAME = "ibm-granite/granite-embedding-97m-multilingual-r2"
MAX_TOKENS = 2048  # longest answer is ~1.4K tokens; 32K supported, but no need to pad attention that far
# float32 is part of the cache identity: transformers >= 5 loads this
# checkpoint in bfloat16 by default, whose vectors are only unit-length to
# ~0.5% -- enough noise to swamp the ~0.01-0.07 day-to-day distances shown.
CACHE_PATH = Path(__file__).resolve().parent / "data" / "cache" / "chatgpt_embeddings_granite97m_fp32.parquet"

# ChatGPT closes almost every answer with a near-identical offer ("Si vous
# voulez, je peux aussi vous faire un comparatif..."), which would pull every
# answer towards the same point regardless of content.
UPSELL = re.compile(
    r"^\s*(si vous (le )?(voulez|souhaitez)|si tu veux|je peux aussi|voulez-vous|souhaitez-vous|"
    r"if you('d| would)? (like|want)|i can also|would you like|want me to)",
    re.I,
)


def clean_answer(markdown: str) -> str:
    paragraphs = [p for p in re.split(r"\n\s*\n", markdown) if p.strip()]
    while paragraphs and UPSELL.match(re.sub(r"[*_#>]", "", paragraphs[-1])):
        paragraphs.pop()
    text = "\n\n".join(paragraphs)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_#>`|]", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _key(text: str) -> str:
    return hashlib.sha1(f"{MODEL_NAME}\n{text}".encode("utf-8")).hexdigest()


def embed_answers(answers: pd.Series) -> np.ndarray:
    """L2-normalised embeddings, one row per answer (same order as `answers`)."""
    texts = [clean_answer(a) for a in answers]
    keys = [_key(t) for t in texts]

    cache = pd.read_parquet(CACHE_PATH) if CACHE_PATH.exists() else pd.DataFrame(columns=["key", "emb"])
    known = dict(zip(cache["key"], cache["emb"]))
    missing = {k: t for k, t in zip(keys, texts) if k not in known}

    if missing:
        import torch  # heavy imports, only when needed
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME, model_kwargs={"dtype": torch.float32})
        model.max_seq_length = MAX_TOKENS
        print(f"  embedding {len(missing)} new ChatGPT answers with {MODEL_NAME} ...", flush=True)
        vectors = model.encode(list(missing.values()), batch_size=8, normalize_embeddings=True)
        known.update(zip(missing.keys(), vectors))
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"key": list(known), "emb": [np.asarray(v) for v in known.values()]}).to_parquet(CACHE_PATH)

    return np.vstack([np.asarray(known[k], dtype=float) for k in keys])
