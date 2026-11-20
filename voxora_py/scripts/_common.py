import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RAW_PATH = ROOT / "data" / "raw" / "hintrain.parquet"
CORPUS_PATH = ROOT / "data" / "processed" / "dev_corpus.parquet"
EMBEDDINGS_PATH = ROOT / "data" / "processed" / "dev_embeddings.npy"
FAISS_PATH = ROOT / "data" / "indexes" / "dev.faiss"
BM25_PATH = ROOT / "data" / "indexes" / "dev_bm25.pkl"
EMBEDDINGS_EN_PATH = ROOT / "data" / "processed" / "dev_embeddings_en.npy"
FAISS_EN_PATH = ROOT / "data" / "indexes" / "dev_en.faiss"
