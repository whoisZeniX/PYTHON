from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    sarvam_api_key: str = ""
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    session_secret: str = ""

    host: str = "127.0.0.1"
    port: int = 8001

    corpus_path: Path = ROOT / "data" / "processed" / "dev_corpus.parquet"
    faiss_path: Path = ROOT / "data" / "indexes" / "dev.faiss"
    faiss_en_path: Path = ROOT / "data" / "indexes" / "dev_en.faiss"
    bm25_path: Path = ROOT / "data" / "indexes" / "dev_bm25.pkl"
    onnx_model_path: Path = ROOT / "onnx" / "model_int8.onnx"
    tokenizer_path: Path = ROOT / "onnx" / "tokenizer.json"
    css_path: Path = ROOT / "static" / "app.css"

    embed_max_tokens: int = 128

    search_top_k: int = 10
    final_top_k: int = 5
    min_dense_score: float = 0.30

    temperature: float = 0.2
    max_completion_tokens: int = 1024
    groq_timeout_s: float = 60.0
    groq_max_retries: int = 3

    sarvam_url: str = "https://api.sarvam.ai/speech-to-text"
    sarvam_model: str = "saaras:v4"
    sarvam_timeout_s: float = 60.0

    min_grounding_overlap: float = 0.10

    history_limit: int = 50
    latency_sample_size: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
