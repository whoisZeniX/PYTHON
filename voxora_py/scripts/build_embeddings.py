import argparse
import time

import numpy as np
import pyarrow.parquet as pq

from _common import CORPUS_PATH, EMBEDDINGS_PATH
from voxora.config import get_settings
from voxora.embedder import E5Embedder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(CORPUS_PATH))
    parser.add_argument("--output", default=str(EMBEDDINGS_PATH))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--column", default="text", help="corpus column to embed (text or english_text)")
    args = parser.parse_args()

    settings = get_settings()
    texts = [str(t or "") for t in pq.read_table(args.input, columns=[args.column])[args.column].to_pylist()]
    embedder = E5Embedder(settings.onnx_model_path, settings.tokenizer_path, settings.embed_max_tokens)

    start = time.time()
    batches = []
    for offset in range(0, len(texts), args.batch_size):
        batches.append(embedder.embed_passages(texts[offset : offset + args.batch_size]))
        print(f"\rEmbedded {min(offset + args.batch_size, len(texts))}/{len(texts)}", end="", flush=True)
    embeddings = np.vstack(batches).astype(np.float32)
    print()

    norms = np.linalg.norm(embeddings, axis=1)
    if not np.isfinite(embeddings).all() or not np.allclose(norms, 1.0, atol=1e-3):
        raise SystemExit("Embeddings are not finite and unit-length.")
    np.save(args.output, embeddings)
    print(f"Saved {embeddings.shape} to {args.output} in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
