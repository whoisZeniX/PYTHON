import argparse

import faiss
import numpy as np

from _common import EMBEDDINGS_PATH, FAISS_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", default=str(EMBEDDINGS_PATH))
    parser.add_argument("--output", default=str(FAISS_PATH))
    args = parser.parse_args()

    embeddings = np.load(args.embeddings).astype(np.float32)
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise SystemExit(f"Bad embeddings array: shape {embeddings.shape}")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, args.output)
    print(f"FAISS IndexFlatIP: {index.ntotal} x {index.d} -> {args.output}")


if __name__ == "__main__":
    main()
