import argparse
from pathlib import Path

import pyarrow.parquet as pq

from _common import BM25_PATH, CORPUS_PATH
from voxora.retrieval import BM25Index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(CORPUS_PATH))
    parser.add_argument("--output", default=str(BM25_PATH))
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    texts = [str(t or "") for t in pq.read_table(args.input, columns=["text"])["text"].to_pylist()]
    index = BM25Index.build(texts, k1=args.k1, b=args.b)
    index.save(Path(args.output))
    print(f"BM25Okapi over {index.corpus_size} documents (k1 {index.k1}, b {index.b}) -> {args.output}")


if __name__ == "__main__":
    main()
