import argparse
import time

import pyarrow as pa
import pyarrow.parquet as pq

from _common import CORPUS_PATH, RAW_PATH

SCHEMA = pa.schema(
    [
        ("document_id", pa.string()),
        ("text", pa.string()),
        ("language", pa.string()),
        ("query_id", pa.string()),
        ("passage_index", pa.string()),
        ("is_selected", pa.string()),
        ("source", pa.string()),
        ("english_text", pa.string()),
        ("query", pa.string()),
        ("query_type", pa.string()),
        ("source_lang", pa.string()),
        ("target_lang", pa.string()),
    ]
)


def clean(value) -> str:
    return " ".join(str(value or "").split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(RAW_PATH))
    parser.add_argument("--output", default=str(CORPUS_PATH))
    parser.add_argument("--max-records", type=int, default=100, help="source query rows to read")
    parser.add_argument("--max-documents", type=int, default=1000)
    args = parser.parse_args()

    start = time.time()
    rows = pq.read_table(args.input).slice(0, args.max_records).to_pylist()

    documents = []
    for row in rows:
        passages = row.get("passages") or {}
        translated = passages.get("Translated_passages") or []
        english = passages.get("English_passages") or []
        selected = passages.get("is_selected") or []
        target_lang = str(row.get("target_lang") or "hi")

        for index, raw in enumerate(translated):
            text = clean(raw)
            if not text:
                continue
            documents.append(
                {
                    "document_id": f"{row['query_id']}_{index}",
                    "text": text,
                    "language": target_lang,
                    "query_id": str(row["query_id"]),
                    "passage_index": str(index),
                    "is_selected": str(int(selected[index]) if index < len(selected) else 0),
                    "source": "ai4bharat/MSMARCO-XI",
                    "english_text": clean(english[index]) if index < len(english) else "",
                    "query": clean(row.get("query")),
                    "query_type": str(row.get("query_type") or ""),
                    "source_lang": str(row.get("source_lang") or "en"),
                    "target_lang": target_lang,
                }
            )
    documents = documents[: args.max_documents]
    if not documents:
        raise SystemExit("No documents were extracted.")

    pq.write_table(pa.Table.from_pylist(documents, schema=SCHEMA), args.output)
    print(f"Source rows: {len(rows)}  documents written: {len(documents)}  -> {args.output}")
    print(f"Time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
