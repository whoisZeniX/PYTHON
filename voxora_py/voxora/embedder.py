from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

DIMENSION = 384


class E5Embedder:
    def __init__(self, model_path: Path, tokenizer_path: Path, max_length: int = 128):
        self.max_length = max_length

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.enable_cpu_mem_arena = False

        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.input_names = {item.name for item in self.session.get_inputs()}

    def _tokenize(self, texts: list[str]) -> dict[str, np.ndarray]:
        size = self.max_length
        ids, masks, types = [], [], []

        for item in self.tokenizer.encode_batch(texts):
            item_ids = item.ids[:size]
            item_mask = item.attention_mask[:size]
            item_types = (item.type_ids or [0] * len(item_ids))[:size]
            pad = size - len(item_ids)
            ids.append(item_ids + [0] * pad)
            masks.append(item_mask + [0] * pad)
            types.append(item_types + [0] * (size - len(item_types)))

        inputs = {
            "input_ids": np.asarray(ids, dtype=np.int64),
            "attention_mask": np.asarray(masks, dtype=np.int64),
        }
        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = np.asarray(types, dtype=np.int64)
        return inputs

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, DIMENSION), dtype=np.float32)

        inputs = self._tokenize(texts)
        tokens = self.session.run(None, inputs)[0]

        mask = inputs["attention_mask"][..., None].astype(np.float32)
        pooled = (tokens * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norms, 1e-12, None)).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed(["query: " + text])

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self.embed(["passage: " + text for text in texts])
