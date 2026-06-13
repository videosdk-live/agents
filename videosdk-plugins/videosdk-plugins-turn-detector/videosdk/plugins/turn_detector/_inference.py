from __future__ import annotations

from typing import Iterable

import numpy as np
from tokenizers import Tokenizer


def make_session_options(*, cpu_mem_arena: bool = True, mem_pattern: bool = True):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.enable_cpu_mem_arena = cpu_mem_arena
    so.enable_mem_pattern = mem_pattern
    return so


def load_tokenizer(
    path: str,
    *,
    max_length: int | None = None,
    pad_to_max: bool = False,
) -> Tokenizer:
    tok = Tokenizer.from_file(path)
    if max_length is not None:
        if tok.truncation is None:
            tok.enable_truncation(max_length=max_length)
        if pad_to_max and tok.padding is None:
            tok.enable_padding(length=max_length)
    return tok


def encode_for_model(
    tokenizer: Tokenizer,
    text: str,
    model_input_names: Iterable[str],
) -> dict[str, np.ndarray]:
    names = set(model_input_names)
    enc = tokenizer.encode(text)
    feed: dict[str, np.ndarray] = {}
    if "input_ids" in names:
        feed["input_ids"] = np.asarray([enc.ids], dtype=np.int64)
    if "attention_mask" in names:
        feed["attention_mask"] = np.asarray([enc.attention_mask], dtype=np.int64)
    if "token_type_ids" in names:
        feed["token_type_ids"] = np.asarray([enc.type_ids], dtype=np.int64)
    return feed


def model_input_names(session) -> list[str]:
    return [i.name for i in session.get_inputs()]
