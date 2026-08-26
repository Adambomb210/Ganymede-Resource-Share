"""A tokenizer stand-in, so masking and batching are testable without a download.

Everything in ``trainer/data.py`` that can go wrong -- the truncation rule, the
completion-only mask, the pad/ignore alignment in ``collate`` -- goes wrong
identically whatever the vocabulary is. Using a real tokenizer here would buy no
extra coverage and would cost a network round trip on every test run, which is
the difference between a suite people run and one they skip.

Word-level, with ids assigned in first-seen order, so a test can reason about
exact ids when it wants to.
"""

from __future__ import annotations


class FakeTokenizer:
    def __init__(self, bos_token_id: int | None = 1, eos_token_id: int | None = 2):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = 0
        self._vocab: dict[str, int] = {}
        self._next = 10  # leave 0-9 for the special ids above

    def _id(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = self._next
            self._next += 1
        return self._vocab[word]

    def __call__(self, text: str, add_special_tokens: bool = True, return_tensors=None):
        ids = [self._id(w) for w in text.split()]
        if add_special_tokens and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
        return {"input_ids": ids}
