"""The Ganymede trainer: what a worker actually runs (docs/02-architecture-v2.md 4.3).

The package is deliberately importable in pieces. ``data`` is pure Python plus a
tokenizer and has no model dependency, so bucketing and masking are testable in
milliseconds; ``model`` and ``train`` pull in `transformers` and `peft`.
"""
