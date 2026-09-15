"""Frozen offline compressors that produce QuRO's cacheable document latents.

Every compressor here satisfies the contract declared in ``src/offline.py``:
``encode_texts(list[str]) -> Tensor[batch, m, h]``.  QuRO never trains them and
never imports their internals outside this package.
"""
