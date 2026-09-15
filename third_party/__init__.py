"""Vendored third-party research code, unmodified.

``modelling_pisco.py``  -- copied from https://huggingface.co/naver/pisco-mistral
``modeling_cocom.py``   -- copied from https://huggingface.co/naver/cocom-v1-16-mistral-7b

They are vendored rather than loaded via ``trust_remote_code`` so that QuRO can
run offline and so that the exact revision used for an experiment is recorded in
this repository.  Do not edit them: QuRO adapts to their API in
``src/compressors/``, never the other way round.
"""
