"""Pytest config. The `aics` package is installed editable via `pip install -e .`
so test code resolves it via site-packages directly — no sys.path mangling
needed.

We keep this file as a marker so pytest treats `tests/` as a package and
the warm-start `__init__.py` is unnecessary.
"""
