"""ESMRank: a learn-to-rank approach for protein variant effect prediction.

Typical use from code::

    import esmrank

    esmrank.ensure_models()                       # download model files if missing
    df = esmrank.rank_variants("protein.fasta", sub=True)
"""

__version__ = '0.1.1'

# Public API, imported lazily so that `import esmrank` stays cheap (torch,
# xgboost, pyhmmer, ... are only loaded when a pipeline step actually runs).
_LAZY = {
    'run_pipeline': 'pipeline',
    'rank_variants': 'pipeline',
    'ensure_models': 'models',
    'download_models': 'models',
    'find_model_dir': 'paths',
    'default_model_dir': 'paths',
}

__all__ = ['__version__', *_LAZY]


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module
        return getattr(import_module(f'.{_LAZY[name]}', __name__), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(list(globals()) + list(_LAZY))
