"""The main module for moneyball."""

try:
    import manhole

    manhole.install(verbose=False)
except ImportError:
    pass

__VERSION__ = "0.0.10"
