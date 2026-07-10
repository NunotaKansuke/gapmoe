try:
    from .galactic_jax import MappedGalacticModel
except ImportError:  # Compatibility with the pre-rename implementation.
    from .galactic_jax import JaxGalacticModel as MappedGalacticModel
