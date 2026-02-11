"""Configuration defaults. All overridable via constructor arguments or CLI flags."""

# vllm-mlx server endpoint
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_API_KEY = "not-needed"

# Model identifier. vllm-mlx uses "default" for the loaded model.
DEFAULT_MODEL = "default"

# Generation defaults
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7

# Composition uses lower temperature (tracing necessity, not exploring)
COMPOSITION_TEMPERATURE = 0.3

# Number of independent collapses
DEFAULT_NUM_COLLAPSES = 3
