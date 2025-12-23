"""Integration test fixtures and configuration."""

import os
import pytest
from pathlib import Path

# Load .env file for integration tests
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


def get_test_model():
    """Get the model name from environment."""
    model = os.environ.get("OPENAI_MODEL")
    if not model:
        raise ValueError("OPENAI_MODEL not set. Please configure .env file.")
    return model


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "requires_openai: marks tests that require OpenAI API key"
    )
    config.addinivalue_line(
        "markers", "requires_langfuse: marks tests that require Langfuse credentials"
    )


@pytest.fixture
def has_openai_key():
    """Check if OpenAI API key is available."""
    return bool(os.environ.get("OPENAI_API_KEY"))


@pytest.fixture
def has_langfuse_keys():
    """Check if Langfuse credentials are available."""
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
    )


def pytest_runtest_setup(item):
    """Skip tests based on required credentials."""
    # Check for requires_openai marker
    if item.get_closest_marker("requires_openai"):
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("Requires OPENAI_API_KEY environment variable")

    # Check for requires_langfuse marker
    if item.get_closest_marker("requires_langfuse"):
        if not (
            os.environ.get("LANGFUSE_PUBLIC_KEY")
            and os.environ.get("LANGFUSE_SECRET_KEY")
        ):
            pytest.skip("Requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
