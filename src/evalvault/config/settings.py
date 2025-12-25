"""Application settings using pydantic-settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    openai_base_url: str | None = Field(
        default=None, description="Custom OpenAI API base URL (optional)"
    )
    openai_model: str = Field(
        default="gpt-5-nano", description="OpenAI model to use for evaluation"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model"
    )

    # Azure OpenAI Configuration (optional)
    azure_api_key: str | None = Field(default=None, description="Azure OpenAI API key")
    azure_endpoint: str | None = Field(default=None, description="Azure OpenAI endpoint URL")
    azure_deployment: str | None = Field(default=None, description="Azure deployment name")
    azure_embedding_deployment: str | None = Field(
        default=None, description="Azure embedding deployment name"
    )
    azure_api_version: str = Field(
        default="2024-02-15-preview", description="Azure API version"
    )

    # Anthropic Configuration (optional)
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    anthropic_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Anthropic Claude model to use for evaluation",
    )

    # Langfuse Configuration (optional)
    langfuse_public_key: str | None = Field(default=None, description="Langfuse public key")
    langfuse_secret_key: str | None = Field(default=None, description="Langfuse secret key")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", description="Langfuse host URL")

    # MLflow Configuration (optional)
    mlflow_tracking_uri: str | None = Field(default=None, description="MLflow tracking server URI")
    mlflow_experiment_name: str = Field(default="evalvault", description="MLflow experiment name")

    # PostgreSQL Configuration (optional)
    postgres_host: str | None = Field(default=None, description="PostgreSQL server host")
    postgres_port: int = Field(default=5432, description="PostgreSQL server port")
    postgres_database: str = Field(default="evalvault", description="PostgreSQL database name")
    postgres_user: str | None = Field(default=None, description="PostgreSQL user")
    postgres_password: str | None = Field(default=None, description="PostgreSQL password")
    postgres_connection_string: str | None = Field(
        default=None, description="PostgreSQL connection string (overrides other postgres settings)"
    )

# Global settings instance (lazy initialization)
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create global settings instance.

    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# For backward compatibility
settings = get_settings()
