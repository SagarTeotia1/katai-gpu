from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Ollama connection
    llm_base_url: str = "http://ollama:11434"

    # Model (multimodal — supports text + image)
    model_id: str = "qwen3.6:27b-bf16"

    @property
    def vision_model_id(self) -> str:
        return self.model_id

    # Generation defaults
    max_tokens: int = 4096
    temperature: float = 0.7

    # Server
    host: str = "0.0.0.0"
    port: int = 8080

    # CORS — comma-separated origins, "*" allows all
    allowed_origins: str = "*"

    @property
    def llm_chat_url(self) -> str:
        return f"{self.llm_base_url}/v1/chat/completions"

    @property
    def llm_models_url(self) -> str:
        return f"{self.llm_base_url}/v1/models"

    @property
    def llm_health_url(self) -> str:
        # Ollama health endpoint — returns list of local models
        return f"{self.llm_base_url}/api/tags"

    @property
    def origins(self) -> list[str]:
        if self.allowed_origins == "*":
            return ["*"]
        return [o.strip() for o in self.allowed_origins.split(",")]


settings = Settings()
