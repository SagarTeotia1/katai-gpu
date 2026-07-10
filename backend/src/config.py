from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # vLLM connection
    llm_base_url: str = "http://vllm:8000"

    # Model (HuggingFace model ID)
    model_id: str = "Qwen/Qwen3.6-27B"

    @property
    def vision_model_id(self) -> str:
        return self.model_id

    # Generation defaults
    max_tokens: int = 4096
    temperature: float = 0.7

    # Vision — higher token limit for exhaustive descriptions
    vision_max_tokens: int = 8192

    # Video — full semantic analysis
    video_max_tokens: int = 16384
    # Chunk analysis — shorter per-chunk budget; json-repair handles truncation
    video_chunk_max_tokens: int = 8192
    # 1 fps = 1 frame/s; fewer prompt tokens → faster prefill + generation start
    video_fps: float = 1.0

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
        return f"{self.llm_base_url}/health"

    @property
    def origins(self) -> list[str]:
        if self.allowed_origins == "*":
            return ["*"]
        return [o.strip() for o in self.allowed_origins.split(",")]


settings = Settings()
