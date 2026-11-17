from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "CSPRO RAG Service"
    version: str = "1.0.0"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "command-r:35b"
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_dir: str = "./chroma_data"
    chroma_in_memory: bool = False
    max_question_length: int = 1000
    default_top_k: int = 5
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
