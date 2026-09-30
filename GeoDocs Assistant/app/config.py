from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    collection: str = "geodocs_v2"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    chunk_chars: int = 1200
    chunk_overlap: int = 150

    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidates: int = 20  # fetched before reranking
    default_mode: str = "hybrid_rerank"
    geocoder_url: str = "https://nominatim.openstreetmap.org/search"
    geocoder_user_agent: str = "geodocs-assistant/0.1 (portfolio project)"

    # refuse below these (tune on your docs); hybrid-only (RRF) has no threshold
    min_score_dense: float = 0.40
    min_score_rerank: float = 0.0


settings = Settings()
