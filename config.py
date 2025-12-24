"""
Configuration management for production
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Centralized configuration with validation"""
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    
    # Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    TABLE_NAME: str = os.getenv("TABLE_NAME", "scripture_embeddings")
    TEXT_COLUMN: str = os.getenv("TEXT_COLUMN", "text")
    ID_COLUMN: str = os.getenv("ID_COLUMN", "id")
    
    # HuggingFace
    HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")
    HF_EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    
    # RAG Settings
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "10"))
    MATCH_COUNT: int = int(os.getenv("MATCH_COUNT", "5"))
    
    # Memory Settings
    MAX_HISTORY_MESSAGES: int = 10
    CONTEXT_MESSAGES: int = 6
    
    # Safety Settings
    ENABLE_MODERATION: bool = os.getenv("ENABLE_MODERATION", "true").lower() == "true"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
        required = [
            ("OPENAI_API_KEY", cls.OPENAI_API_KEY),
            ("SUPABASE_URL", cls.SUPABASE_URL),
            ("SUPABASE_KEY", cls.SUPABASE_KEY),
            ("HF_API_TOKEN", cls.HF_API_TOKEN),
        ]
        
        missing = []
        for name, value in required:
            if not value:
                missing.append(name)
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        return True
    
    @classmethod
    def get_huggingface_url(cls) -> str:
        """Get HuggingFace embedding URL"""
        return f"https://router.huggingface.co/hf-inference/models/{cls.HF_EMBEDDING_MODEL}"


# Validate on import
try:
    Config.validate()
    print("✅ Configuration validated")
except ValueError as e:
    print(f"❌ Configuration error: {e}")
    raise