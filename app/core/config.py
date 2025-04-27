import os
from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    # API information
    PROJECT_NAME: str = "Inventory Analysis API"
    PROJECT_DESCRIPTION: str = "API para análisis predictivo de inventario usando árboles de decisión"
    PROJECT_VERSION: str = "1.0.0"
    API: str = "/api"
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]
    
    # MongoDB settings
    MONGODB_URI: str = os.getenv(
        "MONGODB_URI", 
        "mongodb+srv://moisodev:moiso@cluster0.crz8eun.mongodb.net/EcommerML?retryWrites=true&w=majority"
    )
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "EcommerML")
    
    # Model settings
    MODEL_PATH: str = os.getenv("MODEL_PATH", "./models")
    MODEL_MAX_DEPTH: int = int(os.getenv("MODEL_MAX_DEPTH"))
    DAYS_OF_INVENTORY_TARGET: int = int(os.getenv("DAYS_OF_INVENTORY_TARGET"))
    
    # Inventory thresholds
    CRITICAL_THRESHOLD: float = float(os.getenv("CRITICAL_THRESHOLD"))  # días
    LOW_THRESHOLD: float = float(os.getenv("LOW_THRESHOLD"))           # días
    NORMAL_THRESHOLD: float = float(os.getenv("NORMAL_THRESHOLD"))     # días
    HIGH_THRESHOLD: float = float(os.getenv("HIGH_THRESHOLD"))         # días
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Crear una instancia global de configuraciones
settings = Settings()