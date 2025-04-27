import pymongo
import certifi
import logging
from typing import Dict, Any, List, Optional
import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)

class MongoDB:
    _client = None
    _db = None
    
    @classmethod
    def get_client(cls):
        """Obtiene conexión a MongoDB Atlas"""
        if cls._client is None:
            try:
                cls._client = pymongo.MongoClient(
                    settings.MONGODB_URI,
                    tlsCAFile=certifi.where(),
                    serverSelectionTimeoutMS=5000
                )
                # Verificar conexión
                cls._client.server_info()
                logger.info("Conexión exitosa a MongoDB Atlas!")
            except Exception as e:
                logger.error(f"Error al conectar a MongoDB: {e}")
                cls._client = None
        return cls._client
    
    @classmethod
    def get_db(cls):
        """Obtiene la base de datos"""
        if cls._db is None and cls.get_client() is not None:
            cls._db = cls._client[settings.MONGODB_DB_NAME]
        return cls._db
    
    @classmethod
    async def get_collection_data(cls, collection_name: str) -> List[Dict[str, Any]]:
        """Obtiene todos los documentos de una colección"""
        db = cls.get_db()
        if db is None:  # Cambiado de "if not db:" a "if db is None:"
            logger.error(f"No se pudo obtener la base de datos")
            return []
        
        try:
            collection = db[collection_name]
            data = list(collection.find())
            logger.info(f"Se obtuvieron {len(data)} documentos de la colección {collection_name}")
            return data
        except Exception as e:
            logger.error(f"Error al obtener datos de {collection_name}: {e}")
            return []
    
    @classmethod
    async def get_all_data(cls) -> Dict[str, pd.DataFrame]:
        """Obtiene todos los datos necesarios para el modelo"""
        data_collections = {
            "ventas": await cls.get_collection_data("ventas"),
            "ventadetalles": await cls.get_collection_data("ventadetalles"),  # CORREGIDO
            "productos": await cls.get_collection_data("productos"),
            "producto_variedads": await cls.get_collection_data("producto_variedads"),  # CORREGIDO
            "ingresodetalles": await cls.get_collection_data("ingresodetalles")  # CORREGIDO
        }
        
        # Convertir cada colección a DataFrame
        dataframes = {}
        for name, data in data_collections.items():
            dataframes[name] = pd.DataFrame(data) if data else pd.DataFrame()
        
        return dataframes