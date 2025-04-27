from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class ProductBase(BaseModel):
    """Modelo base para productos"""
    variedad_id: str
    producto_id: str
    nombre_completo: str
    color: Optional[str] = None
    talla: Optional[str] = None
    inventario_actual: float
    precio: float = 0

class ProductFeatures(ProductBase):
    """Modelo con características para el análisis"""
    ventas_30d: float = 0
    ventas_90d: float = 0
    ventas_diarias_prom: float = 0
    dias_stock_restante: float = 0
    tendencia: float = 0
    coef_variacion: float = 0

class InventoryStatus(ProductFeatures):
    """Modelo para el estado de inventario"""
    estado_inventario: str
    
class PredictionRequest(BaseModel):
    """Modelo para solicitar predicciones"""
    productos: List[ProductFeatures]

class RecommendationItem(BaseModel):
    """Modelo para un ítem de recomendación"""
    variedad_id: str
    producto_id: str
    nombre: str
    color: Optional[str] = None
    talla: Optional[str] = None
    estado: str
    urgencia: str
    inventario_actual: int
    dias_restantes: int
    fecha_desabastecimiento: Optional[str] = None
    cantidad_reordenar: Optional[int] = None
    ventas_diarias: float
    tendencia: str
    precio: float
    recomendacion: Optional[str] = None
    
    
class InventoryAnalysis(BaseModel):
    """Modelo para el análisis del inventario"""
    total_productos: int
    distribucion: Dict[str, int]
    productos_criticos: List[InventoryStatus]
    productos_bajos: List[InventoryStatus]
    
class ModelInfo(BaseModel):
    """Información sobre el modelo entrenado"""
    modelo: str = "Árbol de Decisión"
    fecha_entrenamiento: Optional[datetime] = None
    precision: Optional[float] = None
    caracteristicas: List[str] = []
    profundidad_maxima: int = 0
    esta_cargado: bool = False

class ModelTrainingResponse(BaseModel):
    """Respuesta al entrenar el modelo"""
    exito: bool
    mensaje: str
    info_modelo: Optional[ModelInfo] = None
    
    
class ProductInput(BaseModel):
    variedad_id: str  
    producto_id: str
    nombre_completo: str
    color: Optional[str] = ""
    talla: Optional[str] = ""
    inventario_actual: int
    precio: Optional[float] = 0.0
    ventas_30d: float
    ventas_90d: Optional[float] = None
    ventas_diarias_prom: Optional[float] = None
    dias_stock_restante: Optional[float] = None
    tendencia: Optional[float] = None
    coef_variacion: Optional[float] = None
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "variedad_id": "123abc",
                "producto_id": "456def",
                "nombre_completo": "Camiseta Azul (M)",
                "color": "Azul",
                "talla": "M",
                "inventario_actual": 10,
                "precio": 19.99,
                "ventas_30d": 25,
                "ventas_90d": 70,
                "ventas_diarias_prom": 0.83,
                "dias_stock_restante": 12.05,
                "tendencia": 0.15,
                "coef_variacion": 0.3
            }
        }
    }

class ProductBatch(BaseModel):
    productos: List[ProductInput]
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "productos": [
                    {
                        "variedad_id": "123abc",
                        "producto_id": "456def",
                        "nombre_completo": "Camiseta Azul (M)",
                        "color": "Azul",
                        "talla": "M",
                        "inventario_actual": 10,
                        "precio": 19.99,
                        "ventas_30d": 25,
                        "ventas_90d": 70,
                        "ventas_diarias_prom": 0.83,
                        "dias_stock_restante": 12.05,
                        "tendencia": 0.15,
                        "coef_variacion": 0.3
                    }
                ]
            }
        }
    }

class ProductPrediction(BaseModel):
    producto_id: str
    nombre: str 
    inventario_actual: int
    ventas_30d: float
    estado: str
    estado_codigo: int
    dias_restantes: float
    
    
    
class InventoryNewStatus(BaseModel):
    variedad_id: str
    producto_id: str
    nombre_completo: str
    color: Optional[str] = None
    talla: Optional[str] = None
    inventario_actual: int
    precio: Optional[float] = None
    ventas_30d: Optional[int] = None
    ventas_90d: Optional[int] = None
    ventas_diarias_prom: Optional[float] = None
    dias_stock_restante: Optional[float] = None
    tendencia: Optional[float] = None
    coef_variacion: Optional[float] = None
    estado_inventario: str

class InventoryRecommendation(BaseModel):
    variedad_id: str
    producto_id: str
    nombre_completo: str
    inventario_actual: int
    cantidad_recomendada: int
    dias_stock_actual: float
    dias_stock_objetivo: int
    prioridad: int