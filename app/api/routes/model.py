# Fragmento de código con endpoints adicionales para versionado
from fastapi import APIRouter, Depends, HTTPException, Query, status
from typing import Dict, Any, List, Optional
from app.core.database import MongoDB

import pandas as pd

from app.api.deps import get_db_data
from app.core.models.decision_tree import DecisionTreeModel
from app.core.models.version_manager import ModelVersionManager

router = APIRouter()

@router.get("/info")
async def get_model_info():
    """Obtener información del modelo actual"""
    model_info = DecisionTreeModel.get_model_info()
    return {
        "exito": model_info is not None,
        "mensaje": "Información del modelo recuperada" if model_info else "No hay modelo cargado",
        "info_modelo": model_info
    }

@router.post("/train")
async def train_model():
    """Entrena un nuevo modelo de árbol de decisión"""
    db_data = await MongoDB.get_all_data()
    result = await DecisionTreeModel.train(db_data)
    
    if not result["exito"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result
        )
        
    return result


@router.get("/versions")
async def get_model_versions():
    """Obtener todas las versiones de modelos disponibles"""
    versions = ModelVersionManager.get_all_versions()
    
    return {
        "exito": True,
        "mensaje": f"Se encontraron {len(versions)} versiones",
        "versiones": versions,
        "version_activa": ModelVersionManager.get_active_version().get("version_id") if ModelVersionManager.get_active_version() else None
    }

@router.get("/versions/{version_id}")
async def get_model_version(version_id: str):
    """Obtener detalles de una versión específica del modelo"""
    version = ModelVersionManager.get_version_by_id(version_id)
    
    if not version:
        raise HTTPException(status_code=404, detail=f"Versión {version_id} no encontrada")
        
    return {
        "exito": True,
        "mensaje": f"Versión {version_id} recuperada",
        "version": version
    }

@router.post("/versions/{version_id}/activate")
async def activate_model_version(version_id: str):
    """Activar una versión específica del modelo"""
    active_version = ModelVersionManager.activate_version(version_id)
    
    if not active_version:
        raise HTTPException(status_code=404, detail=f"No se pudo activar la versión {version_id}")
        
    return {
        "exito": True,
        "mensaje": f"Versión {version_id} activada correctamente",
        "version": active_version
    }

@router.get("/versions/compare")
async def compare_model_versions(
    version1_id: str = Query(..., description="ID de la primera versión"),
    version2_id: str = Query(..., description="ID de la segunda versión"),
    db_data: Dict[str, pd.DataFrame] = Depends(get_db_data)
):
    """Comparar el rendimiento de dos versiones del modelo"""
    # Implementación simplificada - en un sistema real necesitarías cargar 
    # ambas versiones y evaluar con datos de prueba
    
    v1 = ModelVersionManager.get_version_by_id(version1_id)
    v2 = ModelVersionManager.get_version_by_id(version2_id)
    
    if not v1 or not v2:
        raise HTTPException(status_code=404, detail="Una o ambas versiones no encontradas")
    
    # Simulamos una comparación básica
    comparison = {
        "precision": {
            "version1": v1["model_info"]["precision"],
            "version2": v2["model_info"]["precision"],
            "diferencia": v2["model_info"]["precision"] - v1["model_info"]["precision"]
        },
        "fecha_entrenamiento": {
            "version1": v1["model_info"]["fecha_entrenamiento"],
            "version2": v2["model_info"]["fecha_entrenamiento"]
        },
        "caracteristicas": {
            "version1": v1["model_info"]["caracteristicas"],
            "version2": v2["model_info"]["caracteristicas"],
            "diferencias": list(set(v2["model_info"]["caracteristicas"]) - set(v1["model_info"]["caracteristicas"]))
        }
    }
    
    return {
        "exito": True,
        "mensaje": f"Comparación entre versiones {version1_id} y {version2_id}",
        "comparacion": comparison,
        "recomendacion": "version2" if comparison["precision"]["diferencia"] > 0 else "version1"
    }