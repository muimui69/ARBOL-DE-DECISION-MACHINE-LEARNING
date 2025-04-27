from fastapi import APIRouter, Depends, Query, HTTPException, status
from typing import List, Dict, Any, Optional
import pandas as pd

from app.api.deps import get_db_data
from app.services.cycle_service import CycleService

router = APIRouter()

@router.get("/temporal/simple")
async def simple_temporal_cycles(
    cycle_type: str = Query(..., description="Tipo de ciclo: 'weekly', 'monthly', 'seasonal', 'yearly'"),
    db_data: Dict[str, pd.DataFrame] = Depends(get_db_data)
):
    """
    Análisis temporal simplificado sin joins complejos
    
    - **cycle_type**: Tipo de ciclo (weekly=semanal, monthly=mensual, seasonal=estacional, yearly=anual)
    """
    # Validar tipo de ciclo
    valid_cycle_types = ['weekly', 'monthly', 'seasonal', 'yearly']
    if cycle_type not in valid_cycle_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tipo de ciclo no válido. Debe ser uno de: {', '.join(valid_cycle_types)}"
        )
    
    # Llamar al servicio para análisis simplificado
    result = await CycleService.analyze_temporal_cycles_simple(db_data, cycle_type)
    
    if not result["exito"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["mensaje"]
        )
        
    return result

@router.get("/product/simple")
async def simple_product_cycles(
    cycle_type: str = Query(..., description="Tipo de ciclo: 'lifecycle', 'replenishment', 'promotional'"),
    product_id: Optional[str] = Query(None, description="ID del producto específico a analizar"),
    db_data: Dict[str, pd.DataFrame] = Depends(get_db_data)
):
    """
    Análisis simplificado de ciclos de producto sin joins complejos
    
    - **cycle_type**: Tipo de ciclo (lifecycle, replenishment, promotional)
    - **product_id**: ID del producto específico (opcional)
    """
    # Validar tipo de ciclo
    valid_cycle_types = ['lifecycle', 'replenishment', 'promotional']
    if cycle_type not in valid_cycle_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tipo de ciclo no válido. Debe ser uno de: {', '.join(valid_cycle_types)}"
        )
    
    # Llamar al servicio para análisis simplificado
    result = await CycleService.analyze_product_cycles_simple(db_data, cycle_type, product_id)
    
    if not result["exito"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["mensaje"]
        )
        
    return result

@router.get("/debug")
async def debug_cycles_data(
    db_data: Dict[str, pd.DataFrame] = Depends(get_db_data)
):
    """
    Endpoint para depurar la estructura de datos para análisis de ciclos
    """
    try:
        # Info básica de colecciones
        collections_info = {}
        for key, df in db_data.items():
            collections_info[key] = {
                "rows": len(df),
                "columns": list(df.columns) if not df.empty else []
            }
        
        # Verificar datos clave
        ventas_info = {}
        if "ventas" in db_data and not db_data["ventas"].empty:
            ventas_df = db_data["ventas"]
            ventas_info = {
                "total_ventas": len(ventas_df),
                "fecha_min": str(ventas_df["createdAT"].min()) if "createdAT" in ventas_df.columns else "N/A",
                "fecha_max": str(ventas_df["createdAT"].max()) if "createdAT" in ventas_df.columns else "N/A"
            }
        
        # Verificar estructura de detalles
        detalles_info = {}
        if "ventadetalles" in db_data and not db_data["ventadetalles"].empty:
            detalles_df = db_data["ventadetalles"]
            detalles_info = {
                "total_detalles": len(detalles_df),
                "columnas_clave": {
                    "producto": "producto" in detalles_df.columns,
                    "venta": "venta" in detalles_df.columns,
                    "cantidad": "cantidad" in detalles_df.columns
                }
            }
        
        return {
            "collections": collections_info,
            "ventas_info": ventas_info,
            "detalles_info": detalles_info
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "traceback": __import__("traceback").format_exc()
        }