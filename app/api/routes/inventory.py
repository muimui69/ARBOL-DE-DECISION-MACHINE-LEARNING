from fastapi import APIRouter, Depends, HTTPException, Query,status, Body
from typing import List, Dict, Any,Union

from app.models.schemas import (
    InventoryAnalysis, InventoryStatus, PredictionRequest, 
    RecommendationItem, ModelInfo, ModelTrainingResponse,InventoryRecommendation,InventoryNewStatus
)
from app.api.deps import get_db_data
from app.services.inventary_service import InventoryService
from app.core.models.decision_tree import DecisionTreeModel
from app.models.schemas import ProductInput, ProductPrediction, ProductBatch
import pandas as pd

router = APIRouter()

# @router.post("/predict", response_model=List[ProductPrediction])
# async def predict_inventory_status(data: ProductBatch = Body(...)):
#     """Predice el estado del inventario para una lista de productos"""
#     try:
#         # Obtener la lista de productos
#         products = data.productos
        
#         if not products:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="No se proporcionaron productos para predecir"
#             )
        
#         # Convertir los datos de entrada a DataFrame
#         products_data = []
#         for product in products:
#             # Agregar cada producto como un diccionario
#             products_data.append({
#                 'stock': product.inventario_actual,
#                 'ventas_30d': product.ventas_30d,
#                 'ventas_90d': product.ventas_90d or (product.ventas_30d * 3),
#                 'ventas_diarias_prom': product.ventas_diarias_prom or (product.ventas_30d / 30),
#                 'dias_stock_restante': product.dias_stock_restante or 0.0,
#                 'tendencia': product.tendencia or 0.0,
#                 'coef_variacion': product.coef_variacion or 0.2
#             })
        
#         # Crear DataFrame con las columnas que espera el modelo
#         products_df = pd.DataFrame(products_data)
        
#         # Realizar predicción
#         predictions = await DecisionTreeModel.predict(products_df)
        
#         if not predictions["exito"]:
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail=predictions["error"]
#             )
        
#         # Construir respuesta
#         results = []
#         for i, product in enumerate(products):
#             # Crear el objeto de predicción y añadirlo a la lista
#             prediction = ProductPrediction(
#                 producto_id=product.producto_id,
#                 nombre=product.nombre_completo,
#                 inventario_actual=product.inventario_actual,
#                 ventas_30d=product.ventas_30d,
#                 estado=predictions["resultados"]["estados"][i],
#                 estado_codigo=predictions["resultados"]["codigos"][i],
#                 dias_restantes=predictions["resultados"]["dias_restantes"][i]
#             )
#             results.append(prediction)
        
#         return results
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error en predicción: {str(e)}"
#         )


@router.post("/predict", response_model=List[ProductPrediction])
async def predict_inventory(request: dict):
    """
    Realiza predicciones de estado de inventario para una lista de productos
    """
    try:
        productos = request.get("productos", [])
        if not productos:
            raise HTTPException(
                status_code=400, 
                detail="No se proporcionaron productos para predicción"
            )
            
        # Convertir a DataFrame para el modelo
        df = pd.DataFrame(productos)
        
        # Verificar campos mínimos requeridos
        required_fields = ['inventario_actual', 'ventas_30d', 'ventas_90d', 'ventas_diarias_prom']
        missing = [field for field in required_fields if field not in df.columns]
        if missing:
            # Si faltan ventas_diarias_prom, intentar calcularlo
            if 'ventas_diarias_prom' in missing and 'ventas_30d' in df.columns:
                df['ventas_diarias_prom'] = df['ventas_30d'] / 30
                missing.remove('ventas_diarias_prom')
            
            # Si aún faltan campos, lanzar error
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Faltan campos requeridos: {', '.join(missing)}"
                )
        
        # Calcular días de stock si no está presente
        if 'dias_stock_restante' not in df.columns:
            # Evitar división por cero
            ventas_diarias = df['ventas_diarias_prom'].replace(0, 0.1)
            df['dias_stock_restante'] = df['inventario_actual'] / ventas_diarias
            df.loc[df['inventario_actual'] == 0, 'dias_stock_restante'] = 0
        
        # Llamar al modelo para predecir
        prediction_result = await DecisionTreeModel.predict(df)
        
        if not prediction_result["exito"]:
            raise HTTPException(
                status_code=500, 
                detail=prediction_result["error"]
            )
        
        # Extraer resultados
        estados = prediction_result["resultados"]["estados"]
        codigos = prediction_result["resultados"]["codigos"]
        dias = prediction_result["resultados"]["dias_restantes"]
        
        # Construir respuesta
        predictions = []
        for i, producto in enumerate(productos):
            if i < len(estados) and i < len(codigos) and i < len(dias):
                predictions.append(ProductPrediction(
                    producto_id=producto.get('producto_id', ''),
                    nombre=producto.get('nombre_completo', ''),
                    inventario_actual=producto.get('inventario_actual', 0),
                    ventas_30d=float(producto.get('ventas_30d', 0)),
                    estado=estados[i],
                    estado_codigo=int(codigos[i]),
                    dias_restantes=float(dias[i])
                ))
        
        return predictions
        
    except Exception as e:
        import traceback
        print(f"ERROR en predicción: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error en la predicción: {str(e)}"
        )


@router.get("/analysis", response_model=InventoryAnalysis)
async def analyze_inventory(
    limit: int = Query(10, description="Máximo número de productos por categoría"),
    data: Dict[str, pd.DataFrame] = Depends(get_db_data)
):
    """
    Realiza un análisis completo del inventario:
    - Distribución por categorías
    - Productos críticos
    - Productos con bajo stock
    """
    try:
        # Usar el nuevo servicio de inventario
        analysis_result = await InventoryService.analyze_inventory(data)
        
        if not analysis_result["exito"]:
            raise HTTPException(
                status_code=500,
                detail=analysis_result["error"]
            )
        
        # Obtener datos del resultado
        distribucion = analysis_result["distribucion"]
        productos_criticos = analysis_result["productos_criticos"]
        productos_bajos = analysis_result["productos_bajos"]
        
        # Convertir a formato de respuesta
        criticos_list = []
        for _, row in productos_criticos.head(limit).iterrows():
            criticos_list.append(InventoryStatus(**row.to_dict()))
            
        bajos_list = []
        for _, row in productos_bajos.head(limit).iterrows():
            bajos_list.append(InventoryStatus(**row.to_dict()))
        
        return {
            "total_productos": analysis_result["total_productos"],
            "distribucion": distribucion,
            "productos_criticos": criticos_list,
            "productos_bajos": bajos_list
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error en el análisis de inventario: {str(e)}"
        )

@router.get("/critical", response_model=List[InventoryNewStatus])
async def get_critical_products(
    data: Dict[str, pd.DataFrame] = Depends(get_db_data),
    limit: int = Query(50, description="Máximo número de productos"),
    offset: int = Query(0, description="Número de productos a saltar")
):
    """
    Obtiene la lista de productos en estado crítico
    """
    try:
        # Usar el servicio de inventario
        analysis_result = await InventoryService.analyze_inventory(data)
        
        if not analysis_result["exito"]:
            raise HTTPException(
                status_code=500,
                detail=analysis_result["error"]
            )
        
        # Obtener productos críticos
        productos_criticos = analysis_result["productos_criticos"]
        
        # Aplicar paginación
        paginados = productos_criticos.iloc[offset:offset+limit]
        
        # Convertir a formato de respuesta
        results = []
        for _, row in paginados.iterrows():
            results.append(InventoryStatus(**row.to_dict()))
            
        return results
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener productos críticos: {str(e)}"
        )

@router.get("/recommendations", response_model=List[InventoryRecommendation])
async def get_recommendations(
    data: Dict[str, pd.DataFrame] = Depends(get_db_data),
    limit: int = Query(20, description="Máximo número de recomendaciones"),
    min_days: int = Query(14, description="Días mínimos de stock objetivo")
):
    """
    Genera recomendaciones de reabastecimiento
    """
    try:
        # Obtener recomendaciones del servicio
        recommendations = await InventoryService.get_recommendations(data, min_days)
        
        # Convertir a objetos Pydantic
        recommendation_objects = []
        for rec in recommendations[:limit]:
            recommendation_objects.append(InventoryRecommendation(**rec))
        
        return recommendation_objects
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar recomendaciones: {str(e)}"
        )
