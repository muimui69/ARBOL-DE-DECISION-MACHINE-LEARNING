from fastapi import APIRouter, Depends, HTTPException, Query,status, Body,BackgroundTasks
from typing import List, Dict, Any,Union
from fastapi.responses import JSONResponse

from app.models.schemas import (
    InventoryAnalysis, InventoryStatus, PredictionRequest, 
    RecommendationItem, ModelInfo, ModelTrainingResponse,InventoryRecommendation,InventoryNewStatus
)
from app.api.deps import get_db_data
from app.services.inventary_service import InventoryService
from app.core.models.decision_tree import DecisionTreeModel
from app.models.schemas import ProductInput, ProductPrediction, ProductBatch
import pandas as pd
import time

router = APIRouter()

# Caché simple en memoria (para desarrollo)
_cache = {}
_cache_timestamp = {}
CACHE_TTL = 300  # 5 minutos

#? este no
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


#? este si
# @router.post("/predict", response_model=List[ProductPrediction])
# async def predict_inventory(request: dict):
#     """
#     Realiza predicciones de estado de inventario para una lista de productos
#     """
#     try:
#         productos = request.get("productos", [])
#         if not productos:
#             raise HTTPException(
#                 status_code=400, 
#                 detail="No se proporcionaron productos para predicción"
#             )
            
#         # Convertir a DataFrame para el modelo
#         df = pd.DataFrame(productos)
        
#         # Verificar campos mínimos requeridos
#         required_fields = ['inventario_actual', 'ventas_30d', 'ventas_90d', 'ventas_diarias_prom']
#         missing = [field for field in required_fields if field not in df.columns]
#         if missing:
#             # Si faltan ventas_diarias_prom, intentar calcularlo
#             if 'ventas_diarias_prom' in missing and 'ventas_30d' in df.columns:
#                 df['ventas_diarias_prom'] = df['ventas_30d'] / 30
#                 missing.remove('ventas_diarias_prom')
            
#             # Si aún faltan campos, lanzar error
#             if missing:
#                 raise HTTPException(
#                     status_code=400,
#                     detail=f"Faltan campos requeridos: {', '.join(missing)}"
#                 )
        
#         # Calcular días de stock si no está presente
#         if 'dias_stock_restante' not in df.columns:
#             # Evitar división por cero
#             ventas_diarias = df['ventas_diarias_prom'].replace(0, 0.1)
#             df['dias_stock_restante'] = df['inventario_actual'] / ventas_diarias
#             df.loc[df['inventario_actual'] == 0, 'dias_stock_restante'] = 0
        
#         # Llamar al modelo para predecir
#         prediction_result = await DecisionTreeModel.predict(df)
        
#         if not prediction_result["exito"]:
#             raise HTTPException(
#                 status_code=500, 
#                 detail=prediction_result["error"]
#             )
        
#         # Extraer resultados
#         estados = prediction_result["resultados"]["estados"]
#         codigos = prediction_result["resultados"]["codigos"]
#         dias = prediction_result["resultados"]["dias_restantes"]
        
#         # Construir respuesta
#         predictions = []
#         for i, producto in enumerate(productos):
#             if i < len(estados) and i < len(codigos) and i < len(dias):
#                 predictions.append(ProductPrediction(
#                     producto_id=producto.get('producto_id', ''),
#                     nombre=producto.get('nombre_completo', ''),
#                     inventario_actual=producto.get('inventario_actual', 0),
#                     ventas_30d=float(producto.get('ventas_30d', 0)),
#                     estado=estados[i],
#                     estado_codigo=int(codigos[i]),
#                     dias_restantes=float(dias[i])
#                 ))
        
#         return predictions
        
#     except Exception as e:
#         import traceback
#         print(f"ERROR en predicción: {e}")
#         print(traceback.format_exc())
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error en la predicción: {str(e)}"
#         )


# @router.get("/analysis", response_model=InventoryAnalysis)
# async def analyze_inventory(
#     limit: int = Query(10, description="Máximo número de productos por categoría"),
#     data: Dict[str, pd.DataFrame] = Depends(get_db_data)
# ):
#     """
#     Realiza un análisis completo del inventario:
#     - Distribución por categorías
#     - Productos críticos
#     - Productos con bajo stock
#     """
#     try:
#         # Usar el nuevo servicio de inventario
#         analysis_result = await InventoryService.analyze_inventory(data)
        
#         if not analysis_result["exito"]:
#             raise HTTPException(
#                 status_code=500,
#                 detail=analysis_result["error"]
#             )
        
#         # Obtener datos del resultado
#         distribucion = analysis_result["distribucion"]
#         productos_criticos = analysis_result["productos_criticos"]
#         productos_bajos = analysis_result["productos_bajos"]
        
#         # Convertir a formato de respuesta
#         criticos_list = []
#         for _, row in productos_criticos.head(limit).iterrows():
#             criticos_list.append(InventoryStatus(**row.to_dict()))
            
#         bajos_list = []
#         for _, row in productos_bajos.head(limit).iterrows():
#             bajos_list.append(InventoryStatus(**row.to_dict()))
        
#         return {
#             "total_productos": analysis_result["total_productos"],
#             "distribucion": distribucion,
#             "productos_criticos": criticos_list,
#             "productos_bajos": bajos_list
#         }
    
#     except Exception as e:
#         raise HTTPException(
#             status_code=500, 
#             detail=f"Error en el análisis de inventario: {str(e)}"
#         )

# @router.get("/critical", response_model=List[InventoryNewStatus])
# async def get_critical_products(
#     data: Dict[str, pd.DataFrame] = Depends(get_db_data),
#     limit: int = Query(50, description="Máximo número de productos"),
#     offset: int = Query(0, description="Número de productos a saltar")
# ):
#     """
#     Obtiene la lista de productos en estado crítico
#     """
#     try:
#         # Usar el servicio de inventario
#         analysis_result = await InventoryService.analyze_inventory(data)
        
#         if not analysis_result["exito"]:
#             raise HTTPException(
#                 status_code=500,
#                 detail=analysis_result["error"]
#             )
        
#         # Obtener productos críticos
#         productos_criticos = analysis_result["productos_criticos"]
        
#         # Aplicar paginación
#         paginados = productos_criticos.iloc[offset:offset+limit]
        
#         # Convertir a formato de respuesta
#         results = []
#         for _, row in paginados.iterrows():
#             results.append(InventoryStatus(**row.to_dict()))
            
#         return results
    
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error al obtener productos críticos: {str(e)}"
#         )

# @router.get("/recommendations", response_model=List[InventoryRecommendation])
# async def get_recommendations(
#     data: Dict[str, pd.DataFrame] = Depends(get_db_data),
#     limit: int = Query(20, description="Máximo número de recomendaciones"),
#     min_days: int = Query(14, description="Días mínimos de stock objetivo")
# ):
#     """
#     Genera recomendaciones de reabastecimiento
#     """
#     try:
#         # Obtener recomendaciones del servicio
#         recommendations = await InventoryService.get_recommendations(data, min_days)
        
#         # Convertir a objetos Pydantic
#         recommendation_objects = []
#         for rec in recommendations[:limit]:
#             recommendation_objects.append(InventoryRecommendation(**rec))
        
#         return recommendation_objects
    
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error al generar recomendaciones: {str(e)}"
#         )

async def get_cached_or_compute(cache_key: str, compute_func, *args, **kwargs):
    """Helper para obtener datos en caché o calcularlos"""
    now = time.time()
    
    # Si hay datos en caché y son válidos
    if cache_key in _cache and (now - _cache_timestamp.get(cache_key, 0)) < CACHE_TTL:
        return _cache[cache_key]
    
    # Calcular datos frescos
    result = await compute_func(*args, **kwargs)
    
    # Guardar en caché
    _cache[cache_key] = result
    _cache_timestamp[cache_key] = now
    
    return result

@router.post("/predict", response_model=List[ProductPrediction])
async def predict_inventory(request: dict):
    """Realiza predicciones de estado de inventario"""
    try:
        productos = request.get("productos", [])
        if not productos:
            raise HTTPException(
                status_code=400, 
                detail="No se proporcionaron productos para predicción"
            )
            
        # Limitar cantidad de productos para evitar timeouts
        if len(productos) > 100:
            productos = productos[:100]
            
        # Convertir a DataFrame para el modelo
        df = pd.DataFrame(productos)
        
        # Verificar campos mínimos requeridos
        required_fields = ['inventario_actual', 'ventas_30d']
        missing = [field for field in required_fields if field not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Faltan campos requeridos: {', '.join(missing)}"
            )
        
        # Añadir campos calculados si faltan
        if 'ventas_90d' not in df.columns:
            df['ventas_90d'] = df['ventas_30d'] * 3
            
        if 'ventas_diarias_prom' not in df.columns:
            df['ventas_diarias_prom'] = df['ventas_30d'] / 30
            
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
    data: Dict[str, pd.DataFrame] = Depends(get_db_data),
    background_tasks: BackgroundTasks = None,
):
    """
    Realiza un análisis completo del inventario con caché y optimización
    """
    try:
        # Clave de caché basada en el límite
        cache_key = f"analysis_limit_{limit}"
        
        # Obtener de caché o calcular
        analysis_result = await get_cached_or_compute(
            cache_key,
            InventoryService.analyze_inventory,
            data
        )
        
        if not analysis_result["exito"]:
            raise HTTPException(
                status_code=500,
                detail=analysis_result["error"]
            )
        
        # Obtener datos del resultado
        distribucion = analysis_result["distribucion"]
        productos_criticos = analysis_result["productos_criticos"]
        productos_bajos = analysis_result["productos_bajos"]
        
        # Convertir a formato de respuesta con límite
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
        # Ofrecer versión simplificada como fallback
        try:
            return await analyze_inventory_simple(limit, data)
        except:
            raise HTTPException(
                status_code=500, 
                detail=f"Error en el análisis de inventario: {str(e)}"
            )


# Endpoint alternativo simplificado para casos de timeout
@router.get("/analysis")
async def analyze_inventory_simple(
    limit: int = Query(10, description="Máximo número de productos por categoría"),
    data: Dict[str, pd.DataFrame] = Depends(get_db_data)
):
    """
    Versión simplificada del análisis de inventario para evitar timeouts
    """
    try:
        # Procesar datos básicos - sin cálculos complejos
        productos_df = data.get("productos", pd.DataFrame())
        variedades_df = data.get("producto_variedads", pd.DataFrame())
        
        if variedades_df.empty:
            return {
                "exito": False,
                "mensaje": "No hay datos de variedades disponibles"
            }
        
        # Contar por estado si está disponible, o crear conteo básico
        if 'estado' in variedades_df.columns:
            distribucion = variedades_df['estado'].value_counts().to_dict()
        else:
            distribucion = {"Desconocido": len(variedades_df)}
        
        # Información simplificada de stock
        criticos = []
        bajos = []
        
        if 'cantidad' in variedades_df.columns:
            # Identificar productos con bajo stock (simplificado)
            productos_bajos = variedades_df[variedades_df['cantidad'] < 5].head(limit)
            # Listar productos sin stock
            productos_sin_stock = variedades_df[variedades_df['cantidad'] <= 0].head(limit)
            
            # Si podemos vincular con productos
            if 'producto' in variedades_df.columns and not productos_df.empty:
                # Crear diccionario de IDs a nombres
                producto_nombres = {}
                for _, producto in productos_df.iterrows():
                    if '_id' in producto and 'titulo' in producto:
                        producto_nombres[str(producto['_id'])] = producto['titulo']
                
                # Simplificar para producto sin stock
                for _, var in productos_sin_stock.iterrows():
                    nombre = producto_nombres.get(str(var.get('producto', '')), 'Producto sin nombre')
                    criticos.append(InventoryStatus(
                        producto_id=str(var.get('producto', '')),
                        variedad_id=str(var.get('_id', '')),
                        nombre_completo=nombre,
                        inventario_actual=0,
                        estado_inventario="Crítico"
                    ))
                
                # Simplificar para productos bajos
                for _, var in productos_bajos.iterrows():
                    if var['cantidad'] > 0:  # No duplicar productos sin stock
                        nombre = producto_nombres.get(str(var.get('producto', '')), 'Producto sin nombre')
                        bajos.append(InventoryStatus(
                            producto_id=str(var.get('producto', '')),
                            variedad_id=str(var.get('_id', '')),
                            nombre_completo=nombre,
                            inventario_actual=int(var['cantidad']),
                            estado_inventario="Bajo"
                        ))
        
        return {
            "total_productos": len(variedades_df),
            "distribucion": distribucion,
            "productos_criticos": criticos[:limit],
            "productos_bajos": bajos[:limit]
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en análisis simplificado: {str(e)}"
        )


@router.get("/critical", response_model=List[InventoryNewStatus])
async def get_critical_products(
    data: Dict[str, pd.DataFrame] = Depends(get_db_data),
    limit: int = Query(20, description="Máximo número de productos"), # Reducido a 20 para evitar timeouts
    offset: int = Query(0, description="Número de productos a saltar")
):
    """
    Obtiene la lista de productos en estado crítico (versión optimizada)
    """
    try:
        # Clave de caché
        cache_key = f"critical_limit_{limit}_offset_{offset}"
        
        # Recuperar análisis de cache o calcular
        analysis_result = await get_cached_or_compute(
            cache_key,
            InventoryService.analyze_inventory,
            data
        )
        
        if not analysis_result["exito"]:
            # Intentar enfoque simplificado
            productos_df = data.get("productos", pd.DataFrame())
            variedades_df = data.get("producto_variedads", pd.DataFrame())
            
            if 'cantidad' in variedades_df.columns:
                # Productos sin stock
                productos_sin_stock = variedades_df[variedades_df['cantidad'] <= 0].iloc[offset:offset+limit]
                
                # Crear respuesta simplificada
                results = []
                for _, var in productos_sin_stock.iterrows():
                    results.append(InventoryStatus(
                        producto_id=str(var.get('producto', '')),
                        variedad_id=str(var.get('_id', '')),
                        nombre_completo="Producto " + str(var.get('_id', '')),
                        inventario_actual=0,
                        estado_inventario="Crítico"
                    ))
                
                return results
            
            raise HTTPException(
                status_code=500,
                detail=analysis_result["error"]
            )
        
        # Obtener productos críticos
        productos_criticos = analysis_result["productos_criticos"]
        
        # Aplicar paginación con límite para evitar sobrecarga
        max_limit = min(limit, 50)  # Limitar a 50 máximo
        paginados = productos_criticos.iloc[offset:offset+max_limit]
        
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
    limit: int = Query(10, description="Máximo número de recomendaciones"), # Reducido a 10 para evitar timeouts
    min_days: int = Query(14, description="Días mínimos de stock objetivo")
):
    """
    Genera recomendaciones de reabastecimiento (versión optimizada)
    """
    try:
        # Clave de caché
        cache_key = f"recommendations_limit_{limit}_min_days_{min_days}"
        
        # Recuperar de cache o calcular
        recommendations = await get_cached_or_compute(
            cache_key,
            InventoryService.get_recommendations,
            data, min_days
        )
        
        # Limitar número para evitar timeout
        max_limit = min(limit, 20)
        limited_recommendations = recommendations[:max_limit]
        
        # Convertir a objetos Pydantic
        recommendation_objects = []
        for rec in limited_recommendations:
            recommendation_objects.append(InventoryRecommendation(**rec))
        
        return recommendation_objects
    
    except Exception as e:
        try:
            return await get_simple_recommendations(data, limit, min_days)
        except:
            raise HTTPException(
                status_code=500,
                detail=f"Error al generar recomendaciones: {str(e)}"
            )


# Endpoint alternativo para recomendaciones simples
async def get_simple_recommendations(
    data: Dict[str, pd.DataFrame],
    limit: int = 10,
    min_days: int = 14
):
    """Versión ultra-simplificada para recomendaciones"""
    
    try:
        variedades_df = data.get("producto_variedads", pd.DataFrame())
        productos_df = data.get("productos", pd.DataFrame())
        
        if variedades_df.empty:
            return []
        
        # Crear diccionario de nombres de productos
        producto_nombres = {}
        if not productos_df.empty:
            for _, row in productos_df.iterrows():
                if '_id' in row and 'titulo' in row:
                    producto_nombres[str(row['_id'])] = row['titulo']
        
        recomendaciones = []
        
        # Buscar productos con poco stock directamente
        if 'cantidad' in variedades_df.columns and 'producto' in variedades_df.columns:
            bajo_stock = variedades_df[variedades_df['cantidad'] <= 3].head(limit)
            
            for _, var in bajo_stock.iterrows():
                producto_id = str(var.get('producto', ''))
                nombre = producto_nombres.get(producto_id, f"Producto {producto_id}")
                
                recomendaciones.append(InventoryRecommendation(
                    variedad_id=str(var.get('_id', '')),
                    producto_id=producto_id,
                    nombre_completo=nombre,
                    inventario_actual=int(var.get('cantidad', 0)),
                    cantidad_recomendada=5,  # Valor fijo para simplificar
                    dias_stock_actual=0,
                    dias_stock_objetivo=min_days,
                    prioridad=1
                ))
        
        return recomendaciones
        
    except Exception as e:
        print(f"Error en recomendaciones simples: {str(e)}")
        return []