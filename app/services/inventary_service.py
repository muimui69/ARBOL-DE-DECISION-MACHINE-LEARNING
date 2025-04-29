# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from typing import Dict, Any, List, Optional
# import logging

# from app.core.config import settings

# logger = logging.getLogger(__name__)

# class InventoryService:
#     """Servicio para análisis y recomendaciones de inventario utilizando la misma lógica del modelo"""
    
#     # Constantes para umbrales de inventario (igual que el modelo)
#     CRITICAL_THRESHOLD = 7
#     LOW_THRESHOLD = 15
#     NORMAL_THRESHOLD = 30
#     HIGH_THRESHOLD = 60
    
#     @classmethod
#     async def process_data(cls, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
#         """
#         Procesa los datos de MongoDB con los nombres de colección correctos
#         """
#         try:
#             # Verificar los nombres de las colecciones disponibles
#             print(f"Colecciones recibidas: {list(data.keys())}")
            
#             # Extraer con los nombres correctos de colecciones
#             ventas_df = data.get('ventas', pd.DataFrame())
#             ventadetalles_df = data.get('ventadetalles', pd.DataFrame())
#             productos_df = data.get('productos', pd.DataFrame())
#             producto_variedads_df = data.get('producto_variedads', pd.DataFrame())
#             ingresodetalles_df = data.get('ingresodetalles', pd.DataFrame())
            
#             # Verificar datos mínimos
#             if productos_df.empty:
#                 logger.warning("No hay datos de productos para procesar")
#                 return {}
            
#             if producto_variedads_df.empty:
#                 logger.warning("No hay datos de variedades para procesar")
#                 return {}
            
#             # Convertir IDs a strings
#             for df, name in [
#                 (ventas_df, 'ventas'),
#                 (ventadetalles_df, 'ventadetalles'),
#                 (productos_df, 'productos'),
#                 (producto_variedads_df, 'producto_variedads'),
#                 (ingresodetalles_df, 'ingresodetalles')
#             ]:
#                 if not df.empty and '_id' in df.columns:
#                     logger.info(f"Convirtiendo IDs a string en {name}")
#                     df['_id'] = df['_id'].astype(str)
            
#             # Convertir columnas relacionales a string
#             if not ventadetalles_df.empty:
#                 for col in ['producto', 'variedad', 'venta']:
#                     if col in ventadetalles_df.columns:
#                         ventadetalles_df[col] = ventadetalles_df[col].astype(str)
            
#             if not producto_variedads_df.empty and 'producto' in producto_variedads_df.columns:
#                 producto_variedads_df['producto'] = producto_variedads_df['producto'].astype(str)
            
#             if not ingresodetalles_df.empty:
#                 for col in ['producto_variedad', 'producto']:
#                     if col in ingresodetalles_df.columns:
#                         ingresodetalles_df[col] = ingresodetalles_df[col].astype(str)
            
#             # Calcular stock real basado en ingresos
#             if not ingresodetalles_df.empty and 'producto_variedad' in ingresodetalles_df.columns and 'estado' in ingresodetalles_df.columns:
#                 logger.info("Calculando stock real a partir de ingresodetalles")
#                 # Solo ingresos activos (estado=true)
#                 ingresos_activos = ingresodetalles_df[ingresodetalles_df['estado'] == True]
#                 # Agrupar por variedad
#                 stock_df = ingresos_activos.groupby('producto_variedad').size().reset_index()
#                 stock_df.columns = ['producto_variedad', 'stock_real']
                
#                 # Unir con variedades
#                 producto_variedads_df = pd.merge(
#                     producto_variedads_df,
#                     stock_df,
#                     left_on='_id', 
#                     right_on='producto_variedad',
#                     how='left'
#                 )
                
#                 # Rellenar valores nulos con 0
#                 producto_variedads_df['stock_real'] = producto_variedads_df['stock_real'].fillna(0)
#             else:
#                 # Si no hay datos de ingresos, usar cantidad como stock
#                 if 'cantidad' in producto_variedads_df.columns:
#                     producto_variedads_df['stock_real'] = producto_variedads_df['cantidad']
#                 else:
#                     producto_variedads_df['stock_real'] = 0
#                     logger.warning("No hay información de stock, usando 0 como predeterminado")
            
#             # Convertir fechas de ventas
#             if not ventas_df.empty and 'createdAT' in ventas_df.columns:
#                 ventas_df['fecha'] = pd.to_datetime(ventas_df['createdAT'])
            
#             # Añadir fecha a ventadetalles
#             if not ventadetalles_df.empty and not ventas_df.empty:
#                 if 'createdAT' not in ventadetalles_df.columns and 'venta' in ventadetalles_df.columns:
#                     # Crear diccionario de ID venta -> fecha
#                     venta_fechas = dict(zip(ventas_df['_id'], ventas_df['fecha']))
#                     # Mapear fechas a detalles
#                     ventadetalles_df['fecha'] = ventadetalles_df['venta'].map(venta_fechas)
#                 elif 'createdAT' in ventadetalles_df.columns:
#                     ventadetalles_df['fecha'] = pd.to_datetime(ventadetalles_df['createdAT'])
            
#             # Añadir nombres de productos a variedades
#             if not productos_df.empty and not producto_variedads_df.empty:
#                 if 'titulo' in productos_df.columns and 'producto' in producto_variedads_df.columns:
#                     # Diccionario id -> nombre
#                     producto_nombres = dict(zip(productos_df['_id'], productos_df['titulo']))
#                     # Mapear nombres
#                     producto_variedads_df['nombre_producto'] = producto_variedads_df['producto'].map(producto_nombres)
            
#             return {
#                 'ventas': ventas_df,
#                 'ventadetalles': ventadetalles_df,
#                 'productos': productos_df,
#                 'variedades': producto_variedads_df,
#                 'ingresodetalles': ingresodetalles_df
#             }
        
#         except Exception as e:
#             logger.error(f"Error al procesar datos: {e}")
#             import traceback
#             traceback.print_exc()
#             return {}

#     @classmethod
#     async def generate_features(cls, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
#         """
#         Genera características para análisis de inventario
#         """
#         try:
#             if not data:
#                 logger.warning("No hay datos para generar características")
#                 return pd.DataFrame()
            
#             ventas_df = data.get('ventas', pd.DataFrame())
#             ventadetalles_df = data.get('ventadetalles', pd.DataFrame())
#             productos_df = data.get('productos', pd.DataFrame())
#             variedades_df = data.get('variedades', pd.DataFrame())
            
#             # Si no hay datos de variedades, salir
#             if variedades_df.empty:
#                 logger.warning("No hay datos de variedades para análisis")
#                 return pd.DataFrame()
            
#             # Lista para almacenar características
#             productos_features = []
            
#             # Obtener fecha actual o fecha más reciente de ventas
#             if not ventas_df.empty and 'fecha' in ventas_df.columns:
#                 fecha_actual = ventas_df['fecha'].max()
#             else:
#                 fecha_actual = datetime.now()
            
#             # Si hay datos de ventas, calcular estadísticas
#             if not ventadetalles_df.empty and 'variedad' in ventadetalles_df.columns and 'fecha' in ventadetalles_df.columns:
#                 # Agrupar ventas por variedad y fecha
#                 ventas_diarias = ventadetalles_df.groupby(
#                     ['variedad', pd.Grouper(key='fecha', freq='D')]
#                 )['cantidad'].sum().reset_index()
#             else:
#                 # Si no hay ventas, crear DataFrame vacío
#                 logger.warning("No hay datos de ventas para análisis detallado")
#                 ventas_diarias = pd.DataFrame(columns=['variedad', 'fecha', 'cantidad'])
            
#             # Para cada variedad de producto
#             for _, variedad in variedades_df.iterrows():
#                 variedad_id = variedad['_id']
                
#                 # Stock actual
#                 stock_actual = variedad['stock_real'] if 'stock_real' in variedad else 0
#                 if pd.isna(stock_actual):
#                     stock_actual = 0
                
#                 # Datos básicos
#                 nombre_producto = variedad.get('nombre_producto', 'Sin nombre')
#                 color = variedad.get('color', '')
#                 talla = variedad.get('talla', '')
#                 nombre_completo = f"{nombre_producto}" + (f" ({color}/{talla})" if color or talla else "")
                
#                 # Filtrar ventas de esta variedad
#                 ventas_variedad = ventas_diarias[ventas_diarias['variedad'] == variedad_id]
                
#                 # Si hay ventas, calcular estadísticas
#                 if len(ventas_variedad) > 0:
#                     # Ventas en períodos
#                     ventas_30d = ventas_variedad[
#                         ventas_variedad['fecha'] >= (fecha_actual - timedelta(days=30))
#                     ]['cantidad'].sum()
                    
#                     ventas_90d = ventas_variedad[
#                         ventas_variedad['fecha'] >= (fecha_actual - timedelta(days=90))
#                     ]['cantidad'].sum()
                    
#                     # Tendencia
#                     ventas_30_60d = ventas_variedad[
#                         (ventas_variedad['fecha'] >= (fecha_actual - timedelta(days=60))) &
#                         (ventas_variedad['fecha'] < (fecha_actual - timedelta(days=30)))
#                     ]['cantidad'].sum()
                    
#                     if ventas_30_60d > 0:
#                         tendencia = (ventas_30d - ventas_30_60d) / ventas_30_60d
#                     else:
#                         tendencia = 0
                    
#                     # Ventas diarias promedio
#                     ventas_diarias_prom = ventas_30d / 30 if ventas_30d > 0 else 0.1
                    
#                     # Días de stock restante
#                     dias_stock_restante = stock_actual / ventas_diarias_prom if ventas_diarias_prom > 0 else 100
                    
#                     # Variabilidad
#                     if len(ventas_variedad) > 1 and ventas_variedad['cantidad'].mean() > 0:
#                         coef_variacion = ventas_variedad['cantidad'].std() / ventas_variedad['cantidad'].mean()
#                     else:
#                         coef_variacion = 0
#                 else:
#                     # Sin ventas, valores predeterminados
#                     ventas_30d = 0
#                     ventas_90d = 0
#                     ventas_diarias_prom = 0.1
#                     dias_stock_restante = 0 if stock_actual == 0 else 100
#                     tendencia = 0
#                     coef_variacion = 0
                
#                 # Clasificar estado de inventario
#                 if dias_stock_restante < cls.CRITICAL_THRESHOLD or stock_actual <= 0:
#                     estado_inventario = 'Crítico'
#                 elif dias_stock_restante < cls.LOW_THRESHOLD:
#                     estado_inventario = 'Bajo'
#                 elif dias_stock_restante < cls.NORMAL_THRESHOLD:
#                     estado_inventario = 'Normal'
#                 elif dias_stock_restante < cls.HIGH_THRESHOLD:
#                     estado_inventario = 'Alto'
#                 else:
#                     estado_inventario = 'Exceso'
                
#                 # Guardar características
#                 productos_features.append({
#                     'variedad_id': variedad_id,
#                     'producto_id': variedad.get('producto', ''),
#                     'nombre_completo': nombre_completo,
#                     'color': color,
#                     'talla': talla,
#                     'inventario_actual': stock_actual,
#                     'precio': variedad.get('precio', 0),
#                     'ventas_30d': ventas_30d,
#                     'ventas_90d': ventas_90d,
#                     'ventas_diarias_prom': ventas_diarias_prom,
#                     'dias_stock_restante': dias_stock_restante,
#                     'tendencia': tendencia,
#                     'coef_variacion': coef_variacion,
#                     'estado_inventario': estado_inventario
#                 })
            
#             # Crear DataFrame con características
#             productos_features_df = pd.DataFrame(productos_features)
#             logger.info(f"Se generaron características para {len(productos_features_df)} productos")
#             return productos_features_df
        
#         except Exception as e:
#             logger.error(f"Error al generar características: {e}")
#             import traceback
#             traceback.print_exc()
#             return pd.DataFrame()
    
#     @classmethod
#     async def analyze_inventory(cls, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
#         """
#         Realiza un análisis completo del inventario
#         """
#         try:
#             # Procesar datos
#             processed_data = await cls.process_data(data)
            
#             if not processed_data:
#                 logger.error("No hay datos procesados para análisis de inventario")
#                 return {
#                     "exito": False,
#                     "mensaje": "No hay datos suficientes para análisis",
#                     "error": "Datos procesados insuficientes"
#                 }
            
#             # Generar características
#             features_df = await cls.generate_features(processed_data)
            
#             if features_df.empty:
#                 logger.error("No se generaron características para análisis")
#                 return {
#                     "exito": False,
#                     "mensaje": "No se pudieron generar características",
#                     "error": "DataFrame de características vacío"
#                 }
            
#             # Análisis por estado
#             distribucion = features_df['estado_inventario'].value_counts().to_dict()
            
#             # Productos por categoría
#             productos_criticos = features_df[features_df['estado_inventario'] == 'Crítico']
#             productos_bajos = features_df[features_df['estado_inventario'] == 'Bajo']
            
#             # Ordenar por días de stock (menor a mayor)
#             if not productos_criticos.empty and 'dias_stock_restante' in productos_criticos.columns:
#                 productos_criticos = productos_criticos.sort_values('dias_stock_restante')
                
#             if not productos_bajos.empty and 'dias_stock_restante' in productos_bajos.columns:
#                 productos_bajos = productos_bajos.sort_values('dias_stock_restante')
            
#             return {
#                 "exito": True,
#                 "total_productos": len(features_df),
#                 "distribucion": distribucion,
#                 "productos_criticos": productos_criticos,
#                 "productos_bajos": productos_bajos
#             }
        
#         except Exception as e:
#             logger.error(f"Error en análisis de inventario: {e}")
#             import traceback
#             traceback.print_exc()
#             return {
#                 "exito": False,
#                 "mensaje": "Error al analizar inventario",
#                 "error": str(e)
#             }
    
#     @classmethod
#     async def get_recommendations(cls, data: Dict[str, pd.DataFrame], min_days: int = 14) -> List[Dict[str, Any]]:
#         """
#         Genera recomendaciones de reabastecimiento
#         """
#         try:
#             # Analizar inventario
#             analysis = await cls.analyze_inventory(data)
            
#             if not analysis["exito"]:
#                 logger.error(f"No se pudo analizar el inventario: {analysis.get('error')}")
#                 return []
            
#             # Obtener productos críticos y bajos
#             productos_criticos = analysis["productos_criticos"]
#             productos_bajos = analysis["productos_bajos"]
            
#             # Generar recomendaciones
#             recommendations = []
            
#             # Procesar productos críticos
#             for _, row in productos_criticos.iterrows():
#                 # Calcular cantidad recomendada
#                 dias_actuales = float(row['dias_stock_restante']) if pd.notna(row['dias_stock_restante']) else 0
#                 dias_faltantes = max(0, min_days - dias_actuales)
#                 ventas_diarias = float(row['ventas_diarias_prom']) if pd.notna(row['ventas_diarias_prom']) else 0.1
                
#                 # Cantidad mínima 1 unidad
#                 cantidad_recomendada = max(1, int(round(dias_faltantes * ventas_diarias)))
                
#                 recommendations.append({
#                     "variedad_id": str(row['variedad_id']),
#                     "producto_id": str(row['producto_id']),
#                     "nombre_completo": str(row['nombre_completo']),
#                     "inventario_actual": int(row['inventario_actual']) if pd.notna(row['inventario_actual']) else 0,
#                     "cantidad_recomendada": cantidad_recomendada,
#                     "dias_stock_actual": dias_actuales,
#                     "dias_stock_objetivo": min_days,
#                     "prioridad": 1  # Crítico = prioridad 1
#                 })
            
#             # Procesar productos bajos
#             for _, row in productos_bajos.iterrows():
#                 # Calcular cantidad recomendada
#                 dias_actuales = float(row['dias_stock_restante']) if pd.notna(row['dias_stock_restante']) else 0
#                 dias_faltantes = max(0, min_days - dias_actuales)
#                 ventas_diarias = float(row['ventas_diarias_prom']) if pd.notna(row['ventas_diarias_prom']) else 0.1
                
#                 # Cantidad mínima 1 unidad
#                 cantidad_recomendada = max(1, int(round(dias_faltantes * ventas_diarias)))
                
#                 recommendations.append({
#                     "variedad_id": str(row['variedad_id']),
#                     "producto_id": str(row['producto_id']),
#                     "nombre_completo": str(row['nombre_completo']),
#                     "inventario_actual": int(row['inventario_actual']) if pd.notna(row['inventario_actual']) else 0,
#                     "cantidad_recomendada": cantidad_recomendada,
#                     "dias_stock_actual": dias_actuales,
#                     "dias_stock_objetivo": min_days,
#                     "prioridad": 2  # Bajo = prioridad 2
#                 })
            
#             # Ordenar por prioridad y luego por días de stock
#             recommendations.sort(key=lambda x: (x["prioridad"], x["dias_stock_actual"]))
#             return recommendations
            
#         except Exception as e:
#             logger.error(f"Error al generar recomendaciones: {e}")
#             import traceback
#             traceback.print_exc()
#             return []



import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from app.core.config import settings

logger = logging.getLogger(__name__)

class InventoryService:
    # Constantes previas...
    
    @classmethod
    async def process_data(cls, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        # El código existente es adecuado para este método
        # pues es principalmente preparación de datos
        pass
    
    @classmethod
    async def _process_variedad(cls, variedad, fecha_actual, ventas_diarias):
        """Procesa una variedad individual (para paralelización)"""
        try:
            variedad_id = variedad['_id']
            
            # Stock actual
            stock_actual = variedad['stock_real'] if 'stock_real' in variedad else 0
            if pd.isna(stock_actual):
                stock_actual = 0
            
            # Datos básicos
            nombre_producto = variedad.get('nombre_producto', 'Sin nombre')
            color = variedad.get('color', '')
            talla = variedad.get('talla', '')
            nombre_completo = f"{nombre_producto}" + (f" ({color}/{talla})" if color or talla else "")
            
            # Filtrar ventas de esta variedad
            ventas_variedad = ventas_diarias[ventas_diarias['variedad'] == variedad_id]
            
            # Si hay ventas, calcular estadísticas
            if len(ventas_variedad) > 0:
                # Ventas en períodos
                ventas_30d = ventas_variedad[
                    ventas_variedad['fecha'] >= (fecha_actual - timedelta(days=30))
                ]['cantidad'].sum()
                
                ventas_90d = ventas_variedad[
                    ventas_variedad['fecha'] >= (fecha_actual - timedelta(days=90))
                ]['cantidad'].sum()
                
                # Tendencia
                ventas_30_60d = ventas_variedad[
                    (ventas_variedad['fecha'] >= (fecha_actual - timedelta(days=60))) &
                    (ventas_variedad['fecha'] < (fecha_actual - timedelta(days=30)))
                ]['cantidad'].sum()
                
                if ventas_30_60d > 0:
                    tendencia = (ventas_30d - ventas_30_60d) / ventas_30_60d
                else:
                    tendencia = 0
                
                # Ventas diarias promedio
                ventas_diarias_prom = ventas_30d / 30 if ventas_30d > 0 else 0.1
                
                # Días de stock restante
                dias_stock_restante = stock_actual / ventas_diarias_prom if ventas_diarias_prom > 0 else 100
                
                # Variabilidad
                if len(ventas_variedad) > 1 and ventas_variedad['cantidad'].mean() > 0:
                    coef_variacion = ventas_variedad['cantidad'].std() / ventas_variedad['cantidad'].mean()
                else:
                    coef_variacion = 0
            else:
                # Sin ventas, valores predeterminados
                ventas_30d = 0
                ventas_90d = 0
                ventas_diarias_prom = 0.1
                dias_stock_restante = 0 if stock_actual == 0 else 100
                tendencia = 0
                coef_variacion = 0
            
            # Clasificar estado de inventario
            if dias_stock_restante < cls.CRITICAL_THRESHOLD or stock_actual <= 0:
                estado_inventario = 'Crítico'
            elif dias_stock_restante < cls.LOW_THRESHOLD:
                estado_inventario = 'Bajo'
            elif dias_stock_restante < cls.NORMAL_THRESHOLD:
                estado_inventario = 'Normal'
            elif dias_stock_restante < cls.HIGH_THRESHOLD:
                estado_inventario = 'Alto'
            else:
                estado_inventario = 'Exceso'
            
            # Devolver características
            return {
                'variedad_id': variedad_id,
                'producto_id': variedad.get('producto', ''),
                'nombre_completo': nombre_completo,
                'color': color,
                'talla': talla,
                'inventario_actual': stock_actual,
                'precio': variedad.get('precio', 0),
                'ventas_30d': ventas_30d,
                'ventas_90d': ventas_90d,
                'ventas_diarias_prom': ventas_diarias_prom,
                'dias_stock_restante': dias_stock_restante,
                'tendencia': tendencia,
                'coef_variacion': coef_variacion,
                'estado_inventario': estado_inventario
            }
        except Exception as e:
            logger.error(f"Error procesando variedad {variedad.get('_id')}: {e}")
            return None

    @classmethod
    async def generate_features(cls, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Genera características para análisis de inventario con paralelización
        """
        try:
            if not data:
                logger.warning("No hay datos para generar características")
                return pd.DataFrame()
            
            ventas_df = data.get('ventas', pd.DataFrame())
            ventadetalles_df = data.get('ventadetalles', pd.DataFrame())
            productos_df = data.get('productos', pd.DataFrame())
            variedades_df = data.get('variedades', pd.DataFrame())
            
            # Si no hay datos de variedades, salir
            if variedades_df.empty:
                logger.warning("No hay datos de variedades para análisis")
                return pd.DataFrame()
            
            # Lista para almacenar características
            productos_features = []
            
            # Obtener fecha actual o fecha más reciente de ventas
            if not ventas_df.empty and 'fecha' in ventas_df.columns:
                fecha_actual = ventas_df['fecha'].max()
            else:
                fecha_actual = datetime.now()
            
            # Si hay datos de ventas, calcular estadísticas
            if not ventadetalles_df.empty and 'variedad' in ventadetalles_df.columns and 'fecha' in ventadetalles_df.columns:
                # Agrupar ventas por variedad y fecha - esta operación es costosa pero necesaria
                ventas_diarias = ventadetalles_df.groupby(
                    ['variedad', pd.Grouper(key='fecha', freq='D')]
                )['cantidad'].sum().reset_index()
            else:
                # Si no hay ventas, crear DataFrame vacío
                logger.warning("No hay datos de ventas para análisis detallado")
                ventas_diarias = pd.DataFrame(columns=['variedad', 'fecha', 'cantidad'])
            
            # Número de hilos a utilizar (ajustar según tu servidor)
            num_workers = min(8, max(4, len(variedades_df) // 200))
            logger.info(f"Usando {num_workers} hilos para procesar {len(variedades_df)} variedades")
            
            # Procesar variedades en paralelo con ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Lista para almacenar las tareas futuras
                futures = []
                
                # Enviar trabajo a hilos en lotes para evitar sobrecarga
                batch_size = 100  # Procesar 100 variedades por lote
                for i in range(0, len(variedades_df), batch_size):
                    batch = variedades_df.iloc[i:i+batch_size]
                    
                    for _, variedad in batch.iterrows():
                        # Crear tarea para cada variedad
                        future = executor.submit(
                            asyncio.run,
                            cls._process_variedad(variedad, fecha_actual, ventas_diarias)
                        )
                        futures.append(future)
                
                # Recoger resultados a medida que terminan
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        productos_features.append(result)
            
            # Crear DataFrame con características
            productos_features_df = pd.DataFrame(productos_features)
            logger.info(f"Se generaron características para {len(productos_features_df)} productos")
            return productos_features_df
        
        except Exception as e:
            logger.error(f"Error al generar características: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    @classmethod
    async def analyze_inventory(cls, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Realiza un análisis completo del inventario
        """
        try:
            # Procesar datos
            processed_data = await cls.process_data(data)
            
            if not processed_data:
                logger.error("No hay datos procesados para análisis de inventario")
                return {
                    "exito": False,
                    "mensaje": "No hay datos suficientes para análisis",
                    "error": "Datos procesados insuficientes"
                }
            
            # Generar características (ahora paralelo)
            features_df = await cls.generate_features(processed_data)
            
            if features_df.empty:
                logger.error("No se generaron características para análisis")
                return {
                    "exito": False,
                    "mensaje": "No se pudieron generar características",
                    "error": "DataFrame de características vacío"
                }
            
            # Análisis por estado
            distribucion = features_df['estado_inventario'].value_counts().to_dict()
            
            # Productos por categoría
            productos_criticos = features_df[features_df['estado_inventario'] == 'Crítico']
            productos_bajos = features_df[features_df['estado_inventario'] == 'Bajo']
            
            # Ordenar por días de stock (menor a mayor)
            if not productos_criticos.empty and 'dias_stock_restante' in productos_criticos.columns:
                productos_criticos = productos_criticos.sort_values('dias_stock_restante')
                
            if not productos_bajos.empty and 'dias_stock_restante' in productos_bajos.columns:
                productos_bajos = productos_bajos.sort_values('dias_stock_restante')
            
            return {
                "exito": True,
                "total_productos": len(features_df),
                "distribucion": distribucion,
                "productos_criticos": productos_criticos,
                "productos_bajos": productos_bajos
            }
        
        except Exception as e:
            logger.error(f"Error en análisis de inventario: {e}")
            import traceback
            traceback.print_exc()
            return {
                "exito": False,
                "mensaje": "Error al analizar inventario",
                "error": str(e)
            }
    
    @classmethod
    async def get_recommendations(cls, data: Dict[str, pd.DataFrame], min_days: int = 14) -> List[Dict[str, Any]]:
        """
        Genera recomendaciones de reabastecimiento
        """
        try:
            # Analizar inventario
            analysis = await cls.analyze_inventory(data)
            
            if not analysis["exito"]:
                logger.error(f"No se pudo analizar el inventario: {analysis.get('error')}")
                return []
            
            # Obtener productos críticos y bajos
            productos_criticos = analysis["productos_criticos"]
            productos_bajos = analysis["productos_bajos"]
            
            # Generar recomendaciones
            recommendations = []
            
            # Procesar productos críticos
            for _, row in productos_criticos.iterrows():
                # Calcular cantidad recomendada
                dias_actuales = float(row['dias_stock_restante']) if pd.notna(row['dias_stock_restante']) else 0
                dias_faltantes = max(0, min_days - dias_actuales)
                ventas_diarias = float(row['ventas_diarias_prom']) if pd.notna(row['ventas_diarias_prom']) else 0.1
                
                # Cantidad mínima 1 unidad
                cantidad_recomendada = max(1, int(round(dias_faltantes * ventas_diarias)))
                
                recommendations.append({
                    "variedad_id": str(row['variedad_id']),
                    "producto_id": str(row['producto_id']),
                    "nombre_completo": str(row['nombre_completo']),
                    "inventario_actual": int(row['inventario_actual']) if pd.notna(row['inventario_actual']) else 0,
                    "cantidad_recomendada": cantidad_recomendada,
                    "dias_stock_actual": dias_actuales,
                    "dias_stock_objetivo": min_days,
                    "prioridad": 1  # Crítico = prioridad 1
                })
            
            # Procesar productos bajos
            for _, row in productos_bajos.iterrows():
                # Calcular cantidad recomendada
                dias_actuales = float(row['dias_stock_restante']) if pd.notna(row['dias_stock_restante']) else 0
                dias_faltantes = max(0, min_days - dias_actuales)
                ventas_diarias = float(row['ventas_diarias_prom']) if pd.notna(row['ventas_diarias_prom']) else 0.1
                
                # Cantidad mínima 1 unidad
                cantidad_recomendada = max(1, int(round(dias_faltantes * ventas_diarias)))
                
                recommendations.append({
                    "variedad_id": str(row['variedad_id']),
                    "producto_id": str(row['producto_id']),
                    "nombre_completo": str(row['nombre_completo']),
                    "inventario_actual": int(row['inventario_actual']) if pd.notna(row['inventario_actual']) else 0,
                    "cantidad_recomendada": cantidad_recomendada,
                    "dias_stock_actual": dias_actuales,
                    "dias_stock_objetivo": min_days,
                    "prioridad": 2  # Bajo = prioridad 2
                })
            
            # Ordenar por prioridad y luego por días de stock
            recommendations.sort(key=lambda x: (x["prioridad"], x["dias_stock_actual"]))
            return recommendations
            
        except Exception as e:
            logger.error(f"Error al generar recomendaciones: {e}")
            import traceback
            traceback.print_exc()
            return []