import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CycleService:
    """
    Servicio para analizar diferentes tipos de ciclos en el inventario y ventas
    """
    @classmethod
    async def analyze_temporal_cycles_simple(cls, 
                                        data: Dict[str, pd.DataFrame], 
                                        cycle_type: str) -> Dict[str, Any]:
        """
        Análisis temporal simplificado sin joins complejos
        
        Parameters:
        - data: Diccionario con DataFrames de las colecciones MongoDB
        - cycle_type: Tipo de ciclo (weekly, monthly, seasonal, yearly)
        
        Returns:
        - Diccionario con resultados del análisis
        """
        try:
            # Obtener solo ventas para análisis
            ventas_df = data.get("ventas", pd.DataFrame())
            
            if ventas_df.empty:
                return {"exito": False, "mensaje": "No hay ventas para analizar"}
                
            # Convertir fecha
            ventas_df['fecha'] = pd.to_datetime(ventas_df['createdAT'])
            
            # Análisis según tipo de ciclo
            if cycle_type == 'weekly':
                ventas_df['dia_semana'] = ventas_df['fecha'].dt.day_name()
                
                # Asignar días en español
                dias_mapping = {
                    'Monday': 'Lunes',
                    'Tuesday': 'Martes',
                    'Wednesday': 'Miércoles',
                    'Thursday': 'Jueves',
                    'Friday': 'Viernes',
                    'Saturday': 'Sábado',
                    'Sunday': 'Domingo'
                }
                ventas_df['dia_semana'] = ventas_df['dia_semana'].map(dias_mapping)
                
                # Análisis por día de semana
                result = ventas_df.groupby('dia_semana').agg({
                    'total': ['sum', 'mean', 'count']
                })
                
                # Formatear resultado
                formatted_result = []
                for dia, data in result.iterrows():
                    formatted_result.append({
                        "dia": dia,
                        "total_ventas": float(data[('total', 'sum')]),
                        "promedio_venta": float(data[('total', 'mean')]),
                        "cantidad_ventas": int(data[('total', 'count')])
                    })
                
                # Ordenar los días correctamente
                dia_orden = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
                formatted_result = sorted(formatted_result, key=lambda x: dia_orden.index(x["dia"]) if x["dia"] in dia_orden else 999)
                    
                return {
                    "exito": True,
                    "tipo_ciclo": "weekly",
                    "datos": formatted_result
                }
                
            elif cycle_type == 'monthly':
                # Análisis por día del mes
                ventas_df['dia_mes'] = ventas_df['fecha'].dt.day
                
                # Agrupar por día del mes
                result = ventas_df.groupby('dia_mes').agg({
                    'total': ['sum', 'mean', 'count']
                })
                
                # Formatear resultado
                formatted_result = []
                for dia, data in result.iterrows():
                    formatted_result.append({
                        "dia": int(dia),
                        "total_ventas": float(data[('total', 'sum')]),
                        "promedio_venta": float(data[('total', 'mean')]),
                        "cantidad_ventas": int(data[('total', 'count')])
                    })
                
                # Ordenar por día del mes
                formatted_result = sorted(formatted_result, key=lambda x: x["dia"])
                
                return {
                    "exito": True,
                    "tipo_ciclo": "monthly",
                    "datos": formatted_result
                }
                
            elif cycle_type == 'seasonal':
                # Análisis por temporada (trimestre)
                ventas_df['trimestre'] = ventas_df['fecha'].dt.quarter
                
                # Mapear trimestres a temporadas
                temporadas_mapping = {
                    1: 'Invierno',
                    2: 'Primavera',
                    3: 'Verano',
                    4: 'Otoño'
                }
                ventas_df['temporada'] = ventas_df['trimestre'].map(temporadas_mapping)
                
                # Agrupar por temporada
                result = ventas_df.groupby('temporada').agg({
                    'total': ['sum', 'mean', 'count']
                })
                
                # Formatear resultado
                formatted_result = []
                for temporada, data in result.iterrows():
                    formatted_result.append({
                        "temporada": temporada,
                        "total_ventas": float(data[('total', 'sum')]),
                        "promedio_venta": float(data[('total', 'mean')]),
                        "cantidad_ventas": int(data[('total', 'count')])
                    })
                
                # Ordenar por temporada natural
                temporada_orden = ['Invierno', 'Primavera', 'Verano', 'Otoño']
                formatted_result = sorted(formatted_result, 
                    key=lambda x: temporada_orden.index(x["temporada"]) if x["temporada"] in temporada_orden else 999)
                
                return {
                    "exito": True,
                    "tipo_ciclo": "seasonal",
                    "datos": formatted_result
                }
                
            elif cycle_type == 'yearly':
                # Análisis por mes del año
                ventas_df['mes_numero'] = ventas_df['fecha'].dt.month
                
                # Mapear números de mes a nombres
                meses_mapping = {
                    1: 'Enero',
                    2: 'Febrero',
                    3: 'Marzo',
                    4: 'Abril',
                    5: 'Mayo',
                    6: 'Junio',
                    7: 'Julio',
                    8: 'Agosto',
                    9: 'Septiembre',
                    10: 'Octubre',
                    11: 'Noviembre',
                    12: 'Diciembre'
                }
                ventas_df['mes'] = ventas_df['mes_numero'].map(meses_mapping)
                
                # Agrupar por mes
                result = ventas_df.groupby('mes').agg({
                    'total': ['sum', 'mean', 'count']
                })
                
                # Formatear resultado
                formatted_result = []
                for mes, data in result.iterrows():
                    formatted_result.append({
                        "mes": mes,
                        "total_ventas": float(data[('total', 'sum')]),
                        "promedio_venta": float(data[('total', 'mean')]),
                        "cantidad_ventas": int(data[('total', 'count')])
                    })
                
                # Ordenar por mes natural
                meses_orden = list(meses_mapping.values())
                formatted_result = sorted(formatted_result, 
                    key=lambda x: meses_orden.index(x["mes"]) if x["mes"] in meses_orden else 999)
                
                return {
                    "exito": True,
                    "tipo_ciclo": "yearly",
                    "datos": formatted_result
                }
            else:
                return {
                    "exito": False,
                    "mensaje": f"Tipo de ciclo no válido: {cycle_type}. Opciones válidas: weekly, monthly, seasonal, yearly"
                }
                
        except Exception as e:
            import traceback
            logger.error(f"Error en análisis temporal simplificado: {str(e)}", exc_info=True)
            return {
                "exito": False,
                "mensaje": f"Error en análisis: {str(e)}",
                "traceback": traceback.format_exc()
            }
        
    
    
    @classmethod
    async def analyze_product_cycles_simple(cls, 
                                        data: Dict[str, pd.DataFrame], 
                                        cycle_type: str, 
                                        product_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Análisis simplificado de ciclos de producto sin joins complejos
        """
        try:
            # Validar tipo de ciclo
            valid_cycle_types = ['lifecycle', 'replenishment', 'promotional']
            if cycle_type not in valid_cycle_types:
                return {
                    "exito": False,
                    "mensaje": f"Tipo de ciclo no válido: {cycle_type}",
                }
            
            # Dirigir al método especializado correspondiente
            if cycle_type == 'replenishment':
                return await cls.analyze_replenishment_cycle_simple(data, product_id)
            elif cycle_type == 'lifecycle':
                return await cls.analyze_lifecycle_cycle_simple(data, product_id)
            elif cycle_type == 'promotional':
                return await cls.analyze_promotional_cycle_simple(data, product_id)
                
            return {
                "exito": False,
                "mensaje": f"Análisis simplificado para {cycle_type} no implementado"
            }
            
        except Exception as e:
            logger.error(f"Error en análisis simplificado: {str(e)}", exc_info=True)
            return {
                "exito": False,
                "mensaje": f"Error: {str(e)}"
            }

    @classmethod
    async def analyze_replenishment_cycle_simple(cls, data: Dict[str, pd.DataFrame], product_id: Optional[str] = None):
        """Análisis simplificado de ciclos de reabastecimiento"""
        try:
            # Obtener datos
            ingresos_df = data.get("ingresodetalles", pd.DataFrame())
            productos_df = data.get("productos", pd.DataFrame())
            
            if ingresos_df.empty:
                return {
                    "exito": False,
                    "mensaje": "No hay datos de ingresos para analizar"
                }
            
            # Convertir IDs a string para asegurar compatibilidad
            ingresos_df['producto'] = ingresos_df['producto'].astype(str) if 'producto' in ingresos_df.columns else None
            
            # Filtrar por producto si se especifica
            if product_id:
                ingresos_df = ingresos_df[ingresos_df['producto'] == product_id]
                if ingresos_df.empty:
                    return {
                        "exito": False,
                        "mensaje": f"No hay registros de ingresos para el producto ID: {product_id}"
                    }
            
            # Convertir fechas
            ingresos_df['fecha'] = pd.to_datetime(ingresos_df['createdAT'])
            
            # Crear diccionario para mapear productos a nombres
            producto_nombres = {}
            if not productos_df.empty:
                for _, row in productos_df.iterrows():
                    producto_nombres[str(row['_id'])] = row['titulo'] if 'titulo' in row else f"Producto {row['_id']}"
            
            # Análisis de reabastecimiento por producto
            resultados = []
            
            # Agrupar por producto
            for producto_id, grupo in ingresos_df.groupby('producto'):
                # Necesitamos al menos 2 ingresos para calcular ciclo
                if len(grupo) < 2:
                    continue
                    
                # Ordenar por fecha
                grupo = grupo.sort_values('fecha')
                
                # Calcular días entre ingresos
                grupo['dias_desde_ultimo'] = grupo['fecha'].diff().dt.days
                
                # Calcular estadísticas básicas
                dias_promedio = grupo['dias_desde_ultimo'].mean()
                dias_std = grupo['dias_desde_ultimo'].std()
                ultimo_ingreso = grupo['fecha'].max()
                dias_desde_ultimo = (datetime.now() - ultimo_ingreso).days
                
                # Determinar fase de reabastecimiento
                if dias_desde_ultimo <= dias_promedio * 0.25:
                    fase = "Recién reabastecido"
                elif dias_desde_ultimo <= dias_promedio * 0.75:
                    fase = "Stock disponible"
                elif dias_desde_ultimo <= dias_promedio:
                    fase = "Próximo a reabastecimiento"
                else:
                    fase = "Reabastecimiento retrasado"
                
                # Obtener nombre del producto
                nombre = producto_nombres.get(producto_id, f"Producto {producto_id}")
                
                resultados.append({
                    "producto_id": producto_id,
                    "nombre": nombre,
                    "dias_promedio_reabastecimiento": round(dias_promedio, 1),
                    "desviacion_dias": round(dias_std, 1) if not pd.isna(dias_std) else 0,
                    "ultimo_ingreso": ultimo_ingreso.isoformat(),
                    "dias_desde_ultimo": dias_desde_ultimo,
                    "fase": fase,
                    "cantidad_ingresos": len(grupo)
                })
            
            # Agrupar por fase
            fases = {}
            todas_fases = ["Recién reabastecido", "Stock disponible", "Próximo a reabastecimiento", "Reabastecimiento retrasado"]
            
            for fase in todas_fases:
                productos_fase = [p for p in resultados if p["fase"] == fase]
                fases[fase] = {
                    "productos": [{"id": p["producto_id"], "nombre": p["nombre"]} for p in productos_fase],
                    "cantidad": len(productos_fase),
                    "porcentaje": round(len(productos_fase) / len(resultados) * 100 if resultados else 0, 2)
                }
            
            return {
                "exito": True,
                "mensaje": "Análisis de reabastecimiento simplificado completado",
                "tipo_ciclo": "replenishment",
                "producto_id": product_id if product_id else "Todos",
                "total_productos_analizados": len(resultados),
                "fecha_analisis": datetime.now().isoformat(),
                "fases_ciclo": [
                    {
                        "fase": fase,
                        "productos": fases[fase]["productos"][:10],
                        "cantidad_productos": fases[fase]["cantidad"],
                        "porcentaje": fases[fase]["porcentaje"]
                    } for fase in todas_fases
                ],
                "productos_detalle": sorted(resultados, key=lambda x: x["dias_desde_ultimo"], reverse=True)[:20]
            }
        
        except Exception as e:
            logger.error(f"Error en análisis de reabastecimiento simplificado: {str(e)}", exc_info=True)
            return {
                "exito": False,
                "mensaje": f"Error: {str(e)}"
            }
            
    @classmethod
    async def analyze_lifecycle_cycle_simple(cls, data: Dict[str, pd.DataFrame], product_id: Optional[str] = None):
        """Análisis simplificado del ciclo de vida de productos"""
        try:
            # Obtener datos
            ventas_df = data.get("ventas", pd.DataFrame())
            detalles_df = data.get("ventadetalles", pd.DataFrame())
            productos_df = data.get("productos", pd.DataFrame())
            
            if ventas_df.empty or detalles_df.empty:
                return {
                    "exito": False,
                    "mensaje": "No hay suficientes datos para análisis de ciclo de vida"
                }
            
            # Convertir IDs a string
            for df in [ventas_df, detalles_df, productos_df]:
                for col in df.columns:
                    if col.endswith('_id') or col == '_id' or col == 'venta' or col == 'producto':
                        df[col] = df[col].astype(str)
            
            # Crear diccionario de fechas de ventas
            fecha_ventas = {}
            for _, row in ventas_df.iterrows():
                fecha_ventas[str(row['_id'])] = row['createdAT']
                
            # Añadir columna de fecha a detalles
            detalles_df['fecha'] = detalles_df['venta'].map(fecha_ventas)
            detalles_df = detalles_df.dropna(subset=['fecha'])
            detalles_df['fecha'] = pd.to_datetime(detalles_df['fecha'])
            
            # Crear diccionario de nombres de productos
            nombres_productos = {}
            if not productos_df.empty:
                for _, row in productos_df.iterrows():
                    if 'titulo' in row:
                        nombres_productos[str(row['_id'])] = row['titulo']
                    elif 'nombre' in row:
                        nombres_productos[str(row['_id'])] = row['nombre']
                    else:
                        nombres_productos[str(row['_id'])] = f"Producto {row['_id']}"
                        
            # Filtrar por producto específico si se proporciona
            if product_id:
                detalles_df = detalles_df[detalles_df['producto'] == product_id]
                if detalles_df.empty:
                    return {
                        "exito": False,
                        "mensaje": f"No hay datos de ventas para el producto ID: {product_id}"
                    }
            
            # Procesar datos para análisis de ciclo de vida
            # Agregar mes como periodo para agrupar ventas mensuales
            detalles_df['mes'] = detalles_df['fecha'].dt.to_period('M')
            
            # Agrupar ventas por producto y mes
            producto_mes_ventas = detalles_df.groupby(['producto', 'mes']).agg({
                'cantidad': 'sum',
                'precio': 'sum',
                '_id': 'count'
            }).reset_index()
            
            # Análisis de tendencias por producto
            resultados = []
            
            for producto_id, grupo in producto_mes_ventas.groupby('producto'):
                if len(grupo) < 3:  # Necesitamos al menos 3 meses para analizar tendencia
                    continue
                    
                # Ordenar por mes
                grupo = grupo.sort_values('mes')
                
                # Calcular tendencia (pendiente de la regresión)
                x = np.array(range(len(grupo))).reshape(-1, 1)
                y = grupo['cantidad'].values
                
                try:
                    # Regresión lineal simple
                    pendiente = np.polyfit(x.flatten(), y, 1)[0]
                    
                    # Calcular coeficiente de variación
                    cv = grupo['cantidad'].std() / grupo['cantidad'].mean() if grupo['cantidad'].mean() > 0 else 0
                    
                    # Determinar etapa del ciclo de vida
                    if pendiente > 0.2:  # Crecimiento rápido
                        etapa = "Introducción" if cv > 0.5 else "Crecimiento"
                    elif pendiente > 0:  # Crecimiento lento
                        etapa = "Crecimiento" if cv > 0.3 else "Madurez"
                    elif pendiente > -0.1:  # Estable o ligero declive
                        etapa = "Madurez"
                    else:  # Declive significativo
                        etapa = "Declive"
                    
                    # Obtener nombre del producto
                    nombre = nombres_productos.get(producto_id, f"Producto {producto_id}")
                    
                    resultados.append({
                        "producto_id": producto_id,
                        "nombre": nombre,
                        "ventas_promedio_mensual": round(grupo['cantidad'].mean(), 2),
                        "tendencia": round(pendiente, 3),
                        "variabilidad": round(cv, 2),
                        "meses_analizados": len(grupo),
                        "primer_mes": str(grupo['mes'].min()),
                        "ultimo_mes": str(grupo['mes'].max()),
                        "etapa_ciclo": etapa
                    })
                    
                except Exception as e:
                    logger.warning(f"No se pudo analizar tendencia para producto {producto_id}: {str(e)}")
            
            # Agrupar resultados por etapa
            etapas = {}
            for etapa in ["Introducción", "Crecimiento", "Madurez", "Declive"]:
                productos_etapa = [p for p in resultados if p["etapa_ciclo"] == etapa]
                etapas[etapa] = {
                    "productos": [{"id": p["producto_id"], "nombre": p["nombre"], "tendencia": p["tendencia"]} 
                                for p in productos_etapa],
                    "cantidad": len(productos_etapa),
                    "porcentaje": round(len(productos_etapa) / len(resultados) * 100 if resultados else 0, 2)
                }
            
            # Generar recomendaciones
            recomendaciones = []
            
            # Para productos en introducción
            if etapas["Introducción"]["cantidad"] > 0:
                recomendaciones.append({
                    "tipo": "Productos en Introducción",
                    "accion": "Aumentar visibilidad y promoción",
                    "descripcion": "Los productos nuevos necesitan mayor exposición. Considere ubicaciones destacadas y campañas de lanzamiento.",
                    "productos_afectados": etapas["Introducción"]["cantidad"]
                })
                
            # Para productos en crecimiento
            if etapas["Crecimiento"]["cantidad"] > 0:
                recomendaciones.append({
                    "tipo": "Productos en Crecimiento",
                    "accion": "Optimizar inventario y considerar expansión",
                    "descripcion": "Estos productos tienen potencial de crecimiento. Asegure stock suficiente y considere extender líneas de producto.",
                    "productos_afectados": etapas["Crecimiento"]["cantidad"]
                })
                
            # Para productos en declive
            if etapas["Declive"]["cantidad"] > 0:
                recomendaciones.append({
                    "tipo": "Productos en Declive",
                    "accion": "Evaluar reducción de inventario o discontinuación",
                    "descripcion": "Estos productos están perdiendo demanda. Considere liquidación, promociones de cierre o discontinuación.",
                    "productos_afectados": etapas["Declive"]["cantidad"]
                })
            
            return {
                "exito": True,
                "mensaje": "Análisis de ciclo de vida completado",
                "tipo_ciclo": "lifecycle",
                "producto_id": product_id if product_id else "Todos",
                "total_productos_analizados": len(resultados),
                "fecha_analisis": datetime.now().isoformat(),
                "etapas_ciclo": [
                    {
                        "etapa": etapa,
                        "productos": etapas[etapa]["productos"][:10],  # Limitamos a 10 para la respuesta
                        "cantidad_productos": etapas[etapa]["cantidad"],
                        "porcentaje": etapas[etapa]["porcentaje"]
                    } for etapa in ["Introducción", "Crecimiento", "Madurez", "Declive"]
                ],
                "productos_detalle": sorted(resultados, key=lambda x: x["tendencia"], reverse=True)[:20],
                "recomendaciones": recomendaciones
            }
            
        except Exception as e:
            logger.error(f"Error en análisis de ciclo de vida simplificado: {str(e)}", exc_info=True)
            return {
                "exito": False,
                "mensaje": f"Error en análisis de ciclo de vida: {str(e)}"
            }
            

    @classmethod
    async def analyze_promotional_cycle_simple(cls, data: Dict[str, pd.DataFrame], product_id: Optional[str] = None):
        """Análisis simplificado de ciclos promocionales de productos"""
        try:
            # Obtener datos
            ventas_df = data.get("ventas", pd.DataFrame())
            detalles_df = data.get("ventadetalles", pd.DataFrame())
            productos_df = data.get("productos", pd.DataFrame())
            
            if ventas_df.empty or detalles_df.empty:
                return {
                    "exito": False,
                    "mensaje": "No hay suficientes datos para análisis promocional"
                }
            
            # Convertir IDs a string
            for df in [ventas_df, detalles_df, productos_df]:
                for col in df.columns:
                    if col.endswith('_id') or col == '_id' or col == 'venta' or col == 'producto':
                        df[col] = df[col].astype(str)
            
            # Crear diccionario de fechas de ventas
            fecha_ventas = {}
            for _, row in ventas_df.iterrows():
                fecha_ventas[str(row['_id'])] = row['createdAT']
                
            # Añadir columna de fecha a detalles
            detalles_df['fecha'] = detalles_df['venta'].map(fecha_ventas)
            detalles_df = detalles_df.dropna(subset=['fecha'])
            detalles_df['fecha'] = pd.to_datetime(detalles_df['fecha'])
            
            # Crear diccionario de nombres de productos
            nombres_productos = {}
            if not productos_df.empty:
                for _, row in productos_df.iterrows():
                    if 'titulo' in row:
                        nombres_productos[str(row['_id'])] = row['titulo']
                    elif 'nombre' in row:
                        nombres_productos[str(row['_id'])] = row['nombre']
                    else:
                        nombres_productos[str(row['_id'])] = f"Producto {row['_id']}"
            
            # Filtrar por producto específico si se proporciona
            if product_id:
                detalles_df = detalles_df[detalles_df['producto'] == product_id]
                if detalles_df.empty:
                    return {
                        "exito": False,
                        "mensaje": f"No hay datos de ventas para el producto ID: {product_id}"
                    }
            
            # Calcular precio unitario para detectar promociones
            if 'precio' in detalles_df.columns and 'cantidad' in detalles_df.columns:
                detalles_df['precio_unitario'] = detalles_df['precio'] / detalles_df['cantidad']
            else:
                # Si no tenemos datos de precio/cantidad, no podemos analizar promociones
                return {
                    "exito": False,
                    "mensaje": "No hay suficientes datos de precio y cantidad para análisis promocional"
                }
            
            # Análisis por producto
            resultados_promo = []
            
            for producto_id, grupo in detalles_df.groupby('producto'):
                if len(grupo) < 10:  # Necesitamos suficientes ventas para analizar
                    continue
                
                # Calcular estadísticas de precio
                precio_promedio = grupo['precio_unitario'].mean()
                precio_std = grupo['precio_unitario'].std()
                precio_min = grupo['precio_unitario'].min()
                precio_max = grupo['precio_unitario'].max()
                
                # Detectar posibles promociones (precio significativamente menor al promedio)
                umbral_promocion = precio_promedio - precio_std * 0.5
                grupo['es_promocion'] = grupo['precio_unitario'] < umbral_promocion
                
                # Agrupar por semanas para ver patrones
                grupo['semana'] = grupo['fecha'].dt.isocalendar().week
                grupo['año'] = grupo['fecha'].dt.year
                
                # Agrupar datos por semana
                ventas_semanales = grupo.groupby(['año', 'semana']).agg({
                    'cantidad': 'sum',
                    'precio': 'sum',
                    'es_promocion': 'sum'  # Contar ventas en promoción
                }).reset_index()
                
                ventas_semanales['tiene_promo'] = ventas_semanales['es_promocion'] > 0
                
                # Calcular impacto de promociones si hay datos suficientes
                if ventas_semanales['tiene_promo'].sum() > 0 and (~ventas_semanales['tiene_promo']).sum() > 0:
                    # Media de ventas en semanas con promoción
                    ventas_promo = ventas_semanales[ventas_semanales['tiene_promo']]['cantidad'].mean()
                    # Media de ventas en semanas sin promoción
                    ventas_normal = ventas_semanales[~ventas_semanales['tiene_promo']]['cantidad'].mean()
                    
                    # Impacto porcentual
                    impacto_porcentual = (ventas_promo - ventas_normal) / ventas_normal * 100 if ventas_normal > 0 else 0
                    
                    # Obtener el nombre del producto
                    nombre = nombres_productos.get(producto_id, f"Producto {producto_id}")
                    
                    # Determinar fase actual
                    semanas_con_promo = ventas_semanales[ventas_semanales['tiene_promo']].index
                    ultima_semana = ventas_semanales.index.max()
                    
                    if ultima_semana in semanas_con_promo:
                        fase_actual = "Durante promoción"
                    elif ultima_semana - 1 in semanas_con_promo:
                        fase_actual = "Post-promoción inmediata"
                    else:
                        fase_actual = "Periodo regular"
                    
                    resultados_promo.append({
                        "producto_id": producto_id,
                        "nombre": nombre,
                        "precio_promedio": round(precio_promedio, 2),
                        "precio_min": round(precio_min, 2),
                        "precio_max": round(precio_max, 2),
                        "variacion_precio": round((precio_max - precio_min) / precio_promedio * 100, 2),
                        "ventas_normales_promedio": round(ventas_normal, 2),
                        "ventas_promocion_promedio": round(ventas_promo, 2),
                        "impacto_promocional": round(impacto_porcentual, 2),
                        "semanas_con_promocion": int(ventas_semanales['tiene_promo'].sum()),
                        "total_semanas_analizadas": len(ventas_semanales),
                        "fase_actual": fase_actual
                    })
            
            # Agrupar por impacto promocional
            if resultados_promo:
                alto_impacto = [p for p in resultados_promo if p["impacto_promocional"] > 50]
                impacto_medio = [p for p in resultados_promo if 20 <= p["impacto_promocional"] <= 50]
                bajo_impacto = [p for p in resultados_promo if p["impacto_promocional"] < 20]
                
                # Generar recomendaciones
                recomendaciones = []
                
                # Para productos de alto impacto
                if alto_impacto:
                    recomendaciones.append({
                        "tipo": "Productos con Alto Impacto Promocional",
                        "accion": "Programar promociones estratégicas",
                        "descripcion": "Estos productos responden muy bien a promociones. Programe ofertas en temporadas estratégicas para maximizar ventas.",
                        "productos_afectados": len(alto_impacto)
                    })
                
                # Para productos de bajo impacto
                if bajo_impacto:
                    recomendaciones.append({
                        "tipo": "Productos con Bajo Impacto Promocional",
                        "accion": "Reconsiderar estrategia de descuentos",
                        "descripcion": "Estos productos no aumentan significativamente sus ventas durante promociones. Considere otras estrategias de mercadeo.",
                        "productos_afectados": len(bajo_impacto)
                    })
                
                # Recomendación general
                recomendaciones.append({
                    "tipo": "Optimización de calendario promocional",
                    "accion": "Espaciar promociones para evitar canibalización",
                    "descripcion": "Distribuya promociones a lo largo del tiempo para evitar afectar ventas regulares y mantener márgenes saludables.",
                    "productos_afectados": len(resultados_promo)
                })
                
                return {
                    "exito": True,
                    "mensaje": "Análisis de ciclo promocional completado",
                    "tipo_ciclo": "promotional",
                    "producto_id": product_id if product_id else "Todos",
                    "total_productos_analizados": len(resultados_promo),
                    "fecha_analisis": datetime.now().isoformat(),
                    "resumen_impacto": {
                        "alto_impacto": {
                            "cantidad": len(alto_impacto),
                            "porcentaje": round(len(alto_impacto) / len(resultados_promo) * 100, 2),
                            "productos": [{"id": p["producto_id"], "nombre": p["nombre"], "impacto": p["impacto_promocional"]} 
                                        for p in alto_impacto[:10]]
                        },
                        "impacto_medio": {
                            "cantidad": len(impacto_medio),
                            "porcentaje": round(len(impacto_medio) / len(resultados_promo) * 100, 2),
                            "productos": [{"id": p["producto_id"], "nombre": p["nombre"], "impacto": p["impacto_promocional"]} 
                                        for p in impacto_medio[:10]]
                        },
                        "bajo_impacto": {
                            "cantidad": len(bajo_impacto),
                            "porcentaje": round(len(bajo_impacto) / len(resultados_promo) * 100, 2),
                            "productos": [{"id": p["producto_id"], "nombre": p["nombre"], "impacto": p["impacto_promocional"]} 
                                        for p in bajo_impacto[:10]]
                        }
                    },
                    "productos_detalle": sorted(resultados_promo, key=lambda x: x["impacto_promocional"], reverse=True)[:20],
                    "recomendaciones": recomendaciones
                }
            else:
                return {
                    "exito": False,
                    "mensaje": "No se encontraron suficientes datos para análisis promocional",
                    "datos": []
                }
                
        except Exception as e:
            logger.error(f"Error en análisis promocional simplificado: {str(e)}", exc_info=True)
            return {
                "exito": False,
                "mensaje": f"Error en análisis promocional: {str(e)}"
            }