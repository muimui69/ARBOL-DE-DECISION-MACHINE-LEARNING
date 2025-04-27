import os
import pickle
import joblib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics

from app.core.config import settings
from app.core.models.version_manager import ModelVersionManager

class DecisionTreeModel:
    """Modelo de árbol de decisión para análisis de inventario"""
    
    # Rutas de archivos para el modelo actual
    SCALER_FILE = os.path.join(settings.MODEL_PATH, "scaler.pkl")
    CLASSIFIER_FILE = os.path.join(settings.MODEL_PATH, "classifier.pkl")
    METADATA_FILE = os.path.join(settings.MODEL_PATH, "metadata.pkl")
    REGRESSOR_FILE = os.path.join(settings.MODEL_PATH, "regressor.pkl")
    
    # Constantes para etiquetas de clase
    CRITICAL = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    EXCESS = 4
    
    @classmethod
    def get_model_info(cls) -> Dict[str, Any]:
        """Obtiene la información del modelo cargado actual"""
        if not os.path.exists(cls.METADATA_FILE):
            return None
        
        try:
            with open(cls.METADATA_FILE, 'rb') as f:
                metadata = pickle.load(f)
            return metadata
        except Exception as e:
            print(f"Error al cargar información del modelo: {e}")
            return None
    
    @classmethod
    def _prepare_training_data(cls, data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepara los datos para el entrenamiento del modelo"""
        # Obtener DataFrames con los nombres CORRECTOS de colecciones
        productos_df = data.get('productos', pd.DataFrame())
        ventas_df = data.get('ventas', pd.DataFrame())
        ventadetalles_df = data.get('ventadetalles', pd.DataFrame())  # CORREGIDO
        producto_variedads_df = data.get('producto_variedads', pd.DataFrame())  # CORREGIDO
        ingresodetalles_df = data.get('ingresodetalles', pd.DataFrame())  # CORREGIDO
        
        # Verificar si tenemos datos
        print(f"Productos: {len(productos_df)} registros")
        print(f"Ventas: {len(ventas_df)} registros")
        print(f"Venta detalles: {len(ventadetalles_df)} registros")
        print(f"Variedades: {len(producto_variedads_df)} registros")
        print(f"Ingreso detalles: {len(ingresodetalles_df)} registros")
        
        # Crear inventario_df a partir de ingresodetalles
        if not ingresodetalles_df.empty:
            print("Creando inventario a partir de ingresodetalles...")
            # Examinar contenido
            print("Columnas disponibles:", ingresodetalles_df.columns.tolist())
            
            if 'producto' in ingresodetalles_df.columns and 'estado' in ingresodetalles_df.columns:
                # Solo los registros activos (estado=true)
                activos = ingresodetalles_df[ingresodetalles_df['estado'] == True]
                # Agrupar por producto y contar
                inventario_df = activos.groupby('producto').size().reset_index()
                inventario_df.columns = ['producto_id', 'stock']
                print(f"Inventario creado: {len(inventario_df)} productos con stock")
            else:
                print("No se encontraron las columnas necesarias en ingresodetalles")
                return pd.DataFrame(), np.array([])
        
        # Verificar que tenemos datos mínimos
        if productos_df.empty or ventas_df.empty or ventadetalles_df.empty or inventario_df.empty:
            print("Datos insuficientes para entrenamiento")
            return pd.DataFrame(), np.array([])
        
        try:
            # Sección debugging para ver exactamente qué hay en cada columna
            print("Columnas en ventadetalles_df:", ventadetalles_df.columns.tolist())
            print("Columnas en productos_df:", productos_df.columns.tolist())
            print("Columnas en ventas_df:", ventas_df.columns.tolist())
            
            # Comprobación de tipos de IDs
            if 'producto' in ventadetalles_df.columns:
                print("Tipos de producto en ventadetalles:", ventadetalles_df['producto'].dtype)
                print("Ejemplo de IDs de producto:", ventadetalles_df['producto'].head().tolist())
            
            if '_id' in productos_df.columns:
                print("Tipos de _id en productos:", productos_df['_id'].dtype)
                print("Ejemplo de _id en productos:", productos_df['_id'].head().tolist())
            
            # AQUÍ LA CLAVE: Verificar la columna usada para producto_id en inventario_df
            print("Columnas en inventario_df:", inventario_df.columns.tolist())
            print("Tipos de producto_id en inventario:", inventario_df['producto_id'].dtype)
            
            # Asegurar que todos los IDs son strings para comparación
            if 'producto' in ventadetalles_df.columns:
                ventadetalles_df['producto'] = ventadetalles_df['producto'].astype(str)
            
            if '_id' in productos_df.columns:
                productos_df['_id'] = productos_df['_id'].astype(str)
            
            inventario_df['producto_id'] = inventario_df['producto_id'].astype(str)
            
            # Convertir fechas
            if 'createdAT' in ventas_df.columns:
                ventas_df['fecha'] = pd.to_datetime(ventas_df['createdAT'])
            
                # Obtener fecha más reciente para análisis
                fecha_actual = ventas_df['fecha'].max()
                
                # Calcular periodo de 30 días
                periodo_30d = fecha_actual - pd.Timedelta(days=30)
                periodo_90d = fecha_actual - pd.Timedelta(days=90)
                
                # Unir ventas y detalles
                # Convertimos _id a str para poder unir
                ventas_df['_id'] = ventas_df['_id'].astype(str)
                ventadetalles_df['venta'] = ventadetalles_df['venta'].astype(str)
                
                ventas_completas = pd.merge(
                    ventas_df, 
                    ventadetalles_df,
                    left_on='_id',
                    right_on='venta'
                )
                
                # Filtrar ventas de los últimos 90 días
                ventas_90d = ventas_completas[ventas_completas['fecha'] >= periodo_90d]
                ventas_30d = ventas_completas[ventas_completas['fecha'] >= periodo_30d]
                
                # Calcular ventas por producto
                ventas_producto_90d = ventas_90d.groupby('producto')['cantidad'].sum().reset_index()
                ventas_producto_90d.rename(columns={'cantidad': 'ventas_90d', 'producto': 'producto_id'}, inplace=True)
                
                ventas_producto_30d = ventas_30d.groupby('producto')['cantidad'].sum().reset_index()
                ventas_producto_30d.rename(columns={'cantidad': 'ventas_30d', 'producto': 'producto_id'}, inplace=True)
                
                # Datos para entrenamiento 
                # Convertimos _id de productos a str para poder unir
                productos_df['_id'] = productos_df['_id'].astype(str)
                
                training_data = pd.merge(
                    inventario_df, 
                    productos_df[['_id', 'titulo']],
                    left_on='producto_id',
                    right_on='_id', 
                    how='inner'
                )
                
                ventas_producto_90d['producto_id'] = ventas_producto_90d['producto_id'].astype(str)
                ventas_producto_30d['producto_id'] = ventas_producto_30d['producto_id'].astype(str)
                
                training_data = pd.merge(training_data, ventas_producto_90d, on='producto_id', how='left')
                training_data = pd.merge(training_data, ventas_producto_30d, on='producto_id', how='left')
                
                # Rellenar valores NaN
                training_data['ventas_90d'] = training_data['ventas_90d'].fillna(0)
                training_data['ventas_30d'] = training_data['ventas_30d'].fillna(0)
                
                # Calcular características adicionales
                training_data['ventas_diarias_prom'] = training_data['ventas_90d'] / 90
                
                # Días de stock restante (basado en ventas diarias promedio)
                with np.errstate(divide='ignore', invalid='ignore'):
                    training_data['dias_stock_restante'] = np.where(
                        training_data['ventas_diarias_prom'] > 0,
                        training_data['stock'] / training_data['ventas_diarias_prom'],
                        999  # Un valor alto para productos sin ventas
                    )
                
                # Calcular tendencia (comparando ventas_30d contra el promedio anterior)
                ventas_30d_anteriores = (training_data['ventas_90d'] - training_data['ventas_30d']) / 2
                with np.errstate(divide='ignore', invalid='ignore'):
                    training_data['tendencia'] = np.where(
                        ventas_30d_anteriores > 0,
                        (training_data['ventas_30d'] - ventas_30d_anteriores) / ventas_30d_anteriores,
                        0  # Sin tendencia si no hay ventas anteriores
                    )
                
                # Coeficiente de variación (estimación)
                training_data['coef_variacion'] = 0.2
                
                # Aplicar lógica de inventario para etiquetar
                conditions = [
                    # Crítico: Menos de 7 días de stock o stock cero
                    (training_data['dias_stock_restante'] < 7) | (training_data['stock'] == 0),
                    # Bajo: Entre 7 y 15 días de stock
                    (training_data['dias_stock_restante'] >= 7) & (training_data['dias_stock_restante'] < 15),
                    # Normal: Entre 15 y 30 días de stock
                    (training_data['dias_stock_restante'] >= 15) & (training_data['dias_stock_restante'] < 30),
                    # Alto: Entre 30 y 60 días de stock
                    (training_data['dias_stock_restante'] >= 30) & (training_data['dias_stock_restante'] < 60)
                ]
                
                choices = [cls.CRITICAL, cls.LOW, cls.NORMAL, cls.HIGH]
                
                # Por defecto, productos con más de 60 días de stock son excedentes
                training_data['estado'] = cls.EXCESS
                training_data['estado'] = np.select(conditions, choices, default=cls.EXCESS)
                
                # Seleccionar características para entrenar
                features = [
                    'stock', 'ventas_30d', 'ventas_90d', 
                    'ventas_diarias_prom', 'dias_stock_restante', 
                    'tendencia', 'coef_variacion'
                ]
                
                X = training_data[features]
                y = training_data['estado'].values
                
                print(f"Datos preparados: {len(X)} muestras con {len(features)} características")
                return X, y
            else:
                print("Campo 'createdAT' no encontrado en ventas_df")
                return pd.DataFrame(), np.array([])
                
        except Exception as e:
            print(f"Error al preparar datos para entrenamiento: {e}")
            return pd.DataFrame(), np.array([])
    
    @classmethod
    async def train(cls, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Entrena un nuevo modelo de árbol de decisión con los datos proporcionados"""
        try:
            # Preparar datos
            X, y = cls._prepare_training_data(data)
            
            if len(X) == 0 or len(y) == 0:
                return {
                    "exito": False,
                    "mensaje": "No hay suficientes datos para entrenar el modelo",
                    "error": "Dataset vacío o insuficiente"
                }
            
            print(f"Entrenando modelo con {len(X)} muestras")
            
            # Crear y guardar el escalador
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Crear y entrenar clasificador
            max_depth = 5  # Valor por defecto si no está en settings
            if hasattr(settings, 'MODEL_MAX_DEPTH') and settings.MODEL_MAX_DEPTH:
                try:
                    max_depth = int(settings.MODEL_MAX_DEPTH)
                except (ValueError, TypeError):
                    pass
                    
            classifier = DecisionTreeClassifier(
                max_depth=max_depth,
                random_state=42
            )
            classifier.fit(X_scaled, y)
            
            regressor = DecisionTreeRegressor(
                max_depth=max_depth, 
                random_state=42
            )
            regressor.fit(X_scaled, y)  
            
            # Evaluar modelo
            y_pred = classifier.predict(X_scaled)
            accuracy = metrics.accuracy_score(y, y_pred)
            
            # Guardar metadatos
            features = X.columns.tolist()
            metadata = {
                "modelo": "Árbol de Decisión",
                "fecha_entrenamiento": datetime.now().isoformat(),
                "precision": float(accuracy),
                "caracteristicas": features,
                "profundidad_maxima": max_depth,
                "esta_cargado": True,
                "num_muestras": len(X)
            }
            
            # Crear directorio si no existe
            os.makedirs(settings.MODEL_PATH, exist_ok=True)
            
            # Guardar archivos del modelo
            with open(cls.SCALER_FILE, 'wb') as f:
                pickle.dump(scaler, f)
                
            with open(cls.CLASSIFIER_FILE, 'wb') as f:
                pickle.dump(classifier, f)
                
            with open(cls.METADATA_FILE, 'wb') as f:
                pickle.dump(metadata, f)
                
            with open(cls.REGRESSOR_FILE, 'wb') as f:
                pickle.dump(regressor, f)
            
            # Registrar versión si está disponible el gestor de versiones
            version_id = datetime.now().strftime('v1_%Y%m%d%H%M%S')
            try:
                # Necesitamos crear copias temporales para el versionado
                temp_scaler_file = os.path.join(settings.MODEL_PATH, "temp_scaler.pkl")
                temp_classifier_file = os.path.join(settings.MODEL_PATH, "temp_classifier.pkl") 
                temp_metadata_file = os.path.join(settings.MODEL_PATH, "temp_metadata.pkl")
                temp_regressor_file = os.path.join(settings.MODEL_PATH, "temp_regressor.pkl")  # 
                
                # Copiar archivos 
                with open(cls.SCALER_FILE, 'rb') as src:
                    with open(temp_scaler_file, 'wb') as dst:
                        dst.write(src.read())
                
                with open(cls.CLASSIFIER_FILE, 'rb') as src:
                    with open(temp_classifier_file, 'wb') as dst:
                        dst.write(src.read())
                
                with open(cls.METADATA_FILE, 'rb') as src:
                    with open(temp_metadata_file, 'wb') as dst:
                        dst.write(src.read())
                        
                with open(cls.REGRESSOR_FILE, 'rb') as src:
                    with open(temp_regressor_file, 'wb') as dst:
                        dst.write(src.read())
                        
                # Registrar versión
                files = {
                    "scaler": temp_scaler_file,
                    "classifier": temp_classifier_file,
                    "metadata": temp_metadata_file,
                    "regressor": temp_regressor_file
                }
                
                version_record = ModelVersionManager.save_new_version(metadata, files)
                version_id = version_record["version_id"]
                
                # Limpiar archivos temporales
                for temp_file in files.values():
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        
                print(f"Versión registrada: {version_id}")
                
            except Exception as e:
                print(f"Aviso: No se pudo registrar versión del modelo: {e}")
                print("Continuando sin versionado...")
            
            return {
                "exito": True,
                "mensaje": "Modelo entrenado correctamente",
                "info_modelo": metadata,
                "version": version_id
            }
            
        except Exception as e:
            return {
                "exito": False,
                "mensaje": "Error al entrenar modelo",
                "error": str(e)
            }
            
            
    @classmethod
    async def predict(cls, data: pd.DataFrame) -> Dict[str, Any]:
        """Realiza predicciones para nuevos datos"""
        try:
            if not os.path.exists(cls.CLASSIFIER_FILE) or not os.path.exists(cls.SCALER_FILE) or not os.path.exists(cls.METADATA_FILE):
                return {
                    "exito": False,
                    "mensaje": "Modelo no encontrado",
                    "error": "Entrene el modelo primero"
                }
            
            # Cargar metadata para obtener el orden EXACTO de características
            with open(cls.METADATA_FILE, 'rb') as f:
                metadata = pickle.load(f)
                feature_order = metadata.get("caracteristicas", [])
                    
            print(f"Orden de características durante entrenamiento: {feature_order}")
            
            # Verificar si hay columna inventario_actual pero falta stock
            if 'inventario_actual' in data.columns and 'stock' not in data.columns:
                data = data.rename(columns={'inventario_actual': 'stock'})
            
            # CALCULAR días de stock restante en lugar de usar el valor proporcionado
            with np.errstate(divide='ignore', invalid='ignore'):
                data['dias_stock_restante'] = np.where(
                    data['ventas_diarias_prom'] > 0,
                    data['stock'] / data['ventas_diarias_prom'],
                    999  # Un valor alto para productos sin ventas
                )
            
            # Asegurarse de que todas las columnas necesarias existan
            for col in feature_order:
                if col not in data.columns:
                    print(f"Columna faltante: {col}, rellenando con 0")
                    data[col] = 0.0
            
            # IMPORTANTE: Extraer características en EL MISMO ORDEN que durante el entrenamiento
            X = data[feature_order].copy()  
            print(f"Columnas usadas para predicción: {X.columns.tolist()}")
            
            # Cargar scaler y modelo
            with open(cls.SCALER_FILE, 'rb') as f:
                scaler = pickle.load(f)
                    
            with open(cls.CLASSIFIER_FILE, 'rb') as f:
                classifier = pickle.load(f)
                    
            # Cargar modelo de regresión si existe
            if os.path.exists(cls.REGRESSOR_FILE):
                with open(cls.REGRESSOR_FILE, 'rb') as f:
                    regressor = pickle.load(f)
                has_regressor = True
            else:
                has_regressor = False
            
            # Normalizar datos utilizando el orden correcto de características
            X_scaled = scaler.transform(X)
            
            # Predecir clase
            y_pred = classifier.predict(X_scaled)
            
            # Predecir días restantes con modelo de regresión si está disponible
            if has_regressor:
                dias_predichos = regressor.predict(X_scaled)
            else:
                # Usar el cálculo simple si no hay modelo de regresión
                dias_predichos = data['dias_stock_restante'].values
            
            # Mapear códigos a nombres de estados
            estados = []
            for code in y_pred:
                if code == cls.CRITICAL:
                    estados.append("Crítico")
                elif code == cls.LOW:
                    estados.append("Bajo")
                elif code == cls.NORMAL:
                    estados.append("Normal")
                elif code == cls.HIGH:
                    estados.append("Alto")
                else:
                    estados.append("Exceso")
            
            return {
                "exito": True,
                "mensaje": "Predicción realizada correctamente",
                "resultados": {
                    "estados": estados,
                    "codigos": y_pred.tolist(),
                    "dias_restantes": dias_predichos.tolist()
                }
            }
        except Exception as e:
            print(f"Error en predicción: {e}")
            return {
                "exito": False,
                "mensaje": "Error al realizar la predicción",
                "error": str(e)
            }
            
            
