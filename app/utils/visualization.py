import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from typing import List, Dict, Any, Optional
import logging
from sklearn.tree import plot_tree

logger = logging.getLogger(__name__)

def generate_tree_image(model: Any, feature_names: List[str], class_names: List[str], max_depth: int = 4) -> str:
    """
    Genera una imagen del árbol de decisión codificada en base64
    """
    try:
        plt.figure(figsize=(20, 10))
        plot_tree(
            model,
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=8
        )
        plt.title('Árbol de Decisión para Predicción de Inventario', fontsize=16)
        plt.tight_layout()
        
        # Guardar en buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=120)
        buffer.seek(0)
        
        # Codificar en base64
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return image_base64
    except Exception as e:
        logger.error(f"Error al generar imagen del árbol: {e}")
        return ""

def generate_feature_importance_chart(model: Any, feature_names: List[str]) -> str:
    """
    Genera un gráfico de la importancia de características codificado en base64
    """
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.title('Importancia de Características', fontsize=14)
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        
        # Guardar en buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        
        # Codificar en base64
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return image_base64
    except Exception as e:
        logger.error(f"Error al generar gráfico de importancia: {e}")
        return ""

def generate_inventory_status_chart(df: pd.DataFrame) -> str:
    """
    Genera un gráfico de la distribución de estados de inventario codificado en base64
    """
    try:
        if 'estado_inventario' not in df.columns:
            return ""
        
        state_counts = df['estado_inventario'].value_counts()
        
        plt.figure(figsize=(10, 6))
        colors = ['red', 'orange', 'green', 'blue', 'purple']
        
        plt.pie(
            state_counts,
            labels=state_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        plt.axis('equal')
        plt.title('Distribución de Estados de Inventario')
        
        # Guardar en buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        
        # Codificar en base64
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return image_base64
    except Exception as e:
        logger.error(f"Error al generar gráfico de estados: {e}")
        return ""