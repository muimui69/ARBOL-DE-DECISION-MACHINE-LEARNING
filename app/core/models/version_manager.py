import os
import json
from datetime import datetime
import pickle
from typing import Dict, List, Any, Optional
import pandas as pd

from app.core.config import settings

class ModelVersionManager:
    """Gestor de versiones de modelos para la API de análisis de inventario"""
    
    VERSIONS_FILE = "model_versions.json"
    
    @classmethod
    def get_versions_path(cls):
        """Obtiene la ruta al archivo de versiones"""
        return os.path.join(settings.MODEL_PATH, cls.VERSIONS_FILE)
    
    @classmethod
    def get_all_versions(cls) -> List[Dict[str, Any]]:
        """Obtiene todas las versiones de modelos almacenadas"""
        versions_path = cls.get_versions_path()
        
        if not os.path.exists(versions_path):
            return []
        
        with open(versions_path, 'r') as f:
            return json.load(f)
    
    @classmethod
    def get_latest_version(cls) -> Optional[Dict[str, Any]]:
        """Obtiene la última versión del modelo"""
        versions = cls.get_all_versions()
        return versions[-1] if versions else None
    
    @classmethod
    def get_version_by_id(cls, version_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene una versión específica por ID"""
        versions = cls.get_all_versions()
        for version in versions:
            if version.get('version_id') == version_id:
                return version
        return None
    
    @classmethod
    def save_new_version(cls, model_info: Dict[str, Any], files: Dict[str, str]) -> Dict[str, Any]:
        """Guarda una nueva versión de modelo"""
        versions = cls.get_all_versions()
        
        # Generar ID único para esta versión
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        version_id = f"v{len(versions) + 1}_{timestamp}"
        
        # Crear carpeta para esta versión
        version_path = os.path.join(settings.MODEL_PATH, version_id)
        os.makedirs(version_path, exist_ok=True)
        
        # Guardar archivos versionados
        versioned_files = {}
        for file_key, file_path in files.items():
            filename = os.path.basename(file_path)
            versioned_filename = f"{version_id}_{filename}"
            versioned_path = os.path.join(version_path, versioned_filename)
            
            # Copiar archivo a la ubicación versionada
            with open(file_path, 'rb') as src, open(versioned_path, 'wb') as dst:
                dst.write(src.read())
            
            versioned_files[file_key] = versioned_path
        
        # Crear registro de versión
        version_record = {
            "version_id": version_id,
            "timestamp": datetime.now().isoformat(),
            "model_info": model_info,
            "files": versioned_files,
            "is_active": True
        }
        
        # Desactivar versión anterior
        for version in versions:
            version['is_active'] = False
        
        # Añadir nueva versión
        versions.append(version_record)
        
        # Guardar registro de versiones
        with open(cls.get_versions_path(), 'w') as f:
            json.dump(versions, f, indent=2)
        
        return version_record
    
    @classmethod
    def activate_version(cls, version_id: str) -> Optional[Dict[str, Any]]:
        """Activa una versión específica de modelo"""
        versions = cls.get_all_versions()
        target_version = None
        
        for version in versions:
            if version.get('version_id') == version_id:
                version['is_active'] = True
                target_version = version
            else:
                version['is_active'] = False
        
        if target_version:
            # Guardar registro de versiones actualizado
            with open(cls.get_versions_path(), 'w') as f:
                json.dump(versions, f, indent=2)
            
            # Copiar archivos de modelo a ubicaciones principales
            for file_key, file_path in target_version['files'].items():
                target_filename = file_key + ".pkl"  # Por ejemplo: classifier.pkl
                target_path = os.path.join(settings.MODEL_PATH, target_filename)
                
                # Copiar archivo versionado a ubicación principal
                with open(file_path, 'rb') as src, open(target_path, 'wb') as dst:
                    dst.write(src.read())
            
            return target_version
            
        return None
    
    @classmethod
    def get_active_version(cls) -> Optional[Dict[str, Any]]:
        """Obtiene la versión activa del modelo"""
        versions = cls.get_all_versions()
        for version in versions:
            if version.get('is_active', False):
                return version
        return None