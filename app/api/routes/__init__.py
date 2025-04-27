"""
Módulo que contiene todas las rutas de la API.
"""

from fastapi import APIRouter
from app.api.routes.inventory import router as inventory_router
from app.api.routes.cycle import router as cycle_router
from app.api.routes.model import router as model_router  

# Router principal que agrupa todos los demás routers
api_router = APIRouter()

# Incluir todos los routers de la aplicación con sus prefijos
api_router.include_router(inventory_router, prefix="/inventory", tags=["inventory"])
api_router.include_router(cycle_router, prefix="/cycle", tags=["cycle"])
api_router.include_router(model_router, prefix="/model", tags=["model"])

# Exportar todos los routers para facilitar la importación
__all__ = ["api_router", "inventory_router", "cycle_router","model_router"]