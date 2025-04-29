from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import api_router  
from app.core.config import settings

# Inicializar FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.PROJECT_VERSION,
    openapi_url=f"{settings.API}/openapi.json"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar routers
app.include_router(
    api_router,
    prefix=settings.API,
    # tags=["inventory"]
)

@app.get("/", tags=["status"])
async def root():
    """Endpoint raíz para verificar que la API está funcionando"""
    return {
        "status": "online",
        "message": "Inventory Analysis API is running",
        "version": settings.PROJECT_VERSION,
    }

@app.get("/health", tags=["status"])
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=10005, reload=True, timeout_keep_alive=120, limit_max_requests=500 )