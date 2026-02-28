from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .database import create_tables
from .routers import graphs, experiments, versions, algorithms

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create database tables
    await create_tables()
    yield
    # Shutdown: cleanup if needed
    pass

app = FastAPI(
    title="Froge API", 
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware - allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(graphs.router, prefix="/api/v1")
app.include_router(experiments.router, prefix="/api/v1")
app.include_router(versions.router, prefix="/api/v1")
app.include_router(algorithms.router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Froge API - Graph-Native ML Workflow OS"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
