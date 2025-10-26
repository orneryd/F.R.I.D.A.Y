"""
FastAPI application for the 3D Synaptic Neuron System

This module provides the REST API interface using FastAPI.
The actual application initialization is handled by the main.py module.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import time
from typing import Optional
import logging

from neuron_system.api.models import HealthResponse, ErrorResponse
from neuron_system.core.graph import NeuronGraph
from neuron_system.engines.compression import CompressionEngine
from neuron_system.engines.query import QueryEngine
from neuron_system.engines.training import TrainingEngine
from neuron_system.storage.database import DatabaseManager
from neuron_system.storage.neuron_store import NeuronStore
from neuron_system.storage.synapse_store import SynapseStore
from neuron_system.spatial.spatial_index import SpatialIndex
from neuron_system.config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Application State
# ============================================================================

class AppState:
    """
    Global application state for the API.
    
    This class holds references to all system components that are initialized
    during startup and used by API endpoints.
    """
    def __init__(self):
        self.graph: Optional[NeuronGraph] = None
        self.compression_engine: Optional[CompressionEngine] = None
        self.query_engine: Optional[QueryEngine] = None
        self.training_engine: Optional[TrainingEngine] = None
        self.database: Optional[DatabaseManager] = None
        self.neuron_store: Optional[NeuronStore] = None
        self.synapse_store: Optional[SynapseStore] = None
        self.spatial_index: Optional[SpatialIndex] = None
        self.settings: Optional[Settings] = None
        self.start_time: datetime = datetime.now()


app_state = AppState()


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="3D Synaptic Neuron System API",
    description="A novel knowledge representation system using 3D neural networks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# ============================================================================
# CORS Middleware
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request Timing Middleware
# ============================================================================

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000  # Convert to ms
    response.headers["X-Process-Time-Ms"] = str(round(process_time, 2))
    return response


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions"""
    logger.error(f"ValueError: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error_code="INVALID_INPUT",
            message=str(exc),
            timestamp=datetime.now(),
            recoverable=True
        ).dict()
    )


@app.exception_handler(KeyError)
async def key_error_handler(request: Request, exc: KeyError):
    """Handle KeyError exceptions (e.g., neuron not found)"""
    logger.error(f"KeyError: {str(exc)}")
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error_code="NOT_FOUND",
            message=f"Resource not found: {str(exc)}",
            timestamp=datetime.now(),
            recoverable=True
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="INTERNAL_ERROR",
            message="An internal error occurred",
            details={"error": str(exc)},
            timestamp=datetime.now(),
            recoverable=False
        ).dict()
    )


# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Initialize application components on startup.
    
    This function is called automatically when the FastAPI application starts.
    It initializes all system components in the correct order:
    1. Load configuration settings
    2. Initialize database and storage
    3. Initialize neuron graph and spatial index
    4. Load existing data from database
    5. Initialize processing engines
    """
    logger.info("Starting 3D Synaptic Neuron System API...")
    
    try:
        # Load settings
        app_state.settings = Settings()
        logger.info(f"Loaded settings: database_path={app_state.settings.database_path}")
        
        # Initialize database
        app_state.database = DatabaseManager(app_state.settings.database_path)
        logger.info("Database initialized")
        
        # Initialize stores
        app_state.neuron_store = NeuronStore(app_state.database)
        app_state.synapse_store = SynapseStore(app_state.database)
        logger.info("Stores initialized")
        
        # Initialize neuron graph with bounds from settings
        from neuron_system.core.vector3d import Vector3D
        min_bound = Vector3D(*app_state.settings.spatial_bounds_min)
        max_bound = Vector3D(*app_state.settings.spatial_bounds_max)
        app_state.graph = NeuronGraph(bounds=(min_bound, max_bound))
        logger.info("Neuron graph initialized")
        
        # Attach storage to graph for automatic persistence
        app_state.graph.attach_storage(
            neuron_store=app_state.neuron_store,
            synapse_store=app_state.synapse_store
        )
        
        # Use the spatial index from the graph
        app_state.spatial_index = app_state.graph.spatial_index
        logger.info("Spatial index initialized")
        
        # Load existing neurons and synapses from database
        neurons = app_state.neuron_store.list_all()
        loaded_count = 0
        for neuron in neurons:
            if neuron:  # Skip None values
                app_state.graph.add_neuron(neuron)
                loaded_count += 1
        logger.info(f"Loaded {loaded_count} neurons from database")
        
        synapses = app_state.synapse_store.list_all()
        loaded_count = 0
        for synapse in synapses:
            if synapse:  # Skip None values
                app_state.graph.add_synapse(synapse)
                loaded_count += 1
        logger.info(f"Loaded {loaded_count} synapses from database")
        
        # Initialize engines
        app_state.compression_engine = CompressionEngine()
        logger.info("Compression engine initialized")
        
        app_state.query_engine = QueryEngine(
            neuron_graph=app_state.graph,
            compression_engine=app_state.compression_engine
        )
        logger.info("Query engine initialized")
        
        app_state.training_engine = TrainingEngine(
            neuron_graph=app_state.graph
        )
        logger.info("Training engine initialized")
        
        logger.info("API startup complete!")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Shutting down 3D Synaptic Neuron System API...")
    
    try:
        # Save any pending changes
        if app_state.neuron_store and app_state.graph:
            neurons_to_save = [n for n in app_state.graph.neurons.values()]
            if neurons_to_save:
                app_state.neuron_store.batch_update(neurons_to_save)
            logger.info(f"Saved {len(neurons_to_save)} neurons")
        
        if app_state.synapse_store and app_state.graph:
            synapses_to_save = [s for s in app_state.graph.synapses.values()]
            if synapses_to_save:
                app_state.synapse_store.batch_update(synapses_to_save)
            logger.info(f"Saved {len(synapses_to_save)} synapses")
        
        # Close database connection
        if app_state.database:
            app_state.database.close()
            logger.info("Database connection closed")
        
        logger.info("Shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}", exc_info=True)


# ============================================================================
# Root and Health Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "3D Synaptic Neuron System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    neuron_count = len(app_state.graph.neurons) if app_state.graph else 0
    synapse_count = len(app_state.graph.synapses) if app_state.graph else 0
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        neuron_count=neuron_count,
        synapse_count=synapse_count,
        timestamp=datetime.now()
    )


# ============================================================================
# Register Middleware
# ============================================================================

from neuron_system.api.middleware import (
    rate_limit_middleware,
    logging_middleware,
    security_headers_middleware,
    request_size_middleware,
    cors_cache_middleware,
    health_check_bypass_middleware,
    error_tracking_middleware
)

# Register middleware in order (executed in reverse order)
app.middleware("http")(error_tracking_middleware)
app.middleware("http")(cors_cache_middleware)
app.middleware("http")(security_headers_middleware)
app.middleware("http")(request_size_middleware)
app.middleware("http")(rate_limit_middleware)
app.middleware("http")(logging_middleware)
app.middleware("http")(health_check_bypass_middleware)


# ============================================================================
# Import and Register Routes
# ============================================================================

from neuron_system.api.routes import neurons, synapses, query, training, visualization

app.include_router(neurons.router, prefix="/api/v1", tags=["Neurons"])
app.include_router(synapses.router, prefix="/api/v1", tags=["Synapses"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])
app.include_router(training.router, prefix="/api/v1", tags=["Training"])
app.include_router(visualization.router, prefix="/api/v1", tags=["Visualization"])
