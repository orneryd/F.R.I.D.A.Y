"""
Query and search endpoints
"""
from fastapi import APIRouter, HTTPException, status
from uuid import UUID, uuid4
from datetime import datetime
import time
import logging

from neuron_system.api.models import (
    QueryRequest,
    SpatialQueryRequest,
    QueryResponse,
    ActivatedNeuronResponse,
    NeuronResponse,
    NeighborsResponse,
    Vector3DModel
)
from neuron_system.core.vector3d import Vector3D
from neuron_system.neuron_types.knowledge_neuron import KnowledgeNeuron
from neuron_system.neuron_types.tool_neuron import ToolNeuron

logger = logging.getLogger(__name__)

router = APIRouter()


def get_app_state():
    """Get application state from app context"""
    from neuron_system.api.app import app_state
    return app_state


def neuron_to_response(neuron) -> NeuronResponse:
    """Convert a Neuron object to NeuronResponse"""
    response_data = {
        "id": str(neuron.id),
        "neuron_type": neuron.neuron_type.value,
        "position": Vector3DModel(
            x=neuron.position.x,
            y=neuron.position.y,
            z=neuron.position.z
        ),
        "vector": neuron.vector.tolist() if neuron.vector is not None else None,
        "activation_level": neuron.activation_level,
        "metadata": neuron.metadata,
        "created_at": neuron.created_at,
        "modified_at": neuron.modified_at
    }
    
    # Add type-specific fields
    if isinstance(neuron, KnowledgeNeuron):
        response_data["source_data"] = neuron.source_data
        response_data["semantic_tags"] = neuron.semantic_tags
    elif isinstance(neuron, ToolNeuron):
        response_data["function_signature"] = neuron.function_signature
        response_data["execution_count"] = neuron.execution_count
    
    return NeuronResponse(**response_data)


@router.post("/query", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    """
    Execute a knowledge query against the neuron network
    
    - **query_text**: Text to search for
    - **top_k**: Number of results to return (1-100, default: 10)
    - **propagation_depth**: Depth of activation propagation (1-10, default: 3)
    - **neuron_type_filter**: Optional filter by neuron type
    
    Returns activated neurons sorted by activation score, including any tool execution results.
    """
    state = get_app_state()
    start_time = time.time()
    
    try:
        # Execute query
        results = state.query_engine.query(
            query_text=request.query_text,
            top_k=request.top_k,
            propagation_depth=request.propagation_depth,
            neuron_type_filter=request.neuron_type_filter
        )
        
        # Convert results to response format
        activated_neurons = []
        for result in results:
            neuron = result["neuron"]
            activation_score = result["activation"]
            tool_result = result.get("tool_result")
            
            activated_neurons.append(
                ActivatedNeuronResponse(
                    neuron=neuron_to_response(neuron),
                    activation_score=activation_score,
                    tool_result=tool_result
                )
            )
        
        execution_time_ms = (time.time() - start_time) * 1000
        query_id = str(uuid4())
        
        logger.info(
            f"Query executed: '{request.query_text[:50]}...' "
            f"returned {len(activated_neurons)} results in {execution_time_ms:.2f}ms"
        )
        
        return QueryResponse(
            activated_neurons=activated_neurons,
            execution_time_ms=round(execution_time_ms, 2),
            query_id=query_id
        )
        
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/query/spatial", response_model=QueryResponse)
async def execute_spatial_query(request: SpatialQueryRequest):
    """
    Execute a spatial query to find neurons within a 3D region
    
    - **center**: Center point of the search region (x, y, z)
    - **radius**: Search radius
    - **neuron_type_filter**: Optional filter by neuron type
    
    Returns neurons within the specified spatial region.
    """
    state = get_app_state()
    start_time = time.time()
    
    try:
        center = Vector3D(
            x=request.center.x,
            y=request.center.y,
            z=request.center.z
        )
        
        # Execute spatial query
        results = state.query_engine.spatial_query(
            center=center,
            radius=request.radius,
            neuron_type_filter=request.neuron_type_filter
        )
        
        # Convert results to response format
        activated_neurons = []
        for result in results:
            neuron = result["neuron"]
            activation_score = result.get("activation", 1.0)  # Spatial queries may not have activation
            
            activated_neurons.append(
                ActivatedNeuronResponse(
                    neuron=neuron_to_response(neuron),
                    activation_score=activation_score,
                    tool_result=None
                )
            )
        
        execution_time_ms = (time.time() - start_time) * 1000
        query_id = str(uuid4())
        
        logger.info(
            f"Spatial query executed at ({center.x}, {center.y}, {center.z}) "
            f"with radius {request.radius}, returned {len(activated_neurons)} results "
            f"in {execution_time_ms:.2f}ms"
        )
        
        return QueryResponse(
            activated_neurons=activated_neurons,
            execution_time_ms=round(execution_time_ms, 2),
            query_id=query_id
        )
        
    except Exception as e:
        logger.error(f"Error executing spatial query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/neurons/{neuron_id}/neighbors", response_model=NeighborsResponse)
async def get_neuron_neighbors(neuron_id: str):
    """
    Get all neurons connected to a specific neuron via synapses
    
    - **neuron_id**: UUID of the neuron
    
    Returns all connected neurons with their synapse information.
    """
    state = get_app_state()
    
    try:
        neuron_uuid = UUID(neuron_id)
        
        if neuron_uuid not in state.graph.neurons:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Neuron {neuron_id} not found"
            )
        
        # Get neighbors (outgoing synapses)
        neighbors_data = state.graph.get_neighbors(neuron_uuid)
        
        neighbors = []
        for synapse, target_neuron in neighbors_data:
            neighbors.append({
                "synapse": {
                    "id": str(synapse.id),
                    "weight": synapse.weight,
                    "usage_count": synapse.usage_count,
                    "synapse_type": synapse.synapse_type.value
                },
                "neuron": neuron_to_response(target_neuron).dict()
            })
        
        logger.info(f"Retrieved {len(neighbors)} neighbors for neuron {neuron_id}")
        
        return NeighborsResponse(
            neuron_id=neuron_id,
            neighbors=neighbors,
            count=len(neighbors)
        )
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid UUID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting neuron neighbors: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
