"""
Neuron management endpoints
"""
from fastapi import APIRouter, HTTPException, status
from uuid import UUID, uuid4
from datetime import datetime
import logging

from neuron_system.api.models import (
    NeuronCreateRequest,
    NeuronResponse,
    NeuronBatchCreateRequest,
    NeuronBatchResponse,
    SuccessResponse,
    Vector3DModel
)
from neuron_system.core.neuron import NeuronTypeRegistry
from neuron_system.core.vector3d import Vector3D
from neuron_system.neuron_types.knowledge_neuron import KnowledgeNeuron
from neuron_system.neuron_types.tool_neuron import ToolNeuron
from neuron_system.neuron_types.memory_neuron import MemoryNeuron

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
    elif isinstance(neuron, MemoryNeuron):
        response_data["sequence_data"] = neuron.sequence_data
        response_data["memory_type"] = neuron.memory_type
        response_data["retention_strength"] = neuron.retention_strength
        response_data["temporal_index"] = neuron.temporal_index
    
    return NeuronResponse(**response_data)


@router.post("/neurons", response_model=NeuronResponse, status_code=status.HTTP_201_CREATED)
async def create_neuron(request: NeuronCreateRequest):
    """
    Create a new neuron
    
    - **neuron_type**: Type of neuron (knowledge, tool, etc.)
    - **position**: Optional 3D position (auto-calculated if not provided)
    - **metadata**: Additional metadata
    - Type-specific fields based on neuron_type
    """
    state = get_app_state()
    
    try:
        # Determine position
        if request.position:
            position = Vector3D(
                x=request.position.x,
                y=request.position.y,
                z=request.position.z
            )
        else:
            # Auto-calculate position
            position = None
        
        # Create neuron based on type
        if request.neuron_type == "knowledge":
            if not request.source_data:
                raise ValueError("source_data is required for knowledge neurons")
            
            neuron = state.graph.add_neuron(
                neuron_type="knowledge",
                position=position,
                source_data=request.source_data,
                semantic_tags=request.semantic_tags or [],
                metadata=request.metadata
            )
            
        elif request.neuron_type == "tool":
            if not all([request.function_signature, request.executable_code, 
                       request.input_schema, request.output_schema]):
                raise ValueError(
                    "function_signature, executable_code, input_schema, and output_schema "
                    "are required for tool neurons"
                )
            
            neuron = state.graph.add_neuron(
                neuron_type="tool",
                position=position,
                function_signature=request.function_signature,
                executable_code=request.executable_code,
                input_schema=request.input_schema,
                output_schema=request.output_schema,
                metadata=request.metadata
            )
            
        elif request.neuron_type == "memory":
            neuron = state.graph.add_neuron(
                neuron_type="memory",
                position=position,
                sequence_data=request.sequence_data or [],
                memory_type=request.memory_type or "episodic",
                retention_strength=request.retention_strength or 1.0,
                max_sequence_length=request.max_sequence_length or 100,
                metadata=request.metadata
            )
            
        else:
            raise ValueError(f"Unknown neuron type: {request.neuron_type}")
        
        # Save to database
        state.neuron_store.save(neuron)
        
        logger.info(f"Created neuron {neuron.id} of type {request.neuron_type}")
        
        return neuron_to_response(neuron)
        
    except Exception as e:
        logger.error(f"Error creating neuron: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/neurons/batch", response_model=NeuronBatchResponse, status_code=status.HTTP_201_CREATED)
async def create_neurons_batch(request: NeuronBatchCreateRequest):
    """
    Create multiple neurons in a batch operation
    
    - **neurons**: List of neuron creation requests
    """
    state = get_app_state()
    created_ids = []
    
    try:
        for neuron_request in request.neurons:
            # Determine position
            if neuron_request.position:
                position = Vector3D(
                    x=neuron_request.position.x,
                    y=neuron_request.position.y,
                    z=neuron_request.position.z
                )
            else:
                position = None
            
            # Create neuron based on type
            if neuron_request.neuron_type == "knowledge":
                if not neuron_request.source_data:
                    raise ValueError("source_data is required for knowledge neurons")
                
                neuron = state.graph.add_neuron(
                    neuron_type="knowledge",
                    position=position,
                    source_data=neuron_request.source_data,
                    semantic_tags=neuron_request.semantic_tags or [],
                    metadata=neuron_request.metadata
                )
                
            elif neuron_request.neuron_type == "tool":
                if not all([neuron_request.function_signature, neuron_request.executable_code,
                           neuron_request.input_schema, neuron_request.output_schema]):
                    raise ValueError(
                        "function_signature, executable_code, input_schema, and output_schema "
                        "are required for tool neurons"
                    )
                
                neuron = state.graph.add_neuron(
                    neuron_type="tool",
                    position=position,
                    function_signature=neuron_request.function_signature,
                    executable_code=neuron_request.executable_code,
                    input_schema=neuron_request.input_schema,
                    output_schema=neuron_request.output_schema,
                    metadata=neuron_request.metadata
                )
            elif neuron_request.neuron_type == "memory":
                neuron = state.graph.add_neuron(
                    neuron_type="memory",
                    position=position,
                    sequence_data=neuron_request.sequence_data or [],
                    memory_type=neuron_request.memory_type or "episodic",
                    retention_strength=neuron_request.retention_strength or 1.0,
                    max_sequence_length=neuron_request.max_sequence_length or 100,
                    metadata=neuron_request.metadata
                )
            else:
                raise ValueError(f"Unknown neuron type: {neuron_request.neuron_type}")
            
            # Save to database
            state.neuron_store.save(neuron)
            created_ids.append(str(neuron.id))
        
        logger.info(f"Created {len(created_ids)} neurons in batch")
        
        return NeuronBatchResponse(
            created_ids=created_ids,
            count=len(created_ids)
        )
        
    except Exception as e:
        logger.error(f"Error creating neurons batch: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/neurons/{neuron_id}", response_model=NeuronResponse)
async def get_neuron(neuron_id: str):
    """
    Get a neuron by ID
    
    - **neuron_id**: UUID of the neuron
    """
    state = get_app_state()
    
    try:
        neuron_uuid = UUID(neuron_id)
        
        if neuron_uuid not in state.graph.neurons:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Neuron {neuron_id} not found"
            )
        
        neuron = state.graph.neurons[neuron_uuid]
        return neuron_to_response(neuron)
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid UUID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting neuron: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/neurons/{neuron_id}", response_model=SuccessResponse)
async def delete_neuron(neuron_id: str):
    """
    Delete a neuron by ID
    
    - **neuron_id**: UUID of the neuron
    - Automatically deletes all associated synapses
    """
    state = get_app_state()
    
    try:
        neuron_uuid = UUID(neuron_id)
        
        if neuron_uuid not in state.graph.neurons:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Neuron {neuron_id} not found"
            )
        
        # Remove neuron (this also removes associated synapses)
        state.graph.remove_neuron(neuron_uuid)
        
        # Delete from database
        state.neuron_store.delete(neuron_uuid)
        
        logger.info(f"Deleted neuron {neuron_id}")
        
        return SuccessResponse(
            success=True,
            message=f"Neuron {neuron_id} deleted successfully"
        )
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid UUID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting neuron: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
