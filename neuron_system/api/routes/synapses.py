"""
Synapse management endpoints
"""
from fastapi import APIRouter, HTTPException, status
from uuid import UUID
from typing import Optional
import logging

from neuron_system.api.models import (
    SynapseCreateRequest,
    SynapseResponse,
    SynapseQueryRequest,
    SynapseListResponse,
    SuccessResponse
)
from neuron_system.core.synapse import Synapse, SynapseType

logger = logging.getLogger(__name__)

router = APIRouter()


def get_app_state():
    """Get application state from app context"""
    from neuron_system.api.app import app_state
    return app_state


def synapse_to_response(synapse: Synapse) -> SynapseResponse:
    """Convert a Synapse object to SynapseResponse"""
    return SynapseResponse(
        id=str(synapse.id),
        source_neuron_id=str(synapse.source_neuron_id),
        target_neuron_id=str(synapse.target_neuron_id),
        weight=synapse.weight,
        usage_count=synapse.usage_count,
        last_traversed=synapse.last_traversed,
        synapse_type=synapse.synapse_type.value,
        metadata=synapse.metadata
    )


@router.post("/synapses", response_model=SynapseResponse, status_code=status.HTTP_201_CREATED)
async def create_synapse(request: SynapseCreateRequest):
    """
    Create a new synapse between two neurons
    
    - **source_neuron_id**: UUID of the source neuron
    - **target_neuron_id**: UUID of the target neuron
    - **weight**: Synapse weight between -1.0 and 1.0 (default: 0.5)
    - **synapse_type**: Type of synapse (default: KNOWLEDGE)
    - **metadata**: Additional metadata
    """
    state = get_app_state()
    
    try:
        source_uuid = UUID(request.source_neuron_id)
        target_uuid = UUID(request.target_neuron_id)
        
        # Validate neurons exist
        if source_uuid not in state.graph.neurons:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Source neuron {request.source_neuron_id} not found"
            )
        
        if target_uuid not in state.graph.neurons:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Target neuron {request.target_neuron_id} not found"
            )
        
        # Parse synapse type
        try:
            synapse_type = SynapseType[request.synapse_type]
        except KeyError:
            raise ValueError(f"Invalid synapse type: {request.synapse_type}")
        
        # Create synapse
        synapse = state.graph.add_synapse(
            source_neuron_id=source_uuid,
            target_neuron_id=target_uuid,
            weight=request.weight,
            synapse_type=synapse_type,
            metadata=request.metadata
        )
        
        # Save to database
        state.synapse_store.save(synapse)
        
        logger.info(f"Created synapse {synapse.id} from {source_uuid} to {target_uuid}")
        
        return synapse_to_response(synapse)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating synapse: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/synapses", response_model=SynapseListResponse)
async def query_synapses(
    source_neuron_id: Optional[str] = None,
    target_neuron_id: Optional[str] = None,
    min_weight: Optional[float] = None,
    max_weight: Optional[float] = None
):
    """
    Query synapses by various criteria
    
    - **source_neuron_id**: Filter by source neuron UUID
    - **target_neuron_id**: Filter by target neuron UUID
    - **min_weight**: Minimum synapse weight
    - **max_weight**: Maximum synapse weight
    """
    state = get_app_state()
    
    try:
        synapses = list(state.graph.synapses.values())
        
        # Apply filters
        if source_neuron_id:
            source_uuid = UUID(source_neuron_id)
            synapses = [s for s in synapses if s.source_neuron_id == source_uuid]
        
        if target_neuron_id:
            target_uuid = UUID(target_neuron_id)
            synapses = [s for s in synapses if s.target_neuron_id == target_uuid]
        
        if min_weight is not None:
            synapses = [s for s in synapses if s.weight >= min_weight]
        
        if max_weight is not None:
            synapses = [s for s in synapses if s.weight <= max_weight]
        
        synapse_responses = [synapse_to_response(s) for s in synapses]
        
        return SynapseListResponse(
            synapses=synapse_responses,
            count=len(synapse_responses)
        )
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid UUID format"
        )
    except Exception as e:
        logger.error(f"Error querying synapses: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/synapses/{synapse_id}", response_model=SynapseResponse)
async def get_synapse(synapse_id: str):
    """
    Get a synapse by ID
    
    - **synapse_id**: UUID of the synapse
    """
    state = get_app_state()
    
    try:
        synapse_uuid = UUID(synapse_id)
        
        if synapse_uuid not in state.graph.synapses:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Synapse {synapse_id} not found"
            )
        
        synapse = state.graph.synapses[synapse_uuid]
        return synapse_to_response(synapse)
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid UUID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting synapse: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/synapses/{synapse_id}", response_model=SuccessResponse)
async def delete_synapse(synapse_id: str):
    """
    Delete a synapse by ID
    
    - **synapse_id**: UUID of the synapse
    """
    state = get_app_state()
    
    try:
        synapse_uuid = UUID(synapse_id)
        
        if synapse_uuid not in state.graph.synapses:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Synapse {synapse_id} not found"
            )
        
        # Remove synapse from graph
        del state.graph.synapses[synapse_uuid]
        
        # Delete from database
        state.synapse_store.delete(synapse_uuid)
        
        logger.info(f"Deleted synapse {synapse_id}")
        
        return SuccessResponse(
            success=True,
            message=f"Synapse {synapse_id} deleted successfully"
        )
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid UUID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting synapse: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
