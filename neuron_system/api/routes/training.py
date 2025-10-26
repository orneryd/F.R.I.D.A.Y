"""
Training endpoints for neuron and synapse modification
"""
from fastapi import APIRouter, HTTPException, status
from uuid import UUID
import logging
import numpy as np

from neuron_system.api.models import (
    AdjustNeuronRequest,
    AdjustSynapseRequest,
    CreateToolNeuronRequest,
    TrainingOperationResponse,
    Vector3DModel
)
from neuron_system.core.vector3d import Vector3D

logger = logging.getLogger(__name__)

router = APIRouter()


def get_app_state():
    """Get application state from app context"""
    from neuron_system.api.app import app_state
    return app_state


@router.post("/training/adjust-neuron", response_model=TrainingOperationResponse)
async def adjust_neuron(request: AdjustNeuronRequest):
    """
    Adjust a neuron's vector toward a target.
    
    Incrementally moves the neuron vector closer to the target using:
    new_vector = current_vector + learning_rate * (target - current)
    
    - **neuron_id**: UUID of the neuron to adjust
    - **target_vector**: Target vector (384 dimensions) OR
    - **target_text**: Target text to compress into vector
    - **learning_rate**: Learning rate for adjustment (0.0 to 1.0)
    
    Requirements: 4.1, 4.2, 4.3, 4.5
    """
    state = get_app_state()
    
    try:
        neuron_uuid = UUID(request.neuron_id)
        
        # Check if neuron exists
        if neuron_uuid not in state.graph.neurons:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Neuron {request.neuron_id} not found"
            )
        
        # Determine target vector
        if request.target_vector is not None:
            target_vector = np.array(request.target_vector)
        elif request.target_text is not None:
            # Compress text to vector
            vector, compression_meta = state.compression_engine.compress(
                request.target_text,
                normalize=True
            )
            if not compression_meta.get("success", False):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to compress target text"
                )
            target_vector = vector
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either target_vector or target_text must be provided"
            )
        
        # Perform adjustment
        success = state.training_engine.adjust_neuron(
            neuron_id=neuron_uuid,
            target_vector=target_vector,
            learning_rate=request.learning_rate,
            validate=True
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to adjust neuron vector"
            )
        
        # Get the last operation ID
        operation_id = "unknown"
        if state.training_engine.operation_log:
            operation_id = str(state.training_engine.operation_log[-1].operation_id)
        
        # Save updated neuron to database
        neuron = state.graph.neurons[neuron_uuid]
        state.neuron_store.save(neuron)
        
        logger.info(f"Adjusted neuron {request.neuron_id} with learning_rate={request.learning_rate}")
        
        return TrainingOperationResponse(
            success=True,
            operation_id=operation_id,
            message=f"Neuron {request.neuron_id} adjusted successfully",
            details={
                "neuron_id": request.neuron_id,
                "learning_rate": request.learning_rate,
                "used_text_compression": request.target_text is not None
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adjusting neuron: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/training/adjust-synapse", response_model=TrainingOperationResponse)
async def adjust_synapse(request: AdjustSynapseRequest):
    """
    Modify a synapse's weight.
    
    Supports three operations:
    - **strengthen**: Increase weight by delta (Hebbian learning)
    - **weaken**: Decrease weight by delta (decay)
    - **set**: Set weight to specific value
    
    - **synapse_id**: UUID of the synapse to adjust
    - **operation**: Operation type ('strengthen', 'weaken', or 'set')
    - **delta**: Amount to change weight (for strengthen/weaken)
    - **new_weight**: New weight value (for set operation)
    
    Note: Synapses with weight near 0.0 are automatically deleted.
    
    Requirements: 9.2, 9.3, 9.4
    """
    state = get_app_state()
    
    try:
        synapse_uuid = UUID(request.synapse_id)
        
        # Check if synapse exists
        if synapse_uuid not in state.graph.synapses:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Synapse {request.synapse_id} not found"
            )
        
        # Store initial weight for response
        initial_weight = state.graph.synapses[synapse_uuid].weight
        
        # Perform operation
        success = False
        operation_details = {}
        
        if request.operation == "strengthen":
            if request.delta is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="delta is required for strengthen operation"
                )
            success = state.training_engine.strengthen_synapse(
                synapse_id=synapse_uuid,
                delta=request.delta,
                validate=True
            )
            operation_details["delta"] = request.delta
            
        elif request.operation == "weaken":
            if request.delta is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="delta is required for weaken operation"
                )
            success = state.training_engine.weaken_synapse(
                synapse_id=synapse_uuid,
                delta=request.delta,
                validate=True
            )
            operation_details["delta"] = request.delta
            
        elif request.operation == "set":
            if request.new_weight is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="new_weight is required for set operation"
                )
            success = state.training_engine.adjust_synapse_weight(
                synapse_id=synapse_uuid,
                new_weight=request.new_weight,
                validate=True
            )
            operation_details["new_weight"] = request.new_weight
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to {request.operation} synapse"
            )
        
        # Get the last operation ID
        operation_id = "unknown"
        if state.training_engine.operation_log:
            operation_id = str(state.training_engine.operation_log[-1].operation_id)
        
        # Check if synapse still exists (might have been deleted if weight reached 0)
        synapse_deleted = synapse_uuid not in state.graph.synapses
        final_weight = None if synapse_deleted else state.graph.synapses[synapse_uuid].weight
        
        # Save updated synapse to database (if not deleted)
        if not synapse_deleted:
            synapse = state.graph.synapses[synapse_uuid]
            state.synapse_store.save(synapse)
        else:
            # Delete from database
            state.synapse_store.delete(synapse_uuid)
        
        logger.info(
            f"Adjusted synapse {request.synapse_id} with operation={request.operation}"
            f"{' (deleted)' if synapse_deleted else ''}"
        )
        
        return TrainingOperationResponse(
            success=True,
            operation_id=operation_id,
            message=f"Synapse {request.synapse_id} {request.operation} operation completed"
                    f"{' and deleted (weight reached 0)' if synapse_deleted else ''}",
            details={
                "synapse_id": request.synapse_id,
                "operation": request.operation,
                "initial_weight": initial_weight,
                "final_weight": final_weight,
                "deleted": synapse_deleted,
                **operation_details
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adjusting synapse: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/training/create-tool", response_model=TrainingOperationResponse)
async def create_tool_neuron(request: CreateToolNeuronRequest):
    """
    Create a new tool neuron from description and code.
    
    Dynamically creates a tool neuron, validates its code, and optionally
    connects it to relevant knowledge neurons based on semantic similarity.
    
    - **description**: Natural language description of the tool's functionality
    - **function_signature**: Function signature (e.g., "calculate(x: float, y: float)")
    - **executable_code**: Python code to execute (must set 'result' variable)
    - **input_schema**: JSON Schema for input validation
    - **output_schema**: JSON Schema for output validation
    - **position**: Optional 3D position (auto-calculated if not provided)
    - **connect_to_neurons**: Optional list of neuron IDs to connect to
    
    Requirements: 10.5, 12.1, 12.2
    """
    state = get_app_state()
    
    try:
        # Determine position
        position = None
        if request.position:
            position = Vector3D(
                x=request.position.x,
                y=request.position.y,
                z=request.position.z
            )
        
        # Create the tool neuron using training engine
        tool_neuron_id = state.training_engine.create_tool_neuron(
            description=request.description,
            code=request.executable_code,
            position=position,
            input_schema=request.input_schema,
            output_schema=request.output_schema,
            auto_connect=True,  # Auto-connect to relevant neurons
            connection_threshold=0.7
        )
        
        # Get the created neuron
        tool_neuron = state.graph.neurons[tool_neuron_id]
        
        # Save to database
        state.neuron_store.save(tool_neuron)
        
        # Create manual connections if specified
        manual_connections = 0
        if request.connect_to_neurons:
            from neuron_system.core.synapse import Synapse, SynapseType
            from uuid import uuid4
            
            for neuron_id_str in request.connect_to_neurons:
                try:
                    neuron_id = UUID(neuron_id_str)
                    if neuron_id in state.graph.neurons:
                        synapse = Synapse(
                            id=uuid4(),
                            source_neuron_id=neuron_id,
                            target_neuron_id=tool_neuron_id,
                            weight=0.8,
                            synapse_type=SynapseType.TOOL_INPUT,
                            metadata={"manually_created": True}
                        )
                        state.graph.add_synapse(synapse)
                        state.synapse_store.save(synapse)
                        manual_connections += 1
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to connect to neuron {neuron_id_str}: {str(e)}")
        
        # Get the last operation ID
        operation_id = "unknown"
        if state.training_engine.operation_log:
            operation_id = str(state.training_engine.operation_log[-1].operation_id)
        
        logger.info(
            f"Created tool neuron {tool_neuron_id} with {manual_connections} manual connections"
        )
        
        return TrainingOperationResponse(
            success=True,
            operation_id=operation_id,
            message=f"Tool neuron created successfully",
            details={
                "tool_neuron_id": str(tool_neuron_id),
                "description": request.description,
                "function_signature": request.function_signature,
                "position": {
                    "x": tool_neuron.position.x,
                    "y": tool_neuron.position.y,
                    "z": tool_neuron.position.z
                },
                "manual_connections": manual_connections,
                "auto_connected": True
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating tool neuron: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
