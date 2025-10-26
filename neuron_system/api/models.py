"""
Pydantic models for API request/response validation
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, validator


# ============================================================================
# Neuron Models
# ============================================================================

class Vector3DModel(BaseModel):
    """3D position vector"""
    x: float
    y: float
    z: float


class NeuronCreateRequest(BaseModel):
    """Request to create a new neuron"""
    neuron_type: str = Field(..., description="Type of neuron (knowledge, tool, memory, etc.)")
    position: Optional[Vector3DModel] = Field(None, description="3D position (auto-calculated if not provided)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Knowledge neuron specific
    source_data: Optional[str] = Field(None, description="Source data for knowledge neurons")
    semantic_tags: Optional[List[str]] = Field(None, description="Semantic tags for knowledge neurons")
    
    # Tool neuron specific
    function_signature: Optional[str] = Field(None, description="Function signature for tool neurons")
    executable_code: Optional[str] = Field(None, description="Executable code for tool neurons")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="Input schema for tool neurons")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="Output schema for tool neurons")
    
    # Memory neuron specific
    sequence_data: Optional[List[Dict[str, Any]]] = Field(None, description="Sequence data for memory neurons")
    memory_type: Optional[str] = Field("episodic", description="Type of memory (episodic, semantic, procedural)")
    retention_strength: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Memory retention strength")
    max_sequence_length: Optional[int] = Field(100, ge=1, description="Maximum sequence length for memory neurons")


class NeuronResponse(BaseModel):
    """Response containing neuron data"""
    id: str
    neuron_type: str
    position: Vector3DModel
    vector: Optional[List[float]] = None
    activation_level: float
    metadata: Dict[str, Any]
    created_at: datetime
    modified_at: datetime
    
    # Type-specific fields
    source_data: Optional[str] = None
    semantic_tags: Optional[List[str]] = None
    function_signature: Optional[str] = None
    execution_count: Optional[int] = None
    sequence_data: Optional[List[Dict[str, Any]]] = None
    memory_type: Optional[str] = None
    retention_strength: Optional[float] = None
    temporal_index: Optional[int] = None


class NeuronBatchCreateRequest(BaseModel):
    """Request to create multiple neurons"""
    neurons: List[NeuronCreateRequest]


class NeuronBatchResponse(BaseModel):
    """Response for batch neuron creation"""
    created_ids: List[str]
    count: int


# ============================================================================
# Synapse Models
# ============================================================================

class SynapseCreateRequest(BaseModel):
    """Request to create a new synapse"""
    source_neuron_id: str = Field(..., description="Source neuron UUID")
    target_neuron_id: str = Field(..., description="Target neuron UUID")
    weight: float = Field(0.5, ge=-1.0, le=1.0, description="Synapse weight between -1.0 and 1.0")
    synapse_type: str = Field("KNOWLEDGE", description="Type of synapse")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SynapseResponse(BaseModel):
    """Response containing synapse data"""
    id: str
    source_neuron_id: str
    target_neuron_id: str
    weight: float
    usage_count: int
    last_traversed: Optional[datetime]
    synapse_type: str
    metadata: Dict[str, Any]


class SynapseQueryRequest(BaseModel):
    """Request to query synapses"""
    source_neuron_id: Optional[str] = None
    target_neuron_id: Optional[str] = None
    min_weight: Optional[float] = None
    max_weight: Optional[float] = None


class SynapseListResponse(BaseModel):
    """Response containing list of synapses"""
    synapses: List[SynapseResponse]
    count: int


# ============================================================================
# Query Models
# ============================================================================

class QueryRequest(BaseModel):
    """Request to execute a knowledge query"""
    query_text: str = Field(..., description="Query text to search for")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    propagation_depth: int = Field(3, ge=1, le=10, description="Depth of activation propagation")
    neuron_type_filter: Optional[str] = Field(None, description="Filter by neuron type")


class SpatialQueryRequest(BaseModel):
    """Request to execute a spatial query"""
    center: Vector3DModel = Field(..., description="Center point of query")
    radius: float = Field(..., gt=0, description="Search radius")
    neuron_type_filter: Optional[str] = None


class ActivatedNeuronResponse(BaseModel):
    """Response for an activated neuron"""
    neuron: NeuronResponse
    activation_score: float
    tool_result: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """Response for query execution"""
    activated_neurons: List[ActivatedNeuronResponse]
    execution_time_ms: float
    query_id: str


class NeighborsResponse(BaseModel):
    """Response for neuron neighbors query"""
    neuron_id: str
    neighbors: List[Dict[str, Any]]  # Contains synapse + neuron data
    count: int


# ============================================================================
# Training Models
# ============================================================================

class AdjustNeuronRequest(BaseModel):
    """Request to adjust a neuron's vector"""
    neuron_id: str = Field(..., description="Neuron UUID to adjust")
    target_vector: Optional[List[float]] = Field(None, description="Target vector (384 dimensions)")
    target_text: Optional[str] = Field(None, description="Target text to compress into vector")
    learning_rate: float = Field(0.1, gt=0, le=1.0, description="Learning rate for adjustment")
    
    @validator('target_vector')
    def validate_vector_dimensions(cls, v):
        if v is not None and len(v) != 384:
            raise ValueError("Vector must have exactly 384 dimensions")
        return v


class AdjustSynapseRequest(BaseModel):
    """Request to adjust a synapse's weight"""
    synapse_id: str = Field(..., description="Synapse UUID to adjust")
    operation: str = Field(..., description="Operation: 'strengthen' or 'weaken' or 'set'")
    delta: Optional[float] = Field(None, description="Delta for strengthen/weaken operations")
    new_weight: Optional[float] = Field(None, ge=-1.0, le=1.0, description="New weight for set operation")
    
    @validator('operation')
    def validate_operation(cls, v):
        if v not in ['strengthen', 'weaken', 'set']:
            raise ValueError("Operation must be 'strengthen', 'weaken', or 'set'")
        return v


class CreateToolNeuronRequest(BaseModel):
    """Request to create a new tool neuron"""
    description: str = Field(..., description="Description of the tool's functionality")
    function_signature: str = Field(..., description="Function signature")
    executable_code: str = Field(..., description="Python code to execute")
    input_schema: Dict[str, Any] = Field(..., description="JSON schema for inputs")
    output_schema: Dict[str, Any] = Field(..., description="JSON schema for outputs")
    position: Optional[Vector3DModel] = None
    connect_to_neurons: Optional[List[str]] = Field(None, description="Neuron IDs to connect to")


class TrainingOperationResponse(BaseModel):
    """Response for training operations"""
    success: bool
    operation_id: str
    message: str
    details: Optional[Dict[str, Any]] = None


# ============================================================================
# General Models
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    neuron_count: int
    synapse_count: int
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Error response"""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime
    recoverable: bool = True


class SuccessResponse(BaseModel):
    """Generic success response"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


# ============================================================================
# Visualization Models
# ============================================================================

class VisualizationNeuronResponse(BaseModel):
    """Neuron data for visualization"""
    id: str
    neuron_type: str
    position: Vector3DModel
    activation_level: float
    metadata: Dict[str, Any]
    
    # Visual properties
    color: Optional[str] = None  # Hex color code
    size: Optional[float] = None  # Relative size
    label: Optional[str] = None  # Display label


class VisualizationNeuronsResponse(BaseModel):
    """Response for neuron visualization data"""
    neurons: List[VisualizationNeuronResponse]
    count: int
    bounds: Dict[str, Vector3DModel]  # min and max bounds


class VisualizationSynapseResponse(BaseModel):
    """Synapse data for visualization"""
    id: str
    source_neuron_id: str
    target_neuron_id: str
    weight: float
    synapse_type: str
    
    # Visual properties
    color: Optional[str] = None  # Hex color code
    thickness: Optional[float] = None  # Line thickness


class VisualizationSynapsesResponse(BaseModel):
    """Response for synapse visualization data"""
    synapses: List[VisualizationSynapseResponse]
    count: int


class ClusterBoundary(BaseModel):
    """Boundary information for a cluster"""
    center: Vector3DModel
    radius: float
    neuron_count: int


class VisualizationClusterResponse(BaseModel):
    """Cluster data for visualization"""
    cluster_id: str
    name: str
    neurons: List[str]  # Neuron IDs
    boundary: ClusterBoundary
    metadata: Dict[str, Any]


class VisualizationClustersResponse(BaseModel):
    """Response for cluster visualization data"""
    clusters: List[VisualizationClusterResponse]
    count: int
