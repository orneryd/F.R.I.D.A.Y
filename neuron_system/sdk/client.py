"""
Python SDK client for the 3D Synaptic Neuron System API

This client provides a convenient interface for interacting with the Neuron System API,
handling authentication, request formatting, and response parsing automatically.
"""
import requests
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
import logging

from neuron_system.sdk.exceptions import (
    NeuronSystemError,
    ConnectionError,
    AuthenticationError,
    NotFoundError,
    ValidationError,
    ServerError,
    TimeoutError
)

logger = logging.getLogger(__name__)


class NeuronSystemClient:
    """
    Client for interacting with the 3D Synaptic Neuron System API.
    
    This client provides methods for all API endpoints with automatic authentication,
    error handling, and response parsing.
    
    Example:
        >>> client = NeuronSystemClient(base_url="http://localhost:8000", api_key="your-key")
        >>> neuron = client.create_neuron(
        ...     neuron_type="knowledge",
        ...     source_data="Python is a programming language"
        ... )
        >>> print(neuron["id"])
    
    Args:
        base_url: Base URL of the API (e.g., "http://localhost:8000")
        api_key: Optional API key for authentication
        timeout: Request timeout in seconds (default: 30)
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        """Initialize the Neuron System client"""
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        
        # Set up authentication headers
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})
        
        # Set default headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the API with error handling.
        
        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint path
            data: Request body data (for POST/PUT)
            params: Query parameters (for GET)
        
        Returns:
            Parsed JSON response
        
        Raises:
            ConnectionError: If connection fails
            AuthenticationError: If authentication fails (401)
            NotFoundError: If resource not found (404)
            ValidationError: If request validation fails (400)
            ServerError: If server error occurs (500)
            TimeoutError: If request times out
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                timeout=self.timeout
            )
            
            # Handle different status codes
            if response.status_code == 200 or response.status_code == 201:
                return response.json()
            elif response.status_code == 401:
                raise AuthenticationError("Authentication failed. Check your API key.")
            elif response.status_code == 404:
                error_detail = response.json().get("detail", "Resource not found")
                raise NotFoundError(error_detail)
            elif response.status_code == 400:
                error_detail = response.json().get("detail", "Invalid request")
                raise ValidationError(error_detail)
            elif response.status_code >= 500:
                error_detail = response.json().get("message", "Server error")
                raise ServerError(f"Server error: {error_detail}")
            else:
                raise NeuronSystemError(
                    f"Unexpected status code {response.status_code}: {response.text}"
                )
        
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timed out after {self.timeout} seconds")
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to {url}: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise NeuronSystemError(f"Request failed: {str(e)}")
    
    # ========================================================================
    # Health and Status
    # ========================================================================
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health status of the API.
        
        Returns:
            Health status information including neuron and synapse counts
        
        Example:
            >>> health = client.health_check()
            >>> print(f"Status: {health['status']}, Neurons: {health['neuron_count']}")
        """
        return self._make_request("GET", "/health")
    
    # ========================================================================
    # Neuron Operations
    # ========================================================================
    
    def create_neuron(
        self,
        neuron_type: str,
        position: Optional[Tuple[float, float, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a new neuron.
        
        Args:
            neuron_type: Type of neuron ("knowledge" or "tool")
            position: Optional 3D position as (x, y, z) tuple
            metadata: Optional metadata dictionary
            **kwargs: Type-specific fields:
                For knowledge neurons:
                    - source_data (str): Source data to store
                    - semantic_tags (List[str]): Optional semantic tags
                For tool neurons:
                    - function_signature (str): Function signature
                    - executable_code (str): Python code to execute
                    - input_schema (dict): JSON schema for inputs
                    - output_schema (dict): JSON schema for outputs
        
        Returns:
            Created neuron data including ID and position
        
        Example:
            >>> neuron = client.create_neuron(
            ...     neuron_type="knowledge",
            ...     source_data="Machine learning is a subset of AI"
            ... )
        """
        data = {
            "neuron_type": neuron_type,
            "metadata": metadata or {},
            **kwargs
        }
        
        if position:
            data["position"] = {"x": position[0], "y": position[1], "z": position[2]}
        
        return self._make_request("POST", "/api/v1/neurons", data=data)
    
    def get_neuron(self, neuron_id: str) -> Dict[str, Any]:
        """
        Get a neuron by ID.
        
        Args:
            neuron_id: UUID of the neuron
        
        Returns:
            Neuron data
        
        Example:
            >>> neuron = client.get_neuron("123e4567-e89b-12d3-a456-426614174000")
        """
        return self._make_request("GET", f"/api/v1/neurons/{neuron_id}")
    
    def delete_neuron(self, neuron_id: str) -> Dict[str, Any]:
        """
        Delete a neuron by ID.
        
        This automatically deletes all associated synapses.
        
        Args:
            neuron_id: UUID of the neuron
        
        Returns:
            Success response
        
        Example:
            >>> result = client.delete_neuron("123e4567-e89b-12d3-a456-426614174000")
        """
        return self._make_request("DELETE", f"/api/v1/neurons/{neuron_id}")
    
    def create_neurons_batch(
        self,
        neurons: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create multiple neurons in a batch operation.
        
        Args:
            neurons: List of neuron creation requests
        
        Returns:
            Batch creation response with created IDs
        
        Example:
            >>> result = client.create_neurons_batch([
            ...     {"neuron_type": "knowledge", "source_data": "Data 1"},
            ...     {"neuron_type": "knowledge", "source_data": "Data 2"}
            ... ])
        """
        return self._make_request(
            "POST",
            "/api/v1/neurons/batch",
            data={"neurons": neurons}
        )
    
    # ========================================================================
    # Synapse Operations
    # ========================================================================
    
    def create_synapse(
        self,
        source_neuron_id: str,
        target_neuron_id: str,
        weight: float = 0.5,
        synapse_type: str = "KNOWLEDGE",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a synapse between two neurons.
        
        Args:
            source_neuron_id: UUID of the source neuron
            target_neuron_id: UUID of the target neuron
            weight: Synapse weight between -1.0 and 1.0 (default: 0.5)
            synapse_type: Type of synapse (default: "KNOWLEDGE")
            metadata: Optional metadata dictionary
        
        Returns:
            Created synapse data
        
        Example:
            >>> synapse = client.create_synapse(
            ...     source_neuron_id="123e4567-...",
            ...     target_neuron_id="987f6543-...",
            ...     weight=0.8
            ... )
        """
        data = {
            "source_neuron_id": source_neuron_id,
            "target_neuron_id": target_neuron_id,
            "weight": weight,
            "synapse_type": synapse_type,
            "metadata": metadata or {}
        }
        return self._make_request("POST", "/api/v1/synapses", data=data)
    
    def get_synapse(self, synapse_id: str) -> Dict[str, Any]:
        """
        Get a synapse by ID.
        
        Args:
            synapse_id: UUID of the synapse
        
        Returns:
            Synapse data
        """
        return self._make_request("GET", f"/api/v1/synapses/{synapse_id}")
    
    def query_synapses(
        self,
        source_neuron_id: Optional[str] = None,
        target_neuron_id: Optional[str] = None,
        min_weight: Optional[float] = None,
        max_weight: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Query synapses by various criteria.
        
        Args:
            source_neuron_id: Filter by source neuron UUID
            target_neuron_id: Filter by target neuron UUID
            min_weight: Minimum synapse weight
            max_weight: Maximum synapse weight
        
        Returns:
            List of matching synapses
        
        Example:
            >>> synapses = client.query_synapses(source_neuron_id="123e4567-...")
        """
        params = {}
        if source_neuron_id:
            params["source_neuron_id"] = source_neuron_id
        if target_neuron_id:
            params["target_neuron_id"] = target_neuron_id
        if min_weight is not None:
            params["min_weight"] = min_weight
        if max_weight is not None:
            params["max_weight"] = max_weight
        
        return self._make_request("GET", "/api/v1/synapses", params=params)
    
    def delete_synapse(self, synapse_id: str) -> Dict[str, Any]:
        """
        Delete a synapse by ID.
        
        Args:
            synapse_id: UUID of the synapse
        
        Returns:
            Success response
        """
        return self._make_request("DELETE", f"/api/v1/synapses/{synapse_id}")
    
    # ========================================================================
    # Query Operations
    # ========================================================================
    
    def query(
        self,
        query_text: str,
        top_k: int = 10,
        propagation_depth: int = 3,
        neuron_type_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a knowledge query against the neuron network.
        
        Args:
            query_text: Text to search for
            top_k: Number of results to return (1-100, default: 10)
            propagation_depth: Depth of activation propagation (1-10, default: 3)
            neuron_type_filter: Optional filter by neuron type
        
        Returns:
            Query results with activated neurons and execution time
        
        Example:
            >>> results = client.query("What is machine learning?", top_k=5)
            >>> for result in results["activated_neurons"]:
            ...     print(f"Score: {result['activation_score']}")
        """
        data = {
            "query_text": query_text,
            "top_k": top_k,
            "propagation_depth": propagation_depth
        }
        if neuron_type_filter:
            data["neuron_type_filter"] = neuron_type_filter
        
        return self._make_request("POST", "/api/v1/query", data=data)
    
    def spatial_query(
        self,
        center: Tuple[float, float, float],
        radius: float,
        neuron_type_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a spatial query to find neurons within a 3D region.
        
        Args:
            center: Center point as (x, y, z) tuple
            radius: Search radius
            neuron_type_filter: Optional filter by neuron type
        
        Returns:
            Query results with neurons in the spatial region
        
        Example:
            >>> results = client.spatial_query(center=(0, 0, 0), radius=10.0)
        """
        data = {
            "center": {"x": center[0], "y": center[1], "z": center[2]},
            "radius": radius
        }
        if neuron_type_filter:
            data["neuron_type_filter"] = neuron_type_filter
        
        return self._make_request("POST", "/api/v1/query/spatial", data=data)
    
    def get_neighbors(self, neuron_id: str) -> Dict[str, Any]:
        """
        Get all neurons connected to a specific neuron via synapses.
        
        Args:
            neuron_id: UUID of the neuron
        
        Returns:
            List of connected neurons with synapse information
        
        Example:
            >>> neighbors = client.get_neighbors("123e4567-...")
            >>> print(f"Found {neighbors['count']} neighbors")
        """
        return self._make_request("GET", f"/api/v1/neurons/{neuron_id}/neighbors")
    
    # ========================================================================
    # Training Operations
    # ========================================================================
    
    def adjust_neuron(
        self,
        neuron_id: str,
        target_vector: Optional[List[float]] = None,
        target_text: Optional[str] = None,
        learning_rate: float = 0.1
    ) -> Dict[str, Any]:
        """
        Adjust a neuron's vector toward a target.
        
        Args:
            neuron_id: UUID of the neuron to adjust
            target_vector: Target vector (384 dimensions) OR
            target_text: Target text to compress into vector
            learning_rate: Learning rate for adjustment (0.0 to 1.0)
        
        Returns:
            Training operation response
        
        Example:
            >>> result = client.adjust_neuron(
            ...     neuron_id="123e4567-...",
            ...     target_text="Updated knowledge",
            ...     learning_rate=0.2
            ... )
        """
        data = {
            "neuron_id": neuron_id,
            "learning_rate": learning_rate
        }
        if target_vector is not None:
            data["target_vector"] = target_vector
        if target_text is not None:
            data["target_text"] = target_text
        
        return self._make_request("POST", "/api/v1/training/adjust-neuron", data=data)
    
    def adjust_synapse(
        self,
        synapse_id: str,
        operation: str,
        delta: Optional[float] = None,
        new_weight: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Modify a synapse's weight.
        
        Args:
            synapse_id: UUID of the synapse to adjust
            operation: Operation type ('strengthen', 'weaken', or 'set')
            delta: Amount to change weight (for strengthen/weaken)
            new_weight: New weight value (for set operation)
        
        Returns:
            Training operation response
        
        Example:
            >>> result = client.adjust_synapse(
            ...     synapse_id="123e4567-...",
            ...     operation="strengthen",
            ...     delta=0.1
            ... )
        """
        data = {
            "synapse_id": synapse_id,
            "operation": operation
        }
        if delta is not None:
            data["delta"] = delta
        if new_weight is not None:
            data["new_weight"] = new_weight
        
        return self._make_request("POST", "/api/v1/training/adjust-synapse", data=data)
    
    def create_tool_neuron(
        self,
        description: str,
        function_signature: str,
        executable_code: str,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        position: Optional[Tuple[float, float, float]] = None,
        connect_to_neurons: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a new tool neuron from description and code.
        
        Args:
            description: Natural language description of the tool's functionality
            function_signature: Function signature
            executable_code: Python code to execute
            input_schema: JSON Schema for input validation
            output_schema: JSON Schema for output validation
            position: Optional 3D position as (x, y, z) tuple
            connect_to_neurons: Optional list of neuron IDs to connect to
        
        Returns:
            Training operation response with tool neuron ID
        
        Example:
            >>> result = client.create_tool_neuron(
            ...     description="Calculate sum of two numbers",
            ...     function_signature="add(a: float, b: float)",
            ...     executable_code="result = a + b",
            ...     input_schema={"type": "object", "properties": {...}},
            ...     output_schema={"type": "number"}
            ... )
        """
        data = {
            "description": description,
            "function_signature": function_signature,
            "executable_code": executable_code,
            "input_schema": input_schema,
            "output_schema": output_schema
        }
        if position:
            data["position"] = {"x": position[0], "y": position[1], "z": position[2]}
        if connect_to_neurons:
            data["connect_to_neurons"] = connect_to_neurons
        
        return self._make_request("POST", "/api/v1/training/create-tool", data=data)
    
    def close(self):
        """Close the client session"""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


    # ========================================================================
    # High-Level Convenience Methods
    # ========================================================================
    
    def add_knowledge(
        self,
        text: str,
        tags: Optional[List[str]] = None,
        position: Optional[Tuple[float, float, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        High-level method to add knowledge to the system.
        
        This is a convenience wrapper around create_neuron for knowledge neurons.
        
        Args:
            text: Knowledge text to store
            tags: Optional semantic tags
            position: Optional 3D position as (x, y, z) tuple
            metadata: Optional metadata dictionary
        
        Returns:
            Created knowledge neuron data
        
        Example:
            >>> neuron = client.add_knowledge(
            ...     text="Python is a high-level programming language",
            ...     tags=["programming", "python"]
            ... )
            >>> print(f"Created neuron: {neuron['id']}")
        """
        return self.create_neuron(
            neuron_type="knowledge",
            source_data=text,
            semantic_tags=tags or [],
            position=position,
            metadata=metadata or {}
        )
    
    def add_tool(
        self,
        name: str,
        description: str,
        code: str,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        position: Optional[Tuple[float, float, float]] = None,
        auto_connect: bool = True
    ) -> Dict[str, Any]:
        """
        High-level method to add a tool to the system.
        
        This is a convenience wrapper around create_tool_neuron.
        
        Args:
            name: Name of the tool (used as function signature)
            description: Description of what the tool does
            code: Python code to execute (must set 'result' variable)
            input_schema: JSON Schema for input validation
            output_schema: JSON Schema for output validation
            position: Optional 3D position as (x, y, z) tuple
            auto_connect: Whether to auto-connect to relevant neurons (default: True)
        
        Returns:
            Training operation response with tool neuron ID
        
        Example:
            >>> result = client.add_tool(
            ...     name="add",
            ...     description="Add two numbers",
            ...     code="result = a + b",
            ...     input_schema={
            ...         "type": "object",
            ...         "properties": {
            ...             "a": {"type": "number"},
            ...             "b": {"type": "number"}
            ...         }
            ...     },
            ...     output_schema={"type": "number"}
            ... )
            >>> tool_id = result['details']['tool_neuron_id']
        """
        return self.create_tool_neuron(
            description=description,
            function_signature=name,
            executable_code=code,
            input_schema=input_schema,
            output_schema=output_schema,
            position=position,
            connect_to_neurons=None  # Let auto-connect handle it
        )
    
    def train(
        self,
        neuron_id: str,
        new_knowledge: str,
        learning_rate: float = 0.1
    ) -> Dict[str, Any]:
        """
        High-level method to train a neuron with new knowledge.
        
        This is a convenience wrapper around adjust_neuron that uses text input.
        
        Args:
            neuron_id: UUID of the neuron to train
            new_knowledge: New knowledge text to train toward
            learning_rate: Learning rate for adjustment (0.0 to 1.0, default: 0.1)
        
        Returns:
            Training operation response
        
        Example:
            >>> result = client.train(
            ...     neuron_id="123e4567-...",
            ...     new_knowledge="Python 3.12 was released in 2023",
            ...     learning_rate=0.2
            ... )
            >>> print(f"Training successful: {result['success']}")
        """
        return self.adjust_neuron(
            neuron_id=neuron_id,
            target_text=new_knowledge,
            learning_rate=learning_rate
        )
    
    def search(
        self,
        text: str,
        limit: int = 10,
        depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        High-level method to search for knowledge in the system.
        
        This is a convenience wrapper around query that returns a simplified result list.
        
        Args:
            text: Search query text
            limit: Maximum number of results (default: 10)
            depth: Propagation depth (default: 3)
        
        Returns:
            List of activated neurons with their scores and content
        
        Example:
            >>> results = client.search("machine learning", limit=5)
            >>> for result in results:
            ...     print(f"Score: {result['score']:.2f}")
            ...     print(f"Content: {result['content']}")
        """
        response = self.query(
            query_text=text,
            top_k=limit,
            propagation_depth=depth
        )
        
        # Simplify the response format
        simplified_results = []
        for activated in response["activated_neurons"]:
            neuron = activated["neuron"]
            result = {
                "id": neuron["id"],
                "score": activated["activation_score"],
                "type": neuron["neuron_type"],
                "position": (
                    neuron["position"]["x"],
                    neuron["position"]["y"],
                    neuron["position"]["z"]
                ),
                "metadata": neuron["metadata"]
            }
            
            # Add type-specific content
            if neuron.get("source_data"):
                result["content"] = neuron["source_data"]
                result["tags"] = neuron.get("semantic_tags", [])
            elif neuron.get("function_signature"):
                result["content"] = neuron["function_signature"]
                result["tool_result"] = activated.get("tool_result")
            
            simplified_results.append(result)
        
        return simplified_results
    
    def connect(
        self,
        from_neuron_id: str,
        to_neuron_id: str,
        strength: float = 0.5
    ) -> Dict[str, Any]:
        """
        High-level method to connect two neurons.
        
        This is a convenience wrapper around create_synapse.
        
        Args:
            from_neuron_id: UUID of the source neuron
            to_neuron_id: UUID of the target neuron
            strength: Connection strength between -1.0 and 1.0 (default: 0.5)
        
        Returns:
            Created synapse data
        
        Example:
            >>> synapse = client.connect(
            ...     from_neuron_id="123e4567-...",
            ...     to_neuron_id="987f6543-...",
            ...     strength=0.8
            ... )
        """
        return self.create_synapse(
            source_neuron_id=from_neuron_id,
            target_neuron_id=to_neuron_id,
            weight=strength
        )
    
    def strengthen_connection(
        self,
        synapse_id: str,
        amount: float = 0.1
    ) -> Dict[str, Any]:
        """
        High-level method to strengthen a connection between neurons.
        
        This is a convenience wrapper around adjust_synapse.
        
        Args:
            synapse_id: UUID of the synapse
            amount: Amount to strengthen (default: 0.1)
        
        Returns:
            Training operation response
        
        Example:
            >>> result = client.strengthen_connection("123e4567-...", amount=0.2)
        """
        return self.adjust_synapse(
            synapse_id=synapse_id,
            operation="strengthen",
            delta=amount
        )
    
    def weaken_connection(
        self,
        synapse_id: str,
        amount: float = 0.1
    ) -> Dict[str, Any]:
        """
        High-level method to weaken a connection between neurons.
        
        This is a convenience wrapper around adjust_synapse.
        
        Args:
            synapse_id: UUID of the synapse
            amount: Amount to weaken (default: 0.1)
        
        Returns:
            Training operation response
        
        Example:
            >>> result = client.weaken_connection("123e4567-...", amount=0.05)
        """
        return self.adjust_synapse(
            synapse_id=synapse_id,
            operation="weaken",
            delta=amount
        )
    
    def get_network_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the neuron network.
        
        Returns:
            Network statistics including counts and health status
        
        Example:
            >>> stats = client.get_network_stats()
            >>> print(f"Neurons: {stats['neuron_count']}")
            >>> print(f"Synapses: {stats['synapse_count']}")
        """
        health = self.health_check()
        return {
            "status": health["status"],
            "neuron_count": health["neuron_count"],
            "synapse_count": health["synapse_count"],
            "version": health["version"],
            "timestamp": health["timestamp"]
        }
