"""
ToolNeuron implementation for executable functions.
"""

from typing import Any, Dict, List
from uuid import UUID, uuid4
from datetime import datetime
import numpy as np

from neuron_system.core.neuron import Neuron, NeuronType, NeuronTypeRegistry
from neuron_system.core.vector3d import Vector3D


class ToolNeuron(Neuron):
    """
    Executes functions/tools when activated.
    
    Tool neurons represent executable capabilities embedded in the network.
    """
    
    def __init__(self,
                 function_signature: str = "",
                 executable_code: str = "",
                 input_schema: Dict[str, Any] = None,
                 output_schema: Dict[str, Any] = None,
                 execution_count: int = 0,
                 average_execution_time: float = 0.0,
                 activation_threshold: float = 0.5,
                 **kwargs):
        """
        Initialize tool neuron.
        
        Args:
            function_signature: Function signature description
            executable_code: Python code or reference to execute
            input_schema: JSON Schema for input validation
            output_schema: JSON Schema for output validation
            execution_count: Number of times tool has been executed
            average_execution_time: Average execution time in milliseconds
            activation_threshold: Minimum activation to trigger execution
            **kwargs: Additional arguments
        """
        super().__init__()
        self.neuron_type = NeuronType.TOOL
        self.function_signature = function_signature
        self.executable_code = executable_code
        self.input_schema = input_schema or {}
        self.output_schema = output_schema or {}
        self.execution_count = execution_count
        self.average_execution_time = average_execution_time
        self.activation_threshold = activation_threshold
        
        # Set defaults
        if not self.id:
            self.id = uuid4()
        if not self.created_at:
            self.created_at = datetime.now()
        if not self.modified_at:
            self.modified_at = datetime.now()
    
    def process_activation(self, activation: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tool when activation exceeds threshold.
        
        Args:
            activation: Activation level (0.0 to 1.0)
            context: Additional context including inputs
            
        Returns:
            Dictionary with execution result or reason for not executing
        """
        self.activation_level = activation
        
        if activation < self.activation_threshold:
            return {
                "type": "tool",
                "neuron_id": str(self.id),
                "executed": False,
                "reason": "activation_too_low",
                "activation": activation,
                "threshold": self.activation_threshold,
            }
        
        try:
            result = self.execute(context.get("inputs", {}))
            self.execution_count += 1
            self.modified_at = datetime.now()
            
            return {
                "type": "tool",
                "neuron_id": str(self.id),
                "executed": True,
                "result": result,
                "activation": activation,
                "execution_count": self.execution_count,
            }
        except Exception as e:
            return {
                "type": "tool",
                "neuron_id": str(self.id),
                "executed": False,
                "error": str(e),
                "activation": activation,
            }
    
    def execute(self, inputs: Dict[str, Any]) -> Any:
        """
        Execute the tool function with given inputs.
        
        Executes the tool's code in a sandboxed environment with input validation.
        Tracks execution time and updates statistics.
        
        Args:
            inputs: Input parameters for the tool
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If execution fails
            
        Requirements: 10.1, 10.2, 10.4
        """
        import time
        
        start_time = time.time()
        
        try:
            # Validate inputs against schema
            if self.input_schema:
                validation_errors = self._validate_inputs(inputs)
                if validation_errors:
                    raise ValueError(f"Input validation failed: {validation_errors}")
            
            # Execute the tool code in a sandboxed environment
            result = self._execute_sandboxed(inputs)
            
            # Validate output against schema
            if self.output_schema:
                output_errors = self._validate_output(result)
                if output_errors:
                    raise ValueError(f"Output validation failed: {output_errors}")
            
            # Update execution statistics
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            self._update_execution_stats(execution_time)
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self._update_execution_stats(execution_time, failed=True)
            raise RuntimeError(f"Tool execution failed: {str(e)}") from e
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """
        Validate inputs against the input schema.
        
        Args:
            inputs: Input parameters to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check required fields
        required_fields = self.input_schema.get("required", [])
        for field in required_fields:
            if field not in inputs:
                errors.append(f"Missing required field: {field}")
        
        # Check field types
        properties = self.input_schema.get("properties", {})
        for field, value in inputs.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if expected_type:
                    if not self._check_type(value, expected_type):
                        errors.append(
                            f"Field '{field}' has wrong type: expected {expected_type}, "
                            f"got {type(value).__name__}"
                        )
        
        return errors
    
    def _validate_output(self, output: Any) -> List[str]:
        """
        Validate output against the output schema.
        
        Args:
            output: Output to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # If output schema expects an object, validate it
        if self.output_schema.get("type") == "object" and isinstance(output, dict):
            required_fields = self.output_schema.get("required", [])
            for field in required_fields:
                if field not in output:
                    errors.append(f"Missing required output field: {field}")
            
            properties = self.output_schema.get("properties", {})
            for field, value in output.items():
                if field in properties:
                    expected_type = properties[field].get("type")
                    if expected_type:
                        if not self._check_type(value, expected_type):
                            errors.append(
                                f"Output field '{field}' has wrong type: "
                                f"expected {expected_type}, got {type(value).__name__}"
                            )
        
        return errors
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """
        Check if a value matches the expected JSON Schema type.
        
        Args:
            value: Value to check
            expected_type: Expected type from JSON Schema
            
        Returns:
            True if type matches
        """
        type_mapping = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, skip validation
        
        return isinstance(value, expected_python_type)
    
    def _execute_sandboxed(self, inputs: Dict[str, Any]) -> Any:
        """
        Execute the tool code in a sandboxed environment.
        
        This provides basic sandboxing by restricting the execution environment.
        For production use, consider using more robust sandboxing solutions.
        
        Args:
            inputs: Input parameters for the tool
            
        Returns:
            Execution result
        """
        # Create a restricted execution environment
        # Only allow safe built-ins
        safe_builtins = {
            'abs': abs,
            'all': all,
            'any': any,
            'bool': bool,
            'dict': dict,
            'enumerate': enumerate,
            'float': float,
            'int': int,
            'len': len,
            'list': list,
            'max': max,
            'min': min,
            'range': range,
            'round': round,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'zip': zip,
        }
        
        # Create execution namespace
        namespace = {
            '__builtins__': safe_builtins,
            'inputs': inputs,
            'result': None
        }
        
        # Execute the code
        try:
            exec(self.executable_code, namespace)
            return namespace.get('result')
        except Exception as e:
            raise RuntimeError(f"Sandboxed execution failed: {str(e)}") from e
    
    def _update_execution_stats(self, execution_time_ms: float, failed: bool = False):
        """
        Update execution statistics.
        
        Args:
            execution_time_ms: Execution time in milliseconds
            failed: Whether the execution failed
        """
        if not failed:
            # Update average execution time using running average
            if self.execution_count == 0:
                self.average_execution_time = execution_time_ms
            else:
                # Running average: new_avg = old_avg + (new_value - old_avg) / (count + 1)
                self.average_execution_time = (
                    self.average_execution_time +
                    (execution_time_ms - self.average_execution_time) / (self.execution_count + 1)
                )
        
        # Store execution time in metadata for tracking
        if 'execution_times' not in self.metadata:
            self.metadata['execution_times'] = []
        
        self.metadata['execution_times'].append({
            'time_ms': execution_time_ms,
            'failed': failed,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 100 execution times to avoid memory bloat
        if len(self.metadata['execution_times']) > 100:
            self.metadata['execution_times'] = self.metadata['execution_times'][-100:]
    
    def extract_inputs_from_synapses(
        self,
        graph: 'NeuronGraph',
        activated_neurons: Dict[UUID, float]
    ) -> Dict[str, Any]:
        """
        Extract input parameters from connected neurons via TOOL_INPUT synapses.
        
        Looks at incoming TOOL_INPUT synapses and extracts data from the
        source neurons to build the input parameter dictionary.
        
        Args:
            graph: The neuron graph containing connections
            activated_neurons: Dictionary mapping neuron IDs to activation levels
            
        Returns:
            Dictionary of input parameters extracted from connected neurons
            
        Requirements: 10.3, 10.4
        """
        from neuron_system.core.synapse import SynapseType
        
        inputs = {}
        
        # Get all incoming synapses
        incoming_synapses = graph.get_incoming_synapses(self.id)
        
        # Filter for TOOL_INPUT synapses
        input_synapses = [
            s for s in incoming_synapses
            if s.synapse_type == SynapseType.TOOL_INPUT
        ]
        
        # Extract data from each input synapse
        for synapse in input_synapses:
            source_neuron = graph.get_neuron(synapse.source_neuron_id)
            if source_neuron is None:
                continue
            
            # Get parameter name from synapse metadata
            param_name = synapse.metadata.get('parameter_name')
            if not param_name:
                continue
            
            # Extract value based on source neuron type
            value = self._extract_value_from_neuron(source_neuron, activated_neurons)
            
            if value is not None:
                inputs[param_name] = value
        
        return inputs
    
    def _extract_value_from_neuron(
        self,
        neuron: 'Neuron',
        activated_neurons: Dict[UUID, float]
    ) -> Any:
        """
        Extract a value from a neuron based on its type.
        
        Args:
            neuron: The neuron to extract value from
            activated_neurons: Dictionary of activated neurons
            
        Returns:
            Extracted value or None
        """
        from neuron_system.core.neuron import NeuronType
        
        # For knowledge neurons, extract the source data
        if neuron.neuron_type == NeuronType.KNOWLEDGE:
            if hasattr(neuron, 'source_data'):
                return neuron.source_data
        
        # For tool neurons, use their last execution result if available
        elif neuron.neuron_type == NeuronType.TOOL:
            if 'last_result' in neuron.metadata:
                return neuron.metadata['last_result']
        
        # Default: return activation level
        return activated_neurons.get(neuron.id, 0.0)
    
    def propagate_results_to_outputs(
        self,
        graph: 'NeuronGraph',
        result: Any
    ) -> List[UUID]:
        """
        Propagate execution results to connected neurons via TOOL_OUTPUT synapses.
        
        Sends the tool's execution result to downstream neurons through
        TOOL_OUTPUT synapses, storing the result in their metadata.
        
        Args:
            graph: The neuron graph containing connections
            result: The execution result to propagate
            
        Returns:
            List of neuron IDs that received the result
            
        Requirements: 10.3, 10.4
        """
        from neuron_system.core.synapse import SynapseType
        
        propagated_to = []
        
        # Get all outgoing synapses
        outgoing_synapses = graph.get_outgoing_synapses(self.id)
        
        # Filter for TOOL_OUTPUT synapses
        output_synapses = [
            s for s in outgoing_synapses
            if s.synapse_type == SynapseType.TOOL_OUTPUT
        ]
        
        # Propagate result to each output synapse
        for synapse in output_synapses:
            target_neuron = graph.get_neuron(synapse.target_neuron_id)
            if target_neuron is None:
                continue
            
            # Store result in target neuron's metadata
            target_neuron.metadata['received_tool_result'] = {
                'source_tool_id': str(self.id),
                'result': result,
                'timestamp': datetime.now().isoformat(),
                'synapse_weight': synapse.weight
            }
            
            # Mark synapse as traversed
            synapse.traverse()
            
            propagated_to.append(target_neuron.id)
        
        # Store result in this neuron's metadata for future reference
        self.metadata['last_result'] = result
        self.metadata['last_execution_time'] = datetime.now().isoformat()
        
        return propagated_to
    
    def handle_execution_error(
        self,
        graph: 'NeuronGraph',
        error: Exception
    ) -> Dict[str, Any]:
        """
        Handle tool execution errors and propagate error information.
        
        Stores error information in metadata and optionally propagates
        to connected neurons.
        
        Args:
            graph: The neuron graph containing connections
            error: The exception that occurred
            
        Returns:
            Dictionary with error handling information
            
        Requirements: 10.3, 10.4
        """
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'execution_count': self.execution_count
        }
        
        # Store error in metadata
        if 'execution_errors' not in self.metadata:
            self.metadata['execution_errors'] = []
        
        self.metadata['execution_errors'].append(error_info)
        
        # Keep only last 50 errors
        if len(self.metadata['execution_errors']) > 50:
            self.metadata['execution_errors'] = self.metadata['execution_errors'][-50:]
        
        # Propagate error to output synapses
        from neuron_system.core.synapse import SynapseType
        
        outgoing_synapses = graph.get_outgoing_synapses(self.id)
        output_synapses = [
            s for s in outgoing_synapses
            if s.synapse_type == SynapseType.TOOL_OUTPUT
        ]
        
        for synapse in output_synapses:
            target_neuron = graph.get_neuron(synapse.target_neuron_id)
            if target_neuron:
                target_neuron.metadata['received_tool_error'] = error_info
        
        return error_info
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize tool neuron to dictionary.
        
        Returns:
            Dictionary representation
        """
        base_dict = self._base_to_dict()
        base_dict.update({
            "function_signature": self.function_signature,
            "executable_code": self.executable_code,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "execution_count": self.execution_count,
            "average_execution_time": self.average_execution_time,
            "activation_threshold": self.activation_threshold,
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolNeuron':
        """
        Deserialize tool neuron from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            ToolNeuron instance
        """
        neuron = cls(
            function_signature=data.get("function_signature", ""),
            executable_code=data.get("executable_code", ""),
            input_schema=data.get("input_schema", {}),
            output_schema=data.get("output_schema", {}),
            execution_count=data.get("execution_count", 0),
            average_execution_time=data.get("average_execution_time", 0.0),
            activation_threshold=data.get("activation_threshold", 0.5),
        )
        neuron._base_from_dict(data)
        return neuron


# Register ToolNeuron in the type registry
NeuronTypeRegistry.register("tool", ToolNeuron)
