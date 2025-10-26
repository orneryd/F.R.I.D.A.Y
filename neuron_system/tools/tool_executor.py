"""
Tool execution orchestration for clusters.
"""

from typing import Any, Dict, List
from uuid import UUID
import time
from datetime import datetime

from neuron_system.tools.tool_cluster import ToolCluster
from neuron_system.core.graph import NeuronGraph
from neuron_system.core.neuron import NeuronType


class ToolClusterExecutor:
    """
    Orchestrates execution of tool clusters.
    
    Handles execution order, data flow, and result aggregation for
    tool clusters with complex dependencies.
    
    Requirements: 11.2, 11.3, 11.4
    """
    
    def __init__(self, graph: NeuronGraph):
        """
        Initialize cluster executor.
        
        Args:
            graph: The neuron graph containing tool neurons
        """
        self.graph = graph
    
    def execute_cluster(
        self,
        cluster: ToolCluster,
        inputs: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute all tools in a cluster according to the execution graph.
        
        Validates the graph is acyclic, computes execution order,
        executes tools with proper data flow, and aggregates results.
        
        Args:
            cluster: The tool cluster to execute
            inputs: Input parameters for the cluster
            context: Additional execution context
            
        Returns:
            Dictionary with execution results and metadata
            
        Raises:
            ValueError: If execution graph is invalid
            RuntimeError: If execution fails
            
        Requirements: 11.2, 11.3, 11.4
        """
        start_time = time.time()
        context = context or {}
        
        # Validate graph is acyclic
        if not cluster.validate_acyclic():
            raise ValueError(
                f"Cluster '{cluster.name}' has cyclic dependencies"
            )
        
        # Get execution order
        try:
            execution_order = cluster.get_execution_order()
        except ValueError as e:
            raise ValueError(
                f"Failed to compute execution order for cluster '{cluster.name}': {e}"
            )
        
        # Track execution results for each tool
        tool_results: Dict[UUID, Any] = {}
        tool_errors: Dict[UUID, str] = {}
        execution_log: List[Dict[str, Any]] = []
        
        # Execute tools in order
        for tool_id in execution_order:
            tool_neuron = self.graph.get_neuron(tool_id)
            
            if tool_neuron is None:
                error_msg = f"Tool neuron {tool_id} not found in graph"
                tool_errors[tool_id] = error_msg
                execution_log.append({
                    "tool_id": str(tool_id),
                    "status": "error",
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat()
                })
                continue
            
            if tool_neuron.neuron_type != NeuronType.TOOL:
                error_msg = f"Neuron {tool_id} is not a tool neuron"
                tool_errors[tool_id] = error_msg
                execution_log.append({
                    "tool_id": str(tool_id),
                    "status": "error",
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat()
                })
                continue
            
            # Prepare inputs for this tool
            tool_inputs = self._prepare_tool_inputs(
                tool_id,
                cluster,
                inputs,
                tool_results
            )
            
            # Execute the tool
            try:
                tool_start = time.time()
                result = tool_neuron.execute(tool_inputs)
                tool_time = (time.time() - tool_start) * 1000  # ms
                
                tool_results[tool_id] = result
                execution_log.append({
                    "tool_id": str(tool_id),
                    "status": "success",
                    "execution_time_ms": tool_time,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                error_msg = str(e)
                tool_errors[tool_id] = error_msg
                execution_log.append({
                    "tool_id": str(tool_id),
                    "status": "error",
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Decide whether to continue or abort
                if self._should_abort_on_error(cluster, tool_id):
                    raise RuntimeError(
                        f"Cluster execution aborted due to error in tool {tool_id}: {error_msg}"
                    )
        
        # Aggregate final results from output tools
        final_results = self._aggregate_output_results(
            cluster,
            tool_results,
            tool_errors
        )
        
        # Update cluster statistics
        execution_time = (time.time() - start_time) * 1000  # ms
        self._update_cluster_stats(cluster, execution_time)
        
        return {
            "cluster_id": str(cluster.id),
            "cluster_name": cluster.name,
            "status": "success" if not tool_errors else "partial_success",
            "results": final_results,
            "tool_results": {str(k): v for k, v in tool_results.items()},
            "errors": {str(k): v for k, v in tool_errors.items()},
            "execution_log": execution_log,
            "execution_time_ms": execution_time,
            "tools_executed": len(tool_results),
            "tools_failed": len(tool_errors),
            "timestamp": datetime.now().isoformat()
        }
    
    def _prepare_tool_inputs(
        self,
        tool_id: UUID,
        cluster: ToolCluster,
        cluster_inputs: Dict[str, Any],
        tool_results: Dict[UUID, Any]
    ) -> Dict[str, Any]:
        """
        Prepare inputs for a tool based on dependencies.
        
        Combines cluster-level inputs with results from dependency tools.
        
        Args:
            tool_id: UUID of the tool to prepare inputs for
            cluster: The tool cluster
            cluster_inputs: Input parameters for the cluster
            tool_results: Results from previously executed tools
            
        Returns:
            Dictionary of inputs for the tool
            
        Requirements: 11.3
        """
        tool_inputs = {}
        
        # Get dependencies for this tool
        dependencies = cluster.get_dependencies(tool_id)
        
        # If no dependencies, use cluster inputs
        if not dependencies:
            tool_inputs.update(cluster_inputs)
        else:
            # Collect results from dependency tools
            for dep_id in dependencies:
                if dep_id in tool_results:
                    dep_result = tool_results[dep_id]
                    
                    # If result is a dict, merge it
                    if isinstance(dep_result, dict):
                        tool_inputs.update(dep_result)
                    else:
                        # Store result with dependency ID as key
                        tool_inputs[f"dep_{dep_id}"] = dep_result
            
            # Also include original cluster inputs
            tool_inputs.update(cluster_inputs)
        
        return tool_inputs
    
    def _aggregate_output_results(
        self,
        cluster: ToolCluster,
        tool_results: Dict[UUID, Any],
        tool_errors: Dict[UUID, str]
    ) -> Dict[str, Any]:
        """
        Aggregate results from output tools.
        
        Collects results from tools that have no dependents (exit points).
        
        Args:
            cluster: The tool cluster
            tool_results: Results from all executed tools
            tool_errors: Errors from failed tools
            
        Returns:
            Aggregated results dictionary
            
        Requirements: 11.2
        """
        output_tools = cluster.get_output_tools()
        aggregated = {}
        
        for tool_id in output_tools:
            if tool_id in tool_results:
                result = tool_results[tool_id]
                
                # If result is a dict, merge it
                if isinstance(result, dict):
                    aggregated.update(result)
                else:
                    # Store with tool ID as key
                    aggregated[f"output_{tool_id}"] = result
            elif tool_id in tool_errors:
                # Include error information
                aggregated[f"error_{tool_id}"] = tool_errors[tool_id]
        
        return aggregated
    
    def _should_abort_on_error(
        self,
        cluster: ToolCluster,
        failed_tool_id: UUID
    ) -> bool:
        """
        Determine if cluster execution should abort on tool error.
        
        Args:
            cluster: The tool cluster
            failed_tool_id: UUID of the tool that failed
            
        Returns:
            True if execution should abort, False to continue
        """
        # Check cluster metadata for error handling policy
        error_policy = cluster.metadata.get("error_policy", "continue")
        
        if error_policy == "abort_on_any":
            return True
        elif error_policy == "abort_on_critical":
            # Check if failed tool is marked as critical
            critical_tools = cluster.metadata.get("critical_tools", [])
            return failed_tool_id in critical_tools
        else:  # "continue"
            return False
    
    def _update_cluster_stats(
        self,
        cluster: ToolCluster,
        execution_time_ms: float
    ):
        """
        Update cluster execution statistics.
        
        Args:
            cluster: The tool cluster
            execution_time_ms: Execution time in milliseconds
        """
        cluster.execution_count += 1
        cluster.last_execution_time = datetime.now()
        
        # Update average execution time
        if cluster.execution_count == 1:
            cluster.average_execution_time = execution_time_ms
        else:
            cluster.average_execution_time = (
                cluster.average_execution_time +
                (execution_time_ms - cluster.average_execution_time) / cluster.execution_count
            )
        
        cluster.modified_at = datetime.now()
    
    def validate_cluster(self, cluster: ToolCluster) -> Dict[str, Any]:
        """
        Validate a tool cluster before execution.
        
        Checks that all tool neurons exist, graph is acyclic, and
        dependencies are valid.
        
        Args:
            cluster: The tool cluster to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_errors = []
        validation_warnings = []
        
        # Check all tool neurons exist
        for tool_id in cluster.tool_neurons:
            neuron = self.graph.get_neuron(tool_id)
            if neuron is None:
                validation_errors.append(
                    f"Tool neuron {tool_id} not found in graph"
                )
            elif neuron.neuron_type != NeuronType.TOOL:
                validation_errors.append(
                    f"Neuron {tool_id} is not a tool neuron"
                )
        
        # Check graph is acyclic
        if not cluster.validate_acyclic():
            validation_errors.append(
                "Execution graph contains cycles"
            )
        
        # Check for orphaned tools (no inputs and no outputs)
        input_tools = cluster.get_input_tools()
        output_tools = cluster.get_output_tools()
        
        if not input_tools:
            validation_warnings.append(
                "No input tools found - cluster may not receive inputs"
            )
        
        if not output_tools:
            validation_warnings.append(
                "No output tools found - cluster may not produce outputs"
            )
        
        # Check for unreachable tools
        try:
            execution_order = cluster.get_execution_order()
            if len(execution_order) < len(cluster.tool_neurons):
                validation_warnings.append(
                    f"Only {len(execution_order)} of {len(cluster.tool_neurons)} "
                    "tools are reachable in execution graph"
                )
        except ValueError as e:
            validation_errors.append(f"Failed to compute execution order: {e}")
        
        return {
            "valid": len(validation_errors) == 0,
            "errors": validation_errors,
            "warnings": validation_warnings,
            "tool_count": len(cluster.tool_neurons),
            "input_tools": len(input_tools),
            "output_tools": len(output_tools)
        }
