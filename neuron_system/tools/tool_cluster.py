"""
ToolCluster implementation for grouping multiple tool neurons.
"""

from typing import Any, Dict, List, Set
from uuid import UUID, uuid4
from datetime import datetime
from collections import defaultdict, deque


class ToolCluster:
    """
    Groups multiple Tool Neurons into a named cluster with execution orchestration.
    
    Tool clusters represent complex functionality composed of multiple tools
    that work together through a directed acyclic graph (DAG) of execution.
    
    Requirements: 11.1, 11.5
    """
    
    def __init__(
        self,
        name: str,
        tool_neurons: List[UUID] = None,
        execution_graph: Dict[UUID, List[UUID]] = None,
        input_interface: Dict[str, Any] = None,
        output_interface: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize tool cluster.
        
        Args:
            name: Human-readable name for the cluster
            tool_neurons: List of tool neuron UUIDs in this cluster
            execution_graph: DAG mapping tool UUID to list of dependent tool UUIDs
            input_interface: Schema defining cluster inputs
            output_interface: Schema defining cluster outputs
            metadata: Additional metadata for the cluster
        """
        self.id: UUID = uuid4()
        self.name: str = name
        self.tool_neurons: List[UUID] = tool_neurons or []
        self.execution_graph: Dict[UUID, List[UUID]] = execution_graph or {}
        self.input_interface: Dict[str, Any] = input_interface or {}
        self.output_interface: Dict[str, Any] = output_interface or {}
        self.metadata: Dict[str, Any] = metadata or {}
        self.created_at: datetime = datetime.now()
        self.modified_at: datetime = datetime.now()
        
        # Execution statistics
        self.execution_count: int = 0
        self.average_execution_time: float = 0.0
        self.last_execution_time: datetime = None
    
    def add_tool_neuron(self, tool_neuron_id: UUID):
        """
        Add a tool neuron to the cluster.
        
        Args:
            tool_neuron_id: UUID of the tool neuron to add
        """
        if tool_neuron_id not in self.tool_neurons:
            self.tool_neurons.append(tool_neuron_id)
            self.modified_at = datetime.now()
    
    def remove_tool_neuron(self, tool_neuron_id: UUID) -> bool:
        """
        Remove a tool neuron from the cluster.
        
        Args:
            tool_neuron_id: UUID of the tool neuron to remove
            
        Returns:
            True if removed, False if not found
        """
        if tool_neuron_id in self.tool_neurons:
            self.tool_neurons.remove(tool_neuron_id)
            
            # Remove from execution graph
            if tool_neuron_id in self.execution_graph:
                del self.execution_graph[tool_neuron_id]
            
            # Remove as dependency from other tools
            for deps in self.execution_graph.values():
                if tool_neuron_id in deps:
                    deps.remove(tool_neuron_id)
            
            self.modified_at = datetime.now()
            return True
        return False
    
    def add_dependency(self, tool_id: UUID, depends_on: UUID):
        """
        Add a dependency between two tools in the execution graph.
        
        Args:
            tool_id: UUID of the tool that depends on another
            depends_on: UUID of the tool that must execute first
            
        Raises:
            ValueError: If either tool is not in the cluster
        """
        if tool_id not in self.tool_neurons:
            raise ValueError(f"Tool {tool_id} is not in this cluster")
        if depends_on not in self.tool_neurons:
            raise ValueError(f"Tool {depends_on} is not in this cluster")
        
        if tool_id not in self.execution_graph:
            self.execution_graph[tool_id] = []
        
        if depends_on not in self.execution_graph[tool_id]:
            self.execution_graph[tool_id].append(depends_on)
            self.modified_at = datetime.now()
    
    def remove_dependency(self, tool_id: UUID, depends_on: UUID) -> bool:
        """
        Remove a dependency between two tools.
        
        Args:
            tool_id: UUID of the dependent tool
            depends_on: UUID of the dependency to remove
            
        Returns:
            True if removed, False if not found
        """
        if tool_id in self.execution_graph:
            if depends_on in self.execution_graph[tool_id]:
                self.execution_graph[tool_id].remove(depends_on)
                self.modified_at = datetime.now()
                return True
        return False
    
    def get_dependencies(self, tool_id: UUID) -> List[UUID]:
        """
        Get all dependencies for a tool.
        
        Args:
            tool_id: UUID of the tool
            
        Returns:
            List of tool UUIDs that this tool depends on
        """
        return self.execution_graph.get(tool_id, [])
    
    def get_dependents(self, tool_id: UUID) -> List[UUID]:
        """
        Get all tools that depend on a given tool.
        
        Args:
            tool_id: UUID of the tool
            
        Returns:
            List of tool UUIDs that depend on this tool
        """
        dependents = []
        for tid, deps in self.execution_graph.items():
            if tool_id in deps:
                dependents.append(tid)
        return dependents
    
    def validate_acyclic(self) -> bool:
        """
        Validate that the execution graph is acyclic (DAG).
        
        Uses depth-first search to detect cycles.
        
        Returns:
            True if graph is acyclic, False if cycles detected
            
        Requirements: 11.4
        """
        # Track visited nodes and recursion stack
        visited: Set[UUID] = set()
        rec_stack: Set[UUID] = set()
        
        def has_cycle(node: UUID) -> bool:
            """DFS helper to detect cycles."""
            visited.add(node)
            rec_stack.add(node)
            
            # Check all dependencies
            for dep in self.execution_graph.get(node, []):
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    # Back edge found - cycle detected
                    return True
            
            rec_stack.remove(node)
            return False
        
        # Check all nodes
        for tool_id in self.tool_neurons:
            if tool_id not in visited:
                if has_cycle(tool_id):
                    return False
        
        return True
    
    def get_execution_order(self) -> List[UUID]:
        """
        Calculate execution order using topological sort.
        
        Returns tools in an order where all dependencies are satisfied.
        
        Returns:
            List of tool UUIDs in execution order
            
        Raises:
            ValueError: If graph contains cycles
            
        Requirements: 11.2
        """
        if not self.validate_acyclic():
            raise ValueError("Execution graph contains cycles")
        
        # Calculate in-degree for each node
        in_degree: Dict[UUID, int] = defaultdict(int)
        
        # Initialize all tools with 0 in-degree
        for tool_id in self.tool_neurons:
            in_degree[tool_id] = 0
        
        # Count dependencies (in-degree)
        for tool_id, deps in self.execution_graph.items():
            in_degree[tool_id] = len(deps)
        
        # Queue of tools with no dependencies
        queue = deque([tid for tid in self.tool_neurons if in_degree[tid] == 0])
        execution_order = []
        
        while queue:
            # Process tool with no remaining dependencies
            current = queue.popleft()
            execution_order.append(current)
            
            # Reduce in-degree for dependents
            for dependent in self.get_dependents(current):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # If not all tools processed, there's a cycle (shouldn't happen after validation)
        if len(execution_order) != len(self.tool_neurons):
            raise ValueError("Failed to compute execution order - graph may contain cycles")
        
        return execution_order
    
    def get_input_tools(self) -> List[UUID]:
        """
        Get tools that have no dependencies (entry points).
        
        Returns:
            List of tool UUIDs that are entry points
        """
        input_tools = []
        for tool_id in self.tool_neurons:
            if not self.execution_graph.get(tool_id, []):
                input_tools.append(tool_id)
        return input_tools
    
    def get_output_tools(self) -> List[UUID]:
        """
        Get tools that no other tools depend on (exit points).
        
        Returns:
            List of tool UUIDs that are exit points
        """
        has_dependents = set()
        for deps in self.execution_graph.values():
            has_dependents.update(deps)
        
        output_tools = [tid for tid in self.tool_neurons if tid not in has_dependents]
        return output_tools
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize tool cluster to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": str(self.id),
            "name": self.name,
            "tool_neurons": [str(tid) for tid in self.tool_neurons],
            "execution_graph": {
                str(k): [str(v) for v in vals]
                for k, vals in self.execution_graph.items()
            },
            "input_interface": self.input_interface,
            "output_interface": self.output_interface,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "execution_count": self.execution_count,
            "average_execution_time": self.average_execution_time,
            "last_execution_time": (
                self.last_execution_time.isoformat()
                if self.last_execution_time else None
            ),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolCluster':
        """
        Deserialize tool cluster from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            ToolCluster instance
        """
        cluster = cls(
            name=data["name"],
            tool_neurons=[UUID(tid) for tid in data.get("tool_neurons", [])],
            execution_graph={
                UUID(k): [UUID(v) for v in vals]
                for k, vals in data.get("execution_graph", {}).items()
            },
            input_interface=data.get("input_interface", {}),
            output_interface=data.get("output_interface", {}),
            metadata=data.get("metadata", {}),
        )
        
        cluster.id = UUID(data["id"])
        cluster.created_at = datetime.fromisoformat(data["created_at"])
        cluster.modified_at = datetime.fromisoformat(data["modified_at"])
        cluster.execution_count = data.get("execution_count", 0)
        cluster.average_execution_time = data.get("average_execution_time", 0.0)
        
        if data.get("last_execution_time"):
            cluster.last_execution_time = datetime.fromisoformat(
                data["last_execution_time"]
            )
        
        return cluster
