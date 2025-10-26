"""
SDK module for easy integration with the Neuron System

This module provides a Python client for interacting with the 3D Synaptic Neuron System API.

Example:
    >>> from neuron_system.sdk import NeuronSystemClient
    >>> client = NeuronSystemClient(base_url="http://localhost:8000")
    >>> neuron = client.create_neuron(
    ...     neuron_type="knowledge",
    ...     source_data="Python is a programming language"
    ... )
"""

from neuron_system.sdk.client import NeuronSystemClient
from neuron_system.sdk.exceptions import (
    NeuronSystemError,
    ConnectionError,
    AuthenticationError,
    NotFoundError,
    ValidationError,
    ServerError,
    TimeoutError
)

__all__ = [
    "NeuronSystemClient",
    "NeuronSystemError",
    "ConnectionError",
    "AuthenticationError",
    "NotFoundError",
    "ValidationError",
    "ServerError",
    "TimeoutError"
]
