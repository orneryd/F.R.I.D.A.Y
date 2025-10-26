"""
SDK-specific exceptions for the Neuron System client
"""


class NeuronSystemError(Exception):
    """Base exception for all Neuron System SDK errors"""
    pass


class ConnectionError(NeuronSystemError):
    """Raised when connection to the API fails"""
    pass


class AuthenticationError(NeuronSystemError):
    """Raised when authentication fails"""
    pass


class NotFoundError(NeuronSystemError):
    """Raised when a requested resource is not found"""
    pass


class ValidationError(NeuronSystemError):
    """Raised when request validation fails"""
    pass


class ServerError(NeuronSystemError):
    """Raised when the server returns an error"""
    pass


class TimeoutError(NeuronSystemError):
    """Raised when a request times out"""
    pass
