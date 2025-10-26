"""
Startup script for the 3D Synaptic Neuron System API
"""
import uvicorn
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("Starting 3D Synaptic Neuron System API...")
    print("API Documentation will be available at: http://localhost:8000/docs")
    print("Alternative docs at: http://localhost:8000/redoc")
    print("Press CTRL+C to stop the server")
    print("-" * 60)
    
    uvicorn.run(
        "neuron_system.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )
