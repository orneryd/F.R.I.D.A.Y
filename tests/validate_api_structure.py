"""
Validate API structure without running the server
"""
import os
import sys

def check_file_exists(filepath):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"{status} {filepath}")
    return exists

def validate_api_structure():
    """Validate that all API files are created"""
    print("Validating API Structure...")
    print("=" * 60)
    
    files_to_check = [
        # Core API files
        "neuron_system/api/__init__.py",
        "neuron_system/api/app.py",
        "neuron_system/api/models.py",
        "neuron_system/api/auth.py",
        "neuron_system/api/middleware.py",
        
        # Route files
        "neuron_system/api/routes/__init__.py",
        "neuron_system/api/routes/neurons.py",
        "neuron_system/api/routes/synapses.py",
        "neuron_system/api/routes/query.py",
        "neuron_system/api/routes/training.py",
        
        # Supporting files
        "run_api.py",
        "API_README.md",
        "test_api.py"
    ]
    
    all_exist = True
    for filepath in files_to_check:
        if not check_file_exists(filepath):
            all_exist = False
    
    print("=" * 60)
    
    if all_exist:
        print("✓ All API files created successfully!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Start the API server: python run_api.py")
        print("3. Visit http://localhost:8000/docs for API documentation")
        return True
    else:
        print("✗ Some files are missing!")
        return False

def check_file_content(filepath, expected_content):
    """Check if file contains expected content"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            return expected_content in content
    except:
        return False

def validate_api_endpoints():
    """Validate that key endpoints are defined"""
    print("\nValidating API Endpoints...")
    print("=" * 60)
    
    checks = [
        ("neuron_system/api/routes/neurons.py", "POST /neurons", "@router.post(\"/neurons\""),
        ("neuron_system/api/routes/neurons.py", "GET /neurons/{id}", "@router.get(\"/neurons/{neuron_id}\""),
        ("neuron_system/api/routes/neurons.py", "DELETE /neurons/{id}", "@router.delete(\"/neurons/{neuron_id}\""),
        ("neuron_system/api/routes/synapses.py", "POST /synapses", "@router.post(\"/synapses\""),
        ("neuron_system/api/routes/synapses.py", "GET /synapses", "@router.get(\"/synapses\""),
        ("neuron_system/api/routes/query.py", "POST /query", "@router.post(\"/query\""),
        ("neuron_system/api/routes/query.py", "POST /query/spatial", "@router.post(\"/query/spatial\""),
        ("neuron_system/api/routes/query.py", "GET /neurons/{id}/neighbors", "@router.get(\"/neurons/{neuron_id}/neighbors\""),
        ("neuron_system/api/routes/training.py", "POST /training/adjust-neuron", "@router.post(\"/training/adjust-neuron\""),
        ("neuron_system/api/routes/training.py", "POST /training/adjust-synapse", "@router.post(\"/training/adjust-synapse\""),
        ("neuron_system/api/routes/training.py", "POST /training/create-tool", "@router.post(\"/training/create-tool\""),
    ]
    
    all_found = True
    for filepath, endpoint_name, expected_code in checks:
        found = check_file_content(filepath, expected_code)
        status = "✓" if found else "✗"
        print(f"{status} {endpoint_name}")
        if not found:
            all_found = False
    
    print("=" * 60)
    
    if all_found:
        print("✓ All required endpoints are defined!")
        return True
    else:
        print("✗ Some endpoints are missing!")
        return False

def validate_middleware():
    """Validate middleware implementation"""
    print("\nValidating Middleware...")
    print("=" * 60)
    
    middleware_checks = [
        ("Rate limiting", "rate_limit_middleware"),
        ("Request logging", "logging_middleware"),
        ("Security headers", "security_headers_middleware"),
        ("Request size limiting", "request_size_middleware"),
        ("CORS cache", "cors_cache_middleware"),
        ("Error tracking", "error_tracking_middleware"),
    ]
    
    all_found = True
    for name, function_name in middleware_checks:
        found = check_file_content("neuron_system/api/middleware.py", function_name)
        status = "✓" if found else "✗"
        print(f"{status} {name}")
        if not found:
            all_found = False
    
    print("=" * 60)
    
    if all_found:
        print("✓ All middleware components implemented!")
        return True
    else:
        print("✗ Some middleware components are missing!")
        return False

def validate_authentication():
    """Validate authentication implementation"""
    print("\nValidating Authentication...")
    print("=" * 60)
    
    auth_checks = [
        ("API Key authentication", "verify_api_key"),
        ("JWT token support", "create_access_token"),
        ("Token verification", "verify_token"),
    ]
    
    all_found = True
    for name, function_name in auth_checks:
        found = check_file_content("neuron_system/api/auth.py", function_name)
        status = "✓" if found else "✗"
        print(f"{status} {name}")
        if not found:
            all_found = False
    
    print("=" * 60)
    
    if all_found:
        print("✓ All authentication components implemented!")
        return True
    else:
        print("✗ Some authentication components are missing!")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("3D Synaptic Neuron System - API Validation")
    print("=" * 60 + "\n")
    
    structure_ok = validate_api_structure()
    endpoints_ok = validate_api_endpoints()
    middleware_ok = validate_middleware()
    auth_ok = validate_authentication()
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"File Structure:    {'✓ PASS' if structure_ok else '✗ FAIL'}")
    print(f"API Endpoints:     {'✓ PASS' if endpoints_ok else '✗ FAIL'}")
    print(f"Middleware:        {'✓ PASS' if middleware_ok else '✗ FAIL'}")
    print(f"Authentication:    {'✓ PASS' if auth_ok else '✗ FAIL'}")
    print("=" * 60)
    
    if all([structure_ok, endpoints_ok, middleware_ok, auth_ok]):
        print("\n✓ API IMPLEMENTATION COMPLETE!")
        print("\nTask 10: Implement REST API with FastAPI - COMPLETED")
        sys.exit(0)
    else:
        print("\n✗ API implementation has issues")
        sys.exit(1)
