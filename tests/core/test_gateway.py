"""Test script for gateway service."""

import sys
import os

# Add the workspace to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that gateway imports work correctly."""
    try:
        from services.gateway.main import app
        print("✓ Gateway service imports successfully")
        return True
    except Exception as e:
        print(f"✗ Gateway service import failed: {e}")
        return False

if __name__ == "__main__":
    test_imports()