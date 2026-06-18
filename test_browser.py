"""Test script for browser service."""

import sys
import os

# Add the workspace to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that browser service imports work correctly."""
    try:
        from services.browser.main import app
        print("✓ Browser service imports successfully")
        return True
    except Exception as e:
        print(f"✗ Browser service import failed: {e}")
        return False

if __name__ == "__main__":
    test_imports()