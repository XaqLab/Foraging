"""
Auto-reload setup for live editing of external packages.

This script provides utilities for automatically reloading modules
when you make changes to external packages.
"""

import importlib
import sys

from IPython import get_ipython


def setup_auto_reload(mode: int = 2):
    """
    Setup automatic reloading for Jupyter notebooks.
    This enables automatic reloading of modules when files change.

    Args:
        mode: Autoreload mode (default: 2)
            - 0: Disable automatic reloading
            - 1: Reload only modules imported with %aimport
            - 2: Reload all modules automatically
    """
    ipython = get_ipython()
    if ipython is not None:
        # Check if autoreload is already loaded using a more robust method
        try:
            # Try to access the autoreload extension
            autoreload_loaded = hasattr(ipython, "extension_manager") and hasattr(
                ipython.extension_manager, "extensions"
            )
            if autoreload_loaded:
                autoreload_loaded = "autoreload" in ipython.extension_manager.extensions
            else:
                # Fallback: check if autoreload magic is available
                autoreload_loaded = (
                    "autoreload" in ipython.magics_manager.magics["line"]
                )
        except (AttributeError, KeyError):
            autoreload_loaded = False

        if not autoreload_loaded:
            # Enable automatic reloading
            ipython.run_line_magic("load_ext", "autoreload")
            print("✅ Auto-reload extension loaded!")
        else:
            print("✅ Auto-reload extension already loaded!")

        # Set autoreload mode (this can be run multiple times safely)
        ipython.run_line_magic(
            "autoreload", str(mode)
        )  # Reload all modules automatically
        print(
            f"✅ Auto-reload w/ {mode} mode enabled! Changes to all packages will be automatically loaded."
        )
    else:
        print("⚠️  Not in Jupyter environment. Auto-reload not available.")


def add_external_package(path: str, package_name: str = None, **kwargs):
    """
    Add external package to Python path and setup auto-reload.

    Args:
        path: Path to the external package
        package_name: Name of the package (optional, for reloading)
        **kwargs: Additional arguments for the `setup_auto_reload` function
    """
    # Add to Python path
    if path not in sys.path:
        sys.path.insert(0, path)
        print(f"✅ Added {path} to Python path")

    # Setup auto-reload
    setup_auto_reload(**kwargs)

    # If package name provided, reload it
    if package_name:
        try:
            importlib.import_module(package_name)
            print(f"✅ Loaded {package_name}")
        except ImportError as e:
            print(f"⚠️  Could not load {package_name}: {e}")


def reload_package(package_name: str, **kwargs):
    """
    Manually reload a package.

    Args:
        package_name: Name of the package to reload
        **kwargs: Additional arguments for the `setup_auto_reload` function
    """
    try:
        # Reload the package
        module = importlib.import_module(package_name)
        importlib.reload(module)
        print(f"✅ Reloaded {package_name}")
        # Setup auto-reload
        setup_auto_reload(**kwargs)
    except ImportError as e:
        print(f"❌ Could not reload {package_name}: {e}")


# Quick setup for hex-arena
def setup_hexarena(path: str = "D:/Documents/hex-arena", **kwargs):
    """
    Quick setup for hex-arena with auto-reload.
    """
    add_external_package(path, "hexarena", **kwargs)
