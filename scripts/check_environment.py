#!/usr/bin/env python3
"""
Environment Check Script for RL-MESH-GENERATION

This script verifies that all required dependencies are correctly installed
and provides information about the system configuration.
"""

import sys
import os
import platform
from typing import Dict, List, Tuple
from pathlib import Path

# Add project root to path - CRITICAL for package imports
ROOT_DIR = os.getcwd()
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Also add the parent directory if we're in scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible."""
    version = sys.version_info
    required_major, required_minor = 3, 8

    if version.major == required_major and version.minor >= required_minor:
        return True, f"✅ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"❌ Python {version.major}.{version.minor}.{version.micro} (requires >= {required_major}.{required_minor})"


def check_package_import(package_name: str, optional: bool = False) -> Tuple[bool, str]:
    """Check if a package can be imported."""
    try:
        if package_name == 'torch':
            import torch
            version = torch.__version__
            cuda_available = torch.cuda.is_available()
            device_info = f"CUDA: {'✅' if cuda_available else '❌'}"
            if cuda_available:
                try:
                    device_info += f" (GPU: {torch.cuda.get_device_name()})"
                except:
                    device_info += " (GPU: Available)"
            return True, f"✅ PyTorch {version} ({device_info})"

        elif package_name == 'numpy':
            import numpy as np
            return True, f"✅ NumPy {np.__version__}"

        elif package_name == 'matplotlib':
            import matplotlib
            return True, f"✅ Matplotlib {matplotlib.__version__}"

        elif package_name == 'shapely':
            from shapely import __version__
            return True, f"✅ Shapely {__version__}"

        elif package_name == 'gymnasium':
            import gymnasium
            return True, f"✅ Gymnasium {gymnasium.__version__}"

        elif package_name == 'yaml':
            import yaml
            return True, f"✅ PyYAML (available)"

        elif package_name == 'scipy':
            import scipy
            return True, f"✅ SciPy {scipy.__version__}"

        elif package_name == 'pandas':
            import pandas as pd
            return True, f"✅ Pandas {pd.__version__}"

        elif package_name == 'tqdm':
            import tqdm
            return True, f"✅ tqdm {tqdm.__version__}"

        elif package_name == 'tensorboard':
            import tensorboard
            return True, f"✅ TensorBoard {tensorboard.__version__}"

        elif package_name == 'plotly':
            import plotly
            return True, f"✅ Plotly {plotly.__version__}"

        else:
            __import__(package_name)
            return True, f"✅ {package_name} (available)"

    except ImportError:
        status = "⚠️" if optional else "❌"
        return optional, f"{status} {package_name} (missing{'- optional' if optional else ''})"
    except Exception as e:
        return False, f"❌ {package_name} (error: {str(e)[:50]}...)"


def check_project_structure() -> List[Tuple[bool, str]]:
    """Check if project structure is correct."""
    # Use PROJECT_ROOT as base directory
    base_dir = PROJECT_ROOT if os.path.exists(os.path.join(PROJECT_ROOT, 'rl_mesher')) else ROOT_DIR

    required_paths = [
        'rl_mesher/',
        'rl_mesher/__init__.py',
        'rl_mesher/environment.py',
        'rl_mesher/agent.py',
        'rl_mesher/networks.py',
        'rl_mesher/utils/',
        'rl_mesher/utils/geometry.py',
        'rl_mesher/utils/visualization.py',
        'configs/',
        'configs/default_config.yaml',
        'scripts/',
        'scripts/train.py',
        'scripts/evaluate.py',
        'data/',
        'data/domains/',
    ]

    results = []
    for path in required_paths:
        full_path = os.path.join(base_dir, path)
        if os.path.exists(full_path):
            results.append((True, f"✅ {path}"))
        else:
            results.append((False, f"❌ {path} (missing)"))

    return results


def check_sample_domains() -> List[Tuple[bool, str]]:
    """Check if sample domain files exist."""
    # Use PROJECT_ROOT as base directory
    base_dir = PROJECT_ROOT if os.path.exists(os.path.join(PROJECT_ROOT, 'rl_mesher')) else ROOT_DIR

    domain_files = ['T1.txt', 'T2.txt', 'T3.txt']
    results = []

    for domain_file in domain_files:
        path = os.path.join(base_dir, 'data', 'domains', domain_file)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
                    num_vertices = len(lines)
                results.append((True, f"✅ {domain_file} ({num_vertices} vertices)"))
            except Exception:
                results.append((False, f"❌ {domain_file} (corrupt file)"))
        else:
            results.append((False, f"❌ {domain_file} (missing)"))

    return results


def check_rl_mesher_import() -> Tuple[bool, str]:
    """Check if rl_mesher package can be imported."""
    try:
        # Ensure paths are set correctly
        print(f"Debug: ROOT_DIR = {ROOT_DIR}")
        print(f"Debug: PROJECT_ROOT = {PROJECT_ROOT}")
        print(f"Debug: sys.path includes: {sys.path[:3]}...")

        # Try different import strategies
        import rl_mesher
        info = rl_mesher.get_package_info()
        return True, f"✅ RL-MESH-GENERATION v{info['version']} successfully imported"

    except ImportError as e:
        # Try alternative import method
        try:
            # Try importing directly from the project directory
            sys.path.insert(0, os.path.join(PROJECT_ROOT, 'rl_mesher'))
            import rl_mesher
            info = rl_mesher.get_package_info()
            return True, f"✅ RL-MESH-GENERATION v{info['version']} successfully imported (alternative method)"
        except Exception as e2:
            return False, f"❌ Cannot import rl_mesher: {str(e)}"
    except Exception as e:
        return False, f"❌ Error importing rl_mesher: {str(e)}"


def get_system_info() -> Dict[str, str]:
    """Get system information."""
    return {
        'Platform': platform.platform(),
        'Architecture': platform.architecture()[0],
        'Processor': platform.processor() or 'Unknown',
        'Python Path': sys.executable,
        'Working Directory': os.getcwd(),
        'Script Directory': SCRIPT_DIR,
        'Project Root': PROJECT_ROOT,
    }


def main():
    """Main check function."""
    print("🔍 RL-MESH-GENERATION Environment Check")
    print("=" * 60)

    # System information
    print("\n📊 System Information:")
    system_info = get_system_info()
    for key, value in system_info.items():
        print(f"  {key}: {value}")

    # Python version check
    print("\n🐍 Python Version:")
    python_ok, python_msg = check_python_version()
    print(f"  {python_msg}")

    # Core dependencies
    print("\n📦 Core Dependencies:")
    core_packages = ['torch', 'numpy', 'scipy', 'matplotlib', 'shapely', 'gymnasium', 'yaml', 'pandas']
    core_results = []

    for package in core_packages:
        ok, msg = check_package_import(package)
        core_results.append(ok)
        print(f"  {msg}")

    # Optional dependencies
    print("\n🔧 Optional Dependencies:")
    optional_packages = ['tqdm', 'tensorboard', 'plotly', 'jupyter', 'wandb']
    optional_results = []

    for package in optional_packages:
        ok, msg = check_package_import(package, optional=True)
        optional_results.append(ok)
        print(f"  {msg}")

    # Project structure
    print("\n📁 Project Structure:")
    structure_results = check_project_structure()
    for ok, msg in structure_results:
        print(f"  {msg}")

    # Sample domains
    print("\n🌍 Sample Domains:")
    domain_results = check_sample_domains()
    for ok, msg in domain_results:
        print(f"  {msg}")

    # RL-Mesher import
    print("\n🎯 Package Import:")
    import_ok, import_msg = check_rl_mesher_import()
    print(f"  {import_msg}")

    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)

    core_missing = sum(1 for ok in core_results if not ok)
    structure_missing = sum(1 for ok, _ in structure_results if not ok)
    domain_missing = sum(1 for ok, _ in domain_results if not ok)

    if python_ok and core_missing == 0 and structure_missing == 0 and import_ok:
        print("🎉 Environment is ready for RL-MESH-GENERATION!")
        print("\nNext steps:")
        print("  1. Run demo: python example_usage.py")
        print("  2. Start training: python scripts/train.py")
        print("  3. Check documentation in README.md")
    else:
        print("⚠️  Environment has issues that need to be resolved:")

        if not python_ok:
            print("  - Upgrade Python to version 3.8 or higher")

        if core_missing > 0:
            print(f"  - Install {core_missing} missing core dependencies")
            print("    Run: conda env create -f environment.yml")
            print("    Or: pip install -r requirements.txt")

        if structure_missing > 0:
            print(f"  - {structure_missing} project files are missing")
            print("    Re-download or re-clone the project")

        if domain_missing > 0:
            print(f"  - {domain_missing} sample domain files are missing")
            print("    Check the data/domains/ directory")

        if not import_ok:
            print("  - Fix package import issues")
            print("    Try running: python fix_installation.py")
            print("    Or: pip install -e .")

    # GPU recommendation
    try:
        import torch
        if not torch.cuda.is_available():
            print("\n💡 Recommendation:")
            print("  Consider using GPU acceleration for faster training.")
            print("  Install CUDA-enabled PyTorch if you have a compatible GPU.")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
