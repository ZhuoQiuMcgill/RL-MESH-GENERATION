import os
import shutil
import sys
import argparse
from pathlib import Path


def clear_training_cache(base_dir: str = ".", clear_models: bool = True,
                         clear_logs: bool = False, clear_results: bool = False):
    """
    Clear training cache to fix dimension mismatch issues.

    Args:
        base_dir: Base directory path
        clear_models: Whether to clear saved models
        clear_logs: Whether to clear training logs
        clear_results: Whether to clear all results
    """
    base_path = Path(base_dir)

    print("🧹 Clearing Training Cache")
    print("=" * 50)

    # Clear models directory
    if clear_models:
        models_dir = base_path / "results" / "models"
        if models_dir.exists():
            print(f"🗑️  Removing models directory: {models_dir}")
            shutil.rmtree(models_dir)
            models_dir.mkdir(parents=True, exist_ok=True)
            print("✅ Models directory cleared")
        else:
            print("📁 Models directory not found")

    # Clear logs directory
    if clear_logs:
        logs_dir = base_path / "results" / "logs"
        if logs_dir.exists():
            print(f"🗑️  Removing logs directory: {logs_dir}")
            shutil.rmtree(logs_dir)
            logs_dir.mkdir(parents=True, exist_ok=True)
            print("✅ Logs directory cleared")
        else:
            print("📁 Logs directory not found")

    # Clear all results
    if clear_results:
        results_dir = base_path / "results"
        if results_dir.exists():
            print(f"🗑️  Removing results directory: {results_dir}")
            shutil.rmtree(results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)

            # Recreate subdirectories
            (results_dir / "models").mkdir(exist_ok=True)
            (results_dir / "logs").mkdir(exist_ok=True)
            (results_dir / "figures").mkdir(exist_ok=True)
            print("✅ Results directory cleared and recreated")
        else:
            print("📁 Results directory not found")

    # Clear any Python cache
    pycache_dirs = list(base_path.rglob("__pycache__"))
    if pycache_dirs:
        print(f"🗑️  Removing {len(pycache_dirs)} __pycache__ directories")
        for cache_dir in pycache_dirs:
            shutil.rmtree(cache_dir)
        print("✅ Python cache cleared")

    print("\n✨ Cache clearing completed!")
    print("\n📝 Next steps:")
    print("1. Run training script again: python scripts/enhanced_train_script.py")
    print("2. The agent will reinitialize with correct dimensions")


def verify_configuration():
    """Verify that configuration is consistent."""
    print("\n🔍 Verifying Configuration")
    print("=" * 50)

    try:
        import yaml
        config_path = "configs/default_config.yaml"

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            neighbor_num = config['environment'].get('neighbor_num', 6)
            radius_num = config['environment'].get('radius_num', 3)
            expected_state_dim = 2 * (neighbor_num + radius_num)

            print(f"📊 Configuration:")
            print(f"   neighbor_num: {neighbor_num}")
            print(f"   radius_num: {radius_num}")
            print(f"   Expected state dimension: {expected_state_dim}")
            print("✅ Configuration loaded successfully")

            return expected_state_dim
        else:
            print(f"❌ Configuration file not found: {config_path}")
            return None

    except ImportError:
        print("❌ PyYAML not installed. Please install: pip install pyyaml")
        return None
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return None


def check_environment_dimensions():
    """Check if environment can be initialized with current config."""
    print("\n🧪 Testing Environment Initialization")
    print("=" * 50)

    try:
        # Try to import and initialize environment
        sys.path.insert(0, os.getcwd())

        import yaml

        # Check if we can import the environment module
        try:
            from rl_mesher.environment import MeshEnv
        except ImportError:
            from rl_mesher import MeshEnv

        # Load config
        with open("configs/default_config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        # Try to create environment
        env = MeshEnv(config)
        obs_space_shape = env.observation_space.shape

        print(f"✅ Environment initialized successfully")
        print(f"   Observation space shape: {obs_space_shape}")
        print(f"   neighbor_num: {env.neighbor_num}")
        print(f"   radius_num: {env.radius_num}")

        # Try to get an observation
        obs, _ = env.reset()
        print(f"   Reset observation shape: {obs.shape}")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're running from the project root directory")
        return False
    except Exception as e:
        print(f"❌ Environment initialization failed: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fix RL-Mesh-Generation training issues")
    parser.add_argument("--clear-models", action="store_true",
                        help="Clear saved models (default)")
    parser.add_argument("--clear-logs", action="store_true",
                        help="Also clear training logs")
    parser.add_argument("--clear-all", action="store_true",
                        help="Clear all results (models, logs, figures)")
    parser.add_argument("--no-clear", action="store_true",
                        help="Only verify configuration, don't clear anything")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify configuration and environment")

    args = parser.parse_args()

    print("🔧 RL-Mesh-Generation Training Fix")
    print("=" * 50)
    print("This script fixes tensor dimension mismatch errors")
    print("by clearing cached models and verifying configuration.")
    print()

    # Verify configuration first
    expected_dim = verify_configuration()

    # Test environment
    env_ok = check_environment_dimensions()

    if args.verify_only:
        print("\n📋 Verification Summary:")
        print(f"   Configuration: {'✅ OK' if expected_dim else '❌ Error'}")
        print(f"   Environment: {'✅ OK' if env_ok else '❌ Error'}")
        return

    if not args.no_clear:
        # Clear cache based on arguments
        if args.clear_all:
            clear_training_cache(clear_models=True, clear_logs=True, clear_results=True)
        elif args.clear_logs:
            clear_training_cache(clear_models=True, clear_logs=True, clear_results=False)
        else:
            # Default: clear models only
            clear_training_cache(clear_models=True, clear_logs=False, clear_results=False)

    print("\n🚀 Ready to restart training!")
    print("Run: python scripts/enhanced_train_script.py")


if __name__ == "__main__":
    main()