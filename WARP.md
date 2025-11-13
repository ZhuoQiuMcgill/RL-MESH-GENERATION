# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project overview
- Backend: Python Flask service exposing RL training, prediction, mesh/geometry, and history APIs under src/ui/api.
- Core engine: Mesh generation pipeline with a command pattern and an RL predictor (Stable-Baselines3 SAC) under src/mesh_generator and src/rl.
- Frontend: React (Vite) app in frontend/ consuming the Flask APIs.
- Configuration: All key parameters and data paths are in config/config.yaml.

Common commands (Windows PowerShell)
Backend (Python)
- Create and activate venv, then install backend deps (from imports):
  - py -m venv .venv
  - .\.venv\Scripts\Activate.ps1
  - python -m pip install --upgrade pip
  - pip install flask flask-cors gymnasium stable-baselines3 torch numpy
  - Optional (for plots/animations): pip install matplotlib manim

- Run the API server from repo root:
  - python -m src.ui.app
  - Server listens on http://localhost:5000

- Quick API calls (development):
  - $base = "http://localhost:5000"
  - List meshes: Invoke-RestMethod "$base/mesh/list"
  - Start training: 
    - $body = @{ mesh_name = "simple_square"; max_timesteps = 100000 } | ConvertTo-Json
    - Invoke-RestMethod "$base/training/start" -Method Post -ContentType "application/json" -Body $body
  - Check training status: Invoke-RestMethod "$base/training/status"
  - Create prediction session (requires a model .zip under data/models):
    - $body = @{ mesh_name = "basic1"; predictor_type = "RL"; predictor_config = @{ model_path = "data/models/<MODEL>.zip"; n=2; g=3; beta=6 }; ref_selector_type = "default" } | ConvertTo-Json
    - $session = Invoke-RestMethod "$base/predict/session/create" -Method Post -ContentType "application/json" -Body $body
  - Step once: Invoke-RestMethod "$base/predict/session/$($session.session_id)/next" -Method Post
  - Session quality: Invoke-RestMethod "$base/predict/session/$($session.session_id)/quality?method=hybrid&gamma=1.0"

Frontend (React + Vite)
- Install, dev, build, lint without changing directories:
  - npm --prefix frontend install
  - $env:VITE_API_URL = "http://localhost:5000"  # Backend base URL for dev
  - npm --prefix frontend run dev                 # http://localhost:5173
  - npm --prefix frontend run build               # outputs to frontend/dist
  - npm --prefix frontend run lint

Tests
- Frontend: No test script is defined in frontend/package.json.
- Backend: No Python test runner configured. A small manual script exists:
  - python src/test/detransformation_test.py

High-level architecture
- API service (Flask)
  - Entry: src/ui/app.py creates the Flask app, enables CORS, and registers blueprints via src/ui/api/__init__.py.
  - Blueprints:
    - /training (src/ui/api/training.py): start/stop/status for RL training sessions (threaded via TrainingManager).
    - /mesh (src/ui/api/mesh.py): list meshes, boundary info and vertices using MeshImporter and geometry primitives.
    - /predict (src/ui/api/predict.py): session-based mesh generation using MeshGenerator and RLPredictor; supports step/undo/process_all and quality.
    - /training/history (src/ui/api/training_history.py): query recorded training episodes and best runs.
    - /quality, /geometry, /action: quality metrics, coordinate normalization, and action testing utilities.

- Core mesh generation engine
  - Geometry: src/geometry provides Boundary, Mesh, and helpers. Mesh exposes get_adjacency_dict for frontend visualization.
  - Command pattern: src/interfaces/command.Command defines reversible actions. Concrete actions come from src/rl/action and are wired via ACTION_COMMAND_MAPPING.
  - Orchestrator: src/mesh_generator/mesh_generator.py maintains state (boundary, mesh, history) and steps via ActionManager + command execution. Step codes: 0=ok, 1=invalid/retried, 2=complete, 3=error.
  - Predictor interface: src/interfaces/predictor.Predictor. The RL implementation is src/mesh_generator/rl_predictor.RLPredictor (wraps SB3 SAC), which builds an observation vector aligned with MeshEnv and returns raw action vectors for ActionManager.decode_action.
  - Reference selection: Boundary provides get_ref_vertex; custom selectors live under src/geometry/reference_point_selectors and can be injected into MeshGenerator.

- RL training pipeline
  - Environment: src/rl/environment.MeshEnv (gymnasium) encodes state/action spaces, reward, and termination. Uses ActionManager to realize mesh actions.
  - Agent: src/rl/agent/sb3_sac_agent.SB3SACAgent wraps stable_baselines3.SAC and reads hyperparameters from config/config.yaml (sb3_sac section).
  - Trainer: src/rl/training/sb3_sac_trainer.SB3SACTrainer configures callbacks (_EpisodeCallback) that collect metrics (actor/critic losses, alpha) and persist per-episode detail via history_manager.
  - Training manager: src/ui/training_manager.TrainingManager runs training in a background thread, exposes status/progress, integrates checkpoint load/apply, and writes to data/history/<training_id>/.

- Configuration and data layout
  - config/config.yaml centralizes:
    - paths.data_root and subfolders (mesh/custom/examples).
    - environment (n, g, alpha, beta, max_steps, action enable/auto_remap, M_angle).
    - sb3_sac hyperparameters (learning_rate, buffer_size, net_arch, etc.).
    - training (max_timesteps, evaluation settings, save policy).
  - Data directories (relative to repo root by default):
    - data/mesh/*.txt input boundaries.
    - data/models/*.zip trained SAC models used by /predict.
    - data/checkpoints/ and data/history/ managed by TrainingManager and checkpoint manager.

- Frontend application (frontend/)
  - Vite React app with a feature-based structure (features/train, features/predict, features/history). Uses axios client configured by VITE_API_URL (frontend/src/shared/api/client.js).
  - Canvas-based visualization via frontend/src/utils/CanvasRenderer.js and a React hook in frontend/src/hooks/useCanvasRenderer.js.

Important notes for Warp
- Always run the backend from the repository root using module syntax (python -m src.ui.app) so src imports resolve.
- The backend CORS is open for the exposed prefixes; frontend dev server can call http://localhost:5000 by setting VITE_API_URL.
- config/config.yaml is the single source of truth for environment, training, and SB3 parameters; prefer changing values there over hardcoding.
- No CLAUDE/Cursor/Copilot rule files were found in this repo at the time of writing.

Key references
- API docs index: data/docs/README.md
- Training Manager deep-dive: data/docs/backend/training-manager.md
- Frontend quickstart: frontend/README.md
