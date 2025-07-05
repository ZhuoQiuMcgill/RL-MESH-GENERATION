from flask import Blueprint, request, jsonify

from src.ui.training_manager import TrainingManager

training_manager = TrainingManager()

training_bp = Blueprint("training", __name__, url_prefix="/training")


@training_bp.route("/start", methods=["POST"])
def start_training():
    data = request.get_json(force=True, silent=True) or {}
    mesh_name = data.get("mesh_name")
    subfolder = data.get("subfolder", "mesh")
    episodes = data.get("max_episodes")
    steps = data.get("max_steps")
    try:
        training_manager.start_training(
            mesh_name=mesh_name,
            subfolder=subfolder,
            max_episodes=episodes,
            max_steps=steps,
        )
        return jsonify({"message": "training_started"})
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 400


@training_bp.route("/stop", methods=["POST"])
def stop_training():
    training_manager.stop_training()
    return jsonify({"message": "stop_requested"})


@training_bp.route("/status", methods=["GET"])
def status():
    return jsonify(training_manager.get_status())
