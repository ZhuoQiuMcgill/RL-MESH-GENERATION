from .training import training_bp
from .mesh import mesh_bp
from .checkpoint import checkpoint_bp
from .training_history import training_history_bp
from .quality import quality_bp
from .geometry import geometry_bp


def register_blueprints(app):
    app.register_blueprint(training_bp)
    app.register_blueprint(mesh_bp)
    app.register_blueprint(checkpoint_bp)
    app.register_blueprint(training_history_bp)
    app.register_blueprint(quality_bp)
    app.register_blueprint(geometry_bp)


__all__ = ["register_blueprints", "training_bp", "mesh_bp", "checkpoint_bp", "training_history_bp", "quality_bp", "geometry_bp"]
