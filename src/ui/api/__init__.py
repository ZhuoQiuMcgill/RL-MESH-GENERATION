from flask import Blueprint

from .training import training_bp
from .mesh import mesh_bp


def register_blueprints(app):
    app.register_blueprint(training_bp)
    app.register_blueprint(mesh_bp)


__all__ = ["register_blueprints", "training_bp", "mesh_bp"]
