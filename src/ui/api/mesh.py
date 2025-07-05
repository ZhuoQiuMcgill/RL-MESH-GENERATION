from flask import Blueprint, jsonify, request

from src.utils import MeshImporter

mesh_bp = Blueprint("mesh", __name__, url_prefix="/mesh")
importer = MeshImporter()


@mesh_bp.route("/list", methods=["GET"])
def list_meshes():
    subfolder = request.args.get("subfolder", "mesh")
    meshes = importer.list_available_meshes(subfolder)
    return jsonify({"meshes": meshes})


@mesh_bp.route("/info/<name>", methods=["GET"])
def mesh_info(name: str):
    subfolder = request.args.get("subfolder", "mesh")
    info = importer.get_mesh_info(name, subfolder)
    return jsonify(info)
