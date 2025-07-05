from flask import Flask
from flask_cors import CORS

from src.ui.api import register_blueprints


def create_app() -> Flask:
    """
    创建Flask应用实例

    Returns:
        Flask: 配置好的Flask应用实例
    """
    app = Flask(__name__)

    # 配置CORS，允许前端跨域访问
    CORS(app, resources={
        r"/training/*": {"origins": "*"},
        r"/mesh/*": {"origins": "*"}
    })

    # 注册API蓝图
    register_blueprints(app)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
