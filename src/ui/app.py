import logging
import traceback
from flask import Flask, jsonify
from flask_cors import CORS

from src.ui.api import register_blueprints


def create_app() -> Flask:
    """
    创建Flask应用实例

    Returns:
        Flask: 配置好的Flask应用实例
    """
    app = Flask(__name__)

    # 配置日志记录
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # 输出到控制台
            logging.FileHandler('app.log', encoding='utf-8')  # 输出到文件
        ]
    )

    # 设置Flask应用的日志级别
    app.logger.setLevel(logging.DEBUG)

    # 配置CORS，允许前端跨域访问 - 修复：添加checkpoint和quality路径
    CORS(app, resources={
        r"/training/*": {"origins": "*"},
        r"/mesh/*": {"origins": "*"},
        r"/checkpoint/*": {"origins": "*"},
        r"/quality/*": {"origins": "*"}
    })

    # 注册API蓝图
    register_blueprints(app)

    # 全局错误处理器
    @app.errorhandler(500)
    def handle_internal_error(error):
        """处理500内部服务器错误"""
        # 记录详细的错误信息到日志
        app.logger.error(f"Internal Server Error: {error}")
        app.logger.error(f"Traceback: {traceback.format_exc()}")

        # 打印到控制台（确保开发时能看到）
        print(f"=== 500 Internal Server Error ===")
        print(f"Error: {error}")
        print(f"Traceback:")
        traceback.print_exc()
        print(f"=== End Error ===")

        return jsonify({
            "error": "Internal server error",
            "message": str(error),
            "success": False
        }), 500

    @app.errorhandler(Exception)
    def handle_exception(error):
        """处理所有未捕获的异常"""
        # 记录详细的错误信息到日志
        app.logger.error(f"Unhandled Exception: {error}")
        app.logger.error(f"Traceback: {traceback.format_exc()}")

        # 打印到控制台（确保开发时能看到）
        print(f"=== Unhandled Exception ===")
        print(f"Exception: {error}")
        print(f"Type: {type(error).__name__}")
        print(f"Traceback:")
        traceback.print_exc()
        print(f"=== End Exception ===")

        return jsonify({
            "error": "Internal server error",
            "message": str(error),
            "type": type(error).__name__,
            "success": False
        }), 500

    return app


app = create_app()

if __name__ == "__main__":
    # 开发模式下启用调试
    app.run(host="0.0.0.0", port=5000, debug=True)