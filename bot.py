
#!/usr/bin/env python3
"""
Simple Flask web server that returns success message
"""

from flask import Flask
import threading
import logging

app = Flask(__name__)

# Set up logging for the web server
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WebServer")

@app.route('/')
def home():
    return {
        "status": "success",
        "message": "Web server running successfully!",
        "port": 5000
    }

@app.route('/health')
def health():
    return {
        "status": "healthy",
        "message": "Server is running successfully"
    }

@app.route('/status')
def status():
    return {
        "status": "running",
        "message": "Simple web port return running successfully",
        "service": "Flask Web Server"
    }

def run_web_server():
    """Run the Flask web server"""
    logger.info("🌐 Starting Flask web server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

if __name__ == "__main__":
    run_web_server()
