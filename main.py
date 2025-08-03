import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the app directory to Python path
current_dir = Path(__file__).parent
app_dir = current_dir / "app"
sys.path.insert(0, str(app_dir))
sys.path.insert(0, str(current_dir))

logger.info(f"Python path: {sys.path}")
logger.info(f"Current directory: {current_dir}")
logger.info(f"App directory: {app_dir}")

try:
    # Import and run the FastAPI app
    from app.main import app
    import uvicorn

    logger.info("Successfully imported FastAPI app")

    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 7860))
        logger.info(f"Starting server on port {port}")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )

except Exception as e:
    logger.error(f"Error starting application: {e}")
    raise