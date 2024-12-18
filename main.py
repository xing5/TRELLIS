import os
import logging
os.environ['SPCONV_ALGO'] = 'native'

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.v1 import image_to_3d

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Image to 3D API")

# Mount static files for serving assets
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Add this route before the API routes
@app.get("/")
async def read_root():
    return FileResponse("assets/index.html")

# Include API routes
app.include_router(image_to_3d.router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7070, log_level="info") 