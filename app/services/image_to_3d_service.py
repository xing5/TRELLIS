import os
import uuid
import asyncio
from typing import Dict, Optional
import aiohttp
from PIL import Image
from io import BytesIO
import imageio
import logging

from app.models.image_to_3d import TaskStatus
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from app.core.config import settings

logger = logging.getLogger(__name__)

class ImageTo3DService:
    def __init__(self):
        self._pipeline = None
        self._tasks: Dict[str, Dict] = {}
        
    async def initialize(self):
        """Initialize the Trellis pipeline"""
        if self._pipeline is None:
            self._pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
            self._pipeline.cuda()

    async def _download_image(self, image_url: str) -> Optional[Image.Image]:
        """Download image from URL"""
        logger.info(f"Downloading image from: {image_url}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    logger.info(f"Response status: {response.status}")
                    if response.status != 200:
                        logger.error(f"Failed with status: {response.status}")
                        return None
                    image_data = await response.read()
                    logger.info(f"Downloaded {len(image_data)} bytes")
                    try:
                        return Image.open(BytesIO(image_data))
                    except Exception as e:
                        logger.error(f"Failed to open image: {str(e)}")
                        return None
        except Exception as e:
            logger.error(f"Failed to download: {str(e)}", exc_info=True)
            return None

    async def create_task(self, image_url: str, segm_mode: str) -> str:
        """Create a new 3D conversion task"""
        task_id = str(uuid.uuid4())
        self._tasks[task_id] = {
            "id": task_id,
            "status": TaskStatus.PROCESSING,
            "image_url": image_url,
            "segm_mode": segm_mode
        }
        
        # Start processing in background
        asyncio.create_task(self._process_task(task_id))
        return task_id

    async def _process_task(self, task_id: str):
        """Process the 3D conversion task"""
        try:
            await self.initialize()
            
            task = self._tasks[task_id]
            image = await self._download_image(task["image_url"])
            
            if image is None:
                raise ValueError("Failed to download image")

            # Run the pipeline
            outputs = self._pipeline.run(image, seed=1)

            # Save preview video
            video = render_utils.render_video(outputs['gaussian'][0])['color']
            preview_path = f"assets/previews/{task_id}.mp4"
            os.makedirs(os.path.dirname(preview_path), exist_ok=True)
            imageio.mimsave(preview_path, video, fps=30)

            # Update task with preview
            self._tasks[task_id].update({
                "status": TaskStatus.PREVIEW,
                "preview": f"{settings.BASE_URL}/assets/previews/{task_id}.mp4"
            })

            # Generate GLB
            glb = postprocessing_utils.to_glb(
                outputs['gaussian'][0],
                outputs['mesh'][0],
                simplify=0.95,
                texture_size=1024,
            )
            glb_path = f"assets/models/{task_id}.glb"
            os.makedirs(os.path.dirname(glb_path), exist_ok=True)
            glb.export(glb_path)

            # Update task with success
            self._tasks[task_id].update({
                "status": TaskStatus.SUCCESS,
                "models": {
                    "glb": f"{settings.BASE_URL}/assets/models/{task_id}.glb"
                }
            })

        except Exception as e:
            self._tasks[task_id].update({
                "status": TaskStatus.FAILED,
                "error": {
                    "code": 10001,
                    "message": str(e)
                }
            })

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get the status of a task"""
        return self._tasks.get(task_id) 