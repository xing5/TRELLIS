import os
import uuid
import logging
import requests
from PIL import Image
from io import BytesIO
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import imageio
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrellisWorker:
    def __init__(self, api_base: str = None, api_key: str = None):
        # Use provided URL, environment variable, or default
        self.api_base = api_base or os.getenv('API_BASE_URL', 'http://localhost:3000/api/v1')
        self.api_key = api_key or os.getenv('API_KEY')
        
        self.worker_id = str(uuid.uuid4())
        self.pipeline = None
        # Only set headers if API key is provided
        self.headers = {'Authorization': f'Bearer {self.api_key}'} if self.api_key else {}
        logger.info(f"Worker initialized with API base URL: {self.api_base}")

    def initialize(self):
        """Initialize the pipeline"""
        if self.pipeline is None:
            self.pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
            self.pipeline.cuda()

    def download_image(self, image_url: str) -> Image.Image:
        """Download image from URL"""
        response = requests.get(image_url)
        if response.status_code != 200:
            raise ValueError(f"Failed to download image: {response.status_code}")
        return Image.open(BytesIO(response.content))

    def claim_task(self) -> dict:
        """Try to claim a task from the API"""
        response = requests.post(
            f"{self.api_base}/tasks/claim",
            json={
                "worker_id": self.worker_id,
                "task_type": "image-to-3d"
            },
            headers=self.headers
        )
        data = response.json()
        if data.get("id"):
            logger.info(f"Claimed task: {data}")
        return data

    def update_task(self, task_id: str, status: str, **kwargs):
        """Update task status"""
        response = requests.post(
            f"{self.api_base}/tasks/{task_id}",
            json={
                "worker_id": self.worker_id,
                "status": status,
                **kwargs
            },
            headers=self.headers
        )
        return response.json()

    def upload_file(self, file_path: str, task_id: str) -> str:
        """Upload a file to the asset management API"""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'taskId': task_id}
            response = requests.post(
                f"{self.api_base}/tasks/{task_id}/assets",
                files=files,
                data=data,
                headers=self.headers
            )
            if response.status_code != 200:
                raise ValueError(f"Failed to upload file: {response.status_code}")
            return response.json()["url"]

    def process_task(self, task: dict):
        """Process a single task"""
        task_id = task["id"]
        try:
            # Get image URL from task input
            image_url = task.get("input", {}).get("image_url")
            if not image_url:
                raise ValueError("No image URL provided in task input")
                
            # Download and process image
            image = self.download_image(image_url)
            
            # Run the pipeline
            outputs = self.pipeline.run(image, seed=1)

            # Save and upload preview video
            preview_path = f"assets/{task_id}-preview.mp4"
            os.makedirs(os.path.dirname(preview_path), exist_ok=True)
            video = render_utils.render_video(outputs['gaussian'][0])['color']
            imageio.mimsave(preview_path, video, fps=30)

            # Upload preview and update task
            preview_url = self.upload_file(preview_path, task_id)
            self.update_task(
                task_id,
                status="preview",
                preview_url=preview_url
            )

            # Generate and save GLB
            glb_path = f"assets/{task_id}-model.glb"
            os.makedirs(os.path.dirname(glb_path), exist_ok=True)
            glb = postprocessing_utils.to_glb(
                outputs['gaussian'][0],
                outputs['mesh'][0],
                simplify=0.95,
                texture_size=1024,
            )
            glb.export(glb_path)

            # Upload GLB and update task
            model_url = self.upload_file(glb_path, task_id)
            self.update_task(
                task_id,
                status="success",
                output={
                    "model_url": model_url
                }
            )

            # Clean up local files
            os.remove(preview_path)
            os.remove(glb_path)

        except Exception as e:
            logger.error(f"Task {task_id} failed: {str(e)}")
            self.update_task(
                task_id,
                status="failed",
                error={
                    "code": 10001,
                    "message": str(e)
                }
            )

    def run(self):
        """Main worker loop"""
        self.initialize()
        logger.info(f"Worker {self.worker_id} started")

        while True:
            try:
                # Try to claim a task
                task = self.claim_task()
                if task.get("id"):
                    logger.info(f"Processing task {task['id']}")
                    self.process_task(task)
                else:
                    # No task available, wait before polling again
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error in worker loop: {str(e)}")
                time.sleep(5)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-base', help='Base URL for the API')
    parser.add_argument('--api-key', help='API key for authentication')
    args = parser.parse_args()
    
    worker = TrellisWorker(api_base=args.api_base, api_key=args.api_key)
    worker.run() 