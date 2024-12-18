from fastapi import APIRouter, HTTPException
from app.models.image_to_3d import (
    ImageTo3DRequest,
    ImageTo3DResponse,
    TaskStatusResponse
)
from app.services.image_to_3d_service import ImageTo3DService

router = APIRouter()
service = ImageTo3DService()

@router.post("/image-to-3d", response_model=ImageTo3DResponse)
async def create_3d_task(request: ImageTo3DRequest):
    task_id = await service.create_task(
        image_url=str(request.image_url),
        segm_mode=request.segm_mode
    )
    return ImageTo3DResponse(id=task_id)

@router.get("/image-to-3d/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    task = service.get_task_status(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatusResponse(**task) 