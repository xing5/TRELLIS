from enum import Enum
from typing import Optional, Dict
from pydantic import BaseModel, HttpUrl

class TaskStatus(str, Enum):
    PROCESSING = "processing"
    PREVIEW = "preview"
    SUCCESS = "success"
    FAILED = "failed"

class ErrorResponse(BaseModel):
    code: int
    message: str

class ImageTo3DRequest(BaseModel):
    image_url: HttpUrl
    segm_mode: str = "auto"

class ImageTo3DResponse(BaseModel):
    id: str

class TaskStatusResponse(BaseModel):
    id: str
    status: TaskStatus
    preview: Optional[str] = None
    models: Optional[Dict[str, str]] = None
    error: Optional[ErrorResponse] = None 