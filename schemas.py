from typing import Optional, TypeVar
from pydantic import BaseModel

T = TypeVar('T')

class ContentSchema(BaseModel):
    id: Optional[int] = None
    content_type: Optional[str] = None
    content_url: Optional[str] = None
    content_features: Optional[str] = None

    class Config:
        orm_mode = True
