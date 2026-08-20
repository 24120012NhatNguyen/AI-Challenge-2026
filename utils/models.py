from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator


# Define Pydantic models for request validation
class TextSearchRequest(BaseModel):
    search_space: int
    k: int
    nomic: bool
    clipv2: bool
    textquery: str
    range_filter: int
    filter: bool
    id: Optional[List[int]] = None
    ignore: Optional[bool] = False
    ignore_idxs: Optional[List[int]] = None
    filtervideo: int = 0
    videos: Optional[List[Dict[str, Any]]] = None

    @field_validator("videos", mode="before")
    @classmethod
    def videos_accept_dict_or_list(cls, v: Any) -> Optional[List[Dict[str, Any]]]:
        if v is None:
            return None
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            return None  # dict (vd. {}) từ FE → coi như không có kết quả trước
        return None


class PanelSearchRequest(BaseModel):
    k: int
    search_space: int
    useid: bool
    id: Optional[List[int]] = None
    ignore: Optional[bool] = False
    ignore_idxs: Optional[List[int]] = None
    ocr: str = ""
    asr: str = ""
    dragObject: Optional[List[Dict[str, Any]]] = []
    tags: Optional[List[str]] = []
    amount: Optional[str] = ""


class FeedbackRequest(BaseModel):
    k: int
    videos: Dict[str, Any]
    lst_pos_idxs: List[int]
    lst_neg_idxs: List[int]


class TagRequest(BaseModel):
    text: str


class TranslateRequest(BaseModel):
    textquery: str


# Define Pydantic models for request bodies
class UserRequest(BaseModel):
    user: str


class UsernameRequest(BaseModel):
    username: str


class QuestionNameRequest(BaseModel):
    questionName: str
