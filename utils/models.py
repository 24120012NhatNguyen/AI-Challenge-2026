from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, field_validator


def _int_or_default(value: Any, default: int) -> int:
    """Ép giá trị về int, tự lùi về `default` khi ô nhập bị bỏ trống.

    Các ô số trên UI (K, range, search space) gửi chuỗi rỗng khi người dùng xoá
    trắng. Nếu không xử lý, pydantic trả 422 "Input should be a valid integer"
    và chặn nguyên lượt tìm kiếm.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return default
        try:
            return int(float(text))
        except ValueError:
            return default
    if isinstance(value, (int, float)):
        return int(value)
    return default


# Define Pydantic models for request validation
class TextSearchRequest(BaseModel):
    search_space: int = 0
    k: int = 500
    nomic: bool = True
    clipv2: bool = False
    textquery: str = ""
    range_filter: int = 3
    filter: bool = False
    id: Optional[List[int]] = None
    ignore: Optional[bool] = False
    ignore_idxs: Optional[List[int]] = None
    filtervideo: int = 0
    videos: Optional[List[Dict[str, Any]]] = None

    @field_validator("k", mode="before")
    @classmethod
    def _k(cls, v: Any) -> int:
        return _int_or_default(v, 500)

    @field_validator("range_filter", mode="before")
    @classmethod
    def _range_filter(cls, v: Any) -> int:
        return _int_or_default(v, 3)

    @field_validator("search_space", "filtervideo", mode="before")
    @classmethod
    def _zero_default(cls, v: Any) -> int:
        return _int_or_default(v, 0)

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
    k: int = 500
    search_space: int = 0
    useid: bool = True
    id: Optional[List[int]] = None
    ignore: Optional[bool] = False
    ignore_idxs: Optional[List[int]] = None
    ocr: str = ""
    asr: str = ""
    dragObject: Optional[List[Dict[str, Any]]] = []
    tags: Optional[List[str]] = []
    amount: Optional[str] = ""

    @field_validator("k", mode="before")
    @classmethod
    def _k(cls, v: Any) -> int:
        return _int_or_default(v, 500)

    @field_validator("search_space", mode="before")
    @classmethod
    def _search_space(cls, v: Any) -> int:
        return _int_or_default(v, 0)

    @field_validator("ocr", "asr", "amount", mode="before")
    @classmethod
    def _text(cls, v: Any) -> str:
        return "" if v is None else str(v)


class FeedbackRequest(BaseModel):
    k: int = 500
    # FE gửi state `videos` là mảng kết quả (group_result_by_video); giữ Dict để
    # tương thích ngược với các client cũ.
    videos: Union[List[Dict[str, Any]], Dict[str, Any]] = []
    lst_pos_idxs: List[int] = []
    lst_neg_idxs: List[int] = []

    @field_validator("k", mode="before")
    @classmethod
    def _k(cls, v: Any) -> int:
        return _int_or_default(v, 500)


class TagRequest(BaseModel):
    text: str = ""


class TranslateRequest(BaseModel):
    textquery: str = ""


# Define Pydantic models for request bodies
class UserRequest(BaseModel):
    user: str = ""


class UsernameRequest(BaseModel):
    username: str = ""


class QuestionNameRequest(BaseModel):
    questionName: str = ""
