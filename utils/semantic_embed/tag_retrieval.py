import os

# import glob
# import torch
# import numpy as np
# from typing import List
# import torch.nn.functional as F
import faiss

# from transformers import AutoTokenizer, AutoModel
from utils.logger_config import get_logger
from utils.semantic_extract import semantic_extract

logger = get_logger(__name__)


def GET_PROJECT_ROOT():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(current_dir, "requirements.txt")):
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError("Could not locate project root (requirements.txt not found)")
        current_dir = parent_dir
    return current_dir


PROJECT_ROOT = GET_PROJECT_ROOT()


class tag_retrieval(semantic_extract):
    def __init__(
        self,
        model="sentence-transformers/stsb-xlm-r-multilingual",
        context_path=os.path.join(PROJECT_ROOT, "dict/tag/tag_list.txt"),
        context_vector_path=os.path.join(
            PROJECT_ROOT, "dict/bin/tag_bin/tag_embedding.bin"
        ),
        input_datatype="txt",
        output_datatype="bin",
    ):
        # Cờ khả dụng: dữ liệu tag hỏng KHÔNG được làm chết app.
        self.available = False
        self.unavailable_reason = ""
        self.index = None
        self.raw_data = []

        try:
            self._load(model, context_path, context_vector_path,
                       input_datatype, output_datatype)
        except Exception as e:
            logger.exception("Không khởi tạo được tag recommendation")
            self.available = False
            self.unavailable_reason = f"{type(e).__name__}: {e}"

        if self.available:
            logger.info("Tag recommendation sẵn sàng: %d tag", self.index.ntotal)
        else:
            logger.warning(
                "TAG RECOMMENDATION KHÔNG KHẢ DỤNG - %s. "
                "Ô 'Query to get tag recommendations' sẽ không gợi ý; "
                "các tính năng khác không bị ảnh hưởng.",
                self.unavailable_reason,
            )

    def _load(self, model, context_path, context_vector_path,
              input_datatype, output_datatype):
        for folder in ("dict/bin", "dict/bin/tag_bin"):
            path = os.path.join(PROJECT_ROOT, folder)
            if not os.path.exists(path):
                try:
                    os.makedirs(path, exist_ok=True)
                except OSError as e:
                    self.unavailable_reason = f"không tạo được {folder}: {e}"
                    return

        n_tags = len(semantic_extract.generate_raw_data(context_path, input_datatype))
        if n_tags == 0:
            self.unavailable_reason = f"không đọc được tag nào từ {context_path}"
            return

        # tag_embedding.bin có thể được build từ tag_list.txt cũ -> lệch số
        # vector. Chỉ build lại được nếu thư mục ghi được (trên Kaggle dict/
        # trỏ tới /kaggle/input, chỉ đọc) -> nếu không thì tắt gracefully.
        if os.path.exists(context_vector_path):
            try:
                ntotal = faiss.read_index(context_vector_path).ntotal
            except Exception:
                logger.exception("Không đọc được tag_embedding.bin")
                ntotal = None

            if ntotal != n_tags:
                logger.warning(
                    "tag_embedding.bin lệch (%s vector, cần %d tag từ %s). "
                    "Thử build lại.",
                    ntotal, n_tags, context_path,
                )
                try:
                    os.remove(context_vector_path)
                except OSError as e:
                    self.unavailable_reason = (
                        f"tag_embedding.bin lệch ({ntotal} vector != {n_tags} tag) "
                        f"và không xoá được để build lại (thư mục chỉ đọc): {e}"
                    )
                    return

        try:
            super().__init__(
                model,
                context_path,
                context_vector_path,
                input_datatype,
                output_datatype,
            )
            index = faiss.read_index(context_vector_path)
        except OSError as e:
            self.unavailable_reason = f"không build được tag embedding: {e}"
            return

        if index.ntotal != len(self.raw_data):
            self.unavailable_reason = (
                f"tag_embedding.bin vẫn lệch sau khi build lại: "
                f"{index.ntotal} vector != {len(self.raw_data)} tag"
            )
            return

        self.index = index
        self.available = True

    def __call__(
        self,
        query: str,
        k: int = 3,
    ):
        if not self.available:
            logger.warning("Bỏ qua gợi ý tag: %s", self.unavailable_reason)
            return []

        query_embed = self.get_embedding([query]).to("cpu").numpy()
        k = min(k, self.index.ntotal)
        _, index = self.index.search(query_embed, k)
        # faiss trả -1 khi không đủ kết quả; bỏ qua idx ngoài phạm vi
        return [
            self.raw_data[idx]
            for idx in index[0]
            if 0 <= idx < len(self.raw_data)
        ]


if __name__ == "__main__":
    obj = tag_retrieval()
    print("available:", obj.available, obj.unavailable_reason)
    print(obj("xe hơi màu đỏ.", 3))
    # print(len(obj.raw_data))
    # print(obj.index.ntotal)
