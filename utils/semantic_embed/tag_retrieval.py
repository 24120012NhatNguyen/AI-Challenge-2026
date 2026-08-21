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
        if not os.path.exists(os.path.join(PROJECT_ROOT, "dict/bin")):
            os.mkdir(os.path.join(PROJECT_ROOT, "dict/bin"))

        if not os.path.exists(os.path.join(PROJECT_ROOT, "dict/bin/tag_bin")):
            os.makedirs(os.path.join(PROJECT_ROOT, "dict/bin/tag_bin"), exist_ok=True)

        # Cùng loại lỗi với ma trận ASR: tag_embedding.bin có thể được build từ
        # một tag_list.txt cũ (nhiều tag hơn) -> self.raw_data[idx] văng
        # IndexError và /getrec trả 500. Kiểm tra và build lại nếu lệch.
        self._invalidate_stale_index(context_path, context_vector_path, input_datatype)

        super().__init__(
            model,
            context_path,
            context_vector_path,
            input_datatype,
            output_datatype,
        )
        self.index = faiss.read_index(context_vector_path)

        if self.index.ntotal != len(self.raw_data):
            raise RuntimeError(
                f"tag_embedding.bin vẫn lệch sau khi build lại: "
                f"{self.index.ntotal} vector != {len(self.raw_data)} tag."
            )

    @staticmethod
    def _invalidate_stale_index(context_path, context_vector_path, input_datatype):
        if not os.path.exists(context_vector_path):
            return

        n_tags = len(semantic_extract.generate_raw_data(context_path, input_datatype))
        try:
            ntotal = faiss.read_index(context_vector_path).ntotal
        except Exception:
            logger.exception("Không đọc được tag_embedding.bin, sẽ build lại")
            ntotal = None

        if ntotal == n_tags:
            return

        logger.warning(
            "tag_embedding.bin lệch (%s vector, cần %d tag từ %s). Build lại.",
            ntotal,
            n_tags,
            context_path,
        )
        try:
            os.remove(context_vector_path)
        except OSError:
            logger.exception("Không xoá được %s", context_vector_path)

    def __call__(
        self,
        query: str,
        k: int = 3,
    ):
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
    print(obj("xe hơi màu đỏ.", 3))
    # print(len(obj.raw_data))
    # print(obj.index.ntotal)
