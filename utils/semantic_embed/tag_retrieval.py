import os

# import glob
# import torch
# import numpy as np
# from typing import List
# import torch.nn.functional as F
import faiss

# from transformers import AutoTokenizer, AutoModel
from utils.semantic_extract import semantic_extract


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
            os.mkdir(os.path.join(PROJECT_ROOT, "dict/bin/tag_bin"))

        super().__init__(
            model,
            context_path,
            context_vector_path,
            input_datatype,
            output_datatype,
        )
        self.index = faiss.read_index(context_vector_path)

    def __call__(
        self,
        query: str,
        k: int = 3,
    ):
        query_embed = self.get_embedding([query]).to("cpu").numpy()
        _, index = self.index.search(query_embed, k)
        result = [self.raw_data[idx] for idx in index[0]]
        return result


if __name__ == "__main__":
    obj = tag_retrieval()
    print(obj("xe hơi màu đỏ.", 3))
    # print(len(obj.raw_data))
    # print(obj.index.ntotal)
