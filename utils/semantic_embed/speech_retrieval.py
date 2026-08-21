import os
import faiss
import pickle
import numpy as np
import scipy
from utils.semantic_extract import semantic_extract
from utils.object_retrieval_engine.object_retrieval import load_file
from utils.combine_utils import merge_searching_results_by_addition
from utils.logger_config import get_logger

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

# Thư mục dữ liệu ASR thô. Tuỳ bản dataset mà thư mục được đặt tên "audio_ASR"
# hoặc "audio_ARS" (lỗi đánh máy trong dataset gốc). Trỏ sai tên -> corpus rỗng
# -> ma trận tf-idf lệch kích thước với dict/audio_id2img_id.json -> IndexError
# khi tra cứu. Vì vậy luôn dò tên thật trên đĩa thay vì hard-code.
ASR_DIR_CANDIDATES = ("audio_ASR", "audio_ARS")


def resolve_asr_context_path(project_root=PROJECT_ROOT):
    """Trả về đường dẫn thật của thư mục ASR thô (audio_ASR hoặc audio_ARS)."""
    for name in ASR_DIR_CANDIDATES:
        candidate = os.path.join(project_root, "dict", name)
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(project_root, "dict", ASR_DIR_CANDIDATES[0])


class speech_retrieval(semantic_extract, load_file):
    def __init__(
        self,
        model="sentence-transformers/stsb-xlm-r-multilingual",
        context_path=None,
        context_vector_path=os.path.join(PROJECT_ROOT, "dict/bin/audio_bin"),
        input_datatype="json",
        output_datatype="bin",
        test_mode=False,  # Enable to load raw data for debugging mode
        enable_semantic=False,
    ):
        if context_path is None:
            context_path = resolve_asr_context_path()
        self.context_path = context_path
        logger.info("ASR raw data directory: %s", context_path)

        if not os.path.exists(os.path.join(PROJECT_ROOT, "dict/bin")):
            os.makedirs(os.path.join(PROJECT_ROOT, "dict/bin"), exist_ok=True)

        if not os.path.exists(context_vector_path):
            os.makedirs(context_vector_path, exist_ok=True)

        tfidf_path = os.path.join(context_vector_path, "tfidf_transform_speech.pkl")
        matrix_path = os.path.join(
            context_vector_path, "sparse_context_matrix_speech.npz"
        )

        self.enable_semantic = enable_semantic
        if enable_semantic:
            semantic_extract.__init__(
                self,
                model=model,
                context_path=context_path,
                context_vector_path=os.path.join(
                    context_vector_path, "embed_audio.bin"
                ),
                input_datatype=input_datatype,
                output_datatype=output_datatype,
            )
            self.index = faiss.read_index(
                os.path.join(context_vector_path, "embed_audio.bin")
            )

        # Luôn đọc corpus ASR thô (nhanh, chỉ vài trăm file json) để biết số dòng
        # ma trận ĐÚNG phải là bao nhiêu, rồi so với ma trận đã build sẵn.
        self.raw_data = semantic_extract.generate_raw_data(
            context_path, input_datatype
        )
        expected_rows = len(self.raw_data)
        logger.info("ASR corpus size (expected matrix rows): %d", expected_rows)

        if expected_rows == 0:
            raise RuntimeError(
                f"Không đọc được câu ASR nào từ '{context_path}'. "
                f"Kiểm tra lại thư mục dữ liệu ASR thô (dict/audio_ASR hoặc dict/audio_ARS)."
            )

        self._invalidate_stale_cache(tfidf_path, matrix_path, expected_rows)

        load_file.__init__(
            self,
            clean_data_path=None,  # clean_data_path and context can't not be None at the same time
            save_tfids_object_path=context_vector_path,
            update=False,
            all_datatpye=["speech"],
            context_data=self.raw_data,
            ngram_range=(1, 3),
            input_datatype="json",
        )
        with open(tfidf_path, "rb") as f:
            self.tfidf_transform = pickle.load(f)
        self.context_matrix = scipy.sparse.load_npz(matrix_path)

        if self.context_matrix.shape[0] != expected_rows:
            raise RuntimeError(
                f"Ma trận ASR vẫn lệch sau khi build lại: "
                f"{self.context_matrix.shape[0]} dòng != {expected_rows} câu ASR thô. "
                f"Dữ liệu thô không khớp mapping - kiểm tra lại dataset."
            )

        if not test_mode:
            # raw_data chỉ cần cho lúc build; giải phóng cho nhẹ RAM
            self.raw_data = None

    @staticmethod
    def _invalidate_stale_cache(tfidf_path, matrix_path, expected_rows):
        """Xoá cache tf-idf cũ nếu số dòng ma trận không khớp corpus thô.

        Đây chính là nguyên nhân lỗi `IndexError: index (...) out of range`:
        ma trận được build từ một bộ dữ liệu ASR khác (nhỏ hơn) so với
        dict/audio_id2img_id.json hiện tại.
        """
        if not (os.path.exists(tfidf_path) and os.path.exists(matrix_path)):
            return

        try:
            rows = scipy.sparse.load_npz(matrix_path).shape[0]
        except Exception:
            logger.exception("Không đọc được ma trận ASR, sẽ build lại")
            rows = None

        if rows == expected_rows:
            return

        logger.warning(
            "Ma trận ASR lệch kích thước (%s dòng, cần %d). Xoá cache và build lại.",
            rows,
            expected_rows,
        )
        for path in (tfidf_path, matrix_path):
            try:
                os.remove(path)
            except OSError:
                logger.exception("Không xoá được %s", path)

    def __call__(
        self,
        query: str,
        k: int = 3,
        index=None,
        semantic: bool = True,
        keyword: bool = True,
    ):
        merge_scores = []
        merge_idx_image = []

        if semantic and self.enable_semantic:
            scores, idx_image = self.caculate_semantic(query, k, index)
            scores = scores.flatten()
            idx_image = idx_image.flatten()
            merge_scores.append(scores)
            merge_idx_image.append(idx_image)

        if keyword:
            scores, idx_image = self.caculate_sparse(query, k, index)
            merge_scores.append(scores)
            merge_idx_image.append(idx_image)

        if semantic and keyword and self.enable_semantic:
            scores, idx_image = merge_searching_results_by_addition(
                merge_scores, merge_idx_image
            )

        return scores, idx_image

    def caculate_semantic(
        self,
        query: str,
        k: int = 3,
        index=None,
    ):
        query_embed = self.get_embedding([query]).to("cpu").numpy()
        if index is None:
            scores, sorted_index = self.index.search(query_embed, k)
        else:
            id_selector = faiss.IDSelectorArray(index)
            scores, sorted_index = self.index.search(
                query_embed, k, params=faiss.SearchParametersIVF(sel=id_selector)
            )
        return scores, sorted_index

    def caculate_sparse(
        self,
        query: str,
        k: int,
        index=None,
    ):
        vectorize = self.tfidf_transform.transform([query])
        if index is None:
            # awesome_cossim_topn(a, b, N, 0.01, use_threads=True, n_jobs=4, return_best_ntop=True)
            # scores = cosine_similarity(vectorize, self.context_matrix[transform_type])[0]
            scores = vectorize.dot(self.context_matrix.T).toarray()[0]
            sort_index = np.argsort(scores)[::-1][:k]
            scores = scores[sort_index]
        else:
            # Bỏ qua các audio_idx nằm ngoài ma trận thay vì để scipy ném
            # IndexError và làm sập cả request /panel.
            index = np.asarray(index).astype("int64")
            n_rows = self.context_matrix.shape[0]
            valid = index[(index >= 0) & (index < n_rows)]
            if valid.size != index.size:
                logger.warning(
                    "ASR: bỏ qua %d/%d audio_idx nằm ngoài ma trận (%d dòng)",
                    index.size - valid.size,
                    index.size,
                    n_rows,
                )
            if valid.size == 0:
                return np.array([]), np.array([], dtype="int64")
            # scores = cosine_similarity(vectorize, self.context_matrix[transform_type][index,:])[0]
            scores = vectorize.dot(self.context_matrix[valid, :].T).toarray()[0]
            sort_index = np.argsort(scores)[::-1][:k]
            scores = scores[sort_index]
            sort_index = valid[sort_index]
        return scores, sort_index


if __name__ == "__main__":
    obj = speech_retrieval()
    print(obj("một người đàn ông đang đi bộ trên cầu", 3))
    pass
