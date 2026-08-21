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

# Thư mục cache tf-idf của ASR. Trên Kaggle, dict/ thường là symlink tới
# /kaggle/input (CHỈ ĐỌC) nên không build lại cache tại chỗ được. Nếu dữ liệu
# ASR thô đúng nhưng cache lệch, đặt biến môi trường ASR_CACHE_DIR trỏ tới một
# thư mục ghi được (vd. /kaggle/working/asr_cache_rebuilt) để build lại ở đó:
#     os.environ["ASR_CACHE_DIR"] = "/kaggle/working/asr_cache_rebuilt"
# Nếu dữ liệu thô vốn đã hỏng (không khớp mapping) thì build lại cũng vô ích,
# khi đó ASR sẽ tự tắt và app vẫn chạy bình thường.
DEFAULT_ASR_CACHE_DIR = os.path.join(PROJECT_ROOT, "dict/bin/audio_bin")


def resolve_asr_cache_path():
    return os.environ.get("ASR_CACHE_DIR") or DEFAULT_ASR_CACHE_DIR


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
        context_vector_path=None,
        input_datatype="json",
        output_datatype="bin",
        test_mode=False,  # Enable to load raw data for debugging mode
        enable_semantic=False,
    ):
        if context_path is None:
            context_path = resolve_asr_context_path()
        if context_vector_path is None:
            context_vector_path = resolve_asr_cache_path()
        self.context_path = context_path
        self.context_vector_path = context_vector_path
        logger.info("ASR raw data directory: %s", context_path)
        logger.info("ASR cache directory: %s", context_vector_path)

        # Tạo thư mục cache nếu thiếu; thất bại (read-only) thì để _load_sparse
        # xử lý và tắt ASR, không được ném lỗi làm chết app.
        if not os.path.exists(context_vector_path):
            try:
                os.makedirs(context_vector_path, exist_ok=True)
            except OSError:
                logger.warning(
                    "Không tạo được thư mục cache ASR %s (chỉ đọc)", context_vector_path
                )

        tfidf_path = os.path.join(context_vector_path, "tfidf_transform_speech.pkl")
        matrix_path = os.path.join(
            context_vector_path, "sparse_context_matrix_speech.npz"
        )

        self.enable_semantic = enable_semantic
        # Cờ cho biết ASR có dùng được không. Dữ liệu ASR hỏng KHÔNG được phép
        # làm chết app - ASR chỉ là 1 trong nhiều modality, text search và các
        # tính năng khác phải chạy bình thường.
        self.available = False
        self.unavailable_reason = ""
        self.tfidf_transform = None
        self.context_matrix = None
        self.raw_data = None

        if enable_semantic:
            try:
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
            except Exception:
                logger.exception("Không load được phần semantic của ASR, tắt semantic")
                self.enable_semantic = False

        try:
            self._load_sparse(context_path, input_datatype, tfidf_path, matrix_path,
                              context_vector_path, test_mode)
        except Exception as e:
            # Bất kỳ lỗi nào ở đây cũng chỉ tắt ASR, không ném lên trên.
            logger.exception("Không khởi tạo được ASR retrieval")
            self._disable(f"{type(e).__name__}: {e}")

        if self.available:
            logger.info(
                "ASR sẵn sàng: ma trận %s", tuple(self.context_matrix.shape)
            )
        else:
            logger.warning(
                "=" * 70
                + "\nASR KHÔNG KHẢ DỤNG - %s\n"
                "App vẫn chạy bình thường; ô ASR trên Panel sẽ không trả kết quả.\n"
                "Các tính năng khác (text search, KNN, OCR, object, tag) không bị ảnh hưởng.\n"
                + "=" * 70,
                self.unavailable_reason,
            )

    def _disable(self, reason):
        self.available = False
        self.unavailable_reason = reason
        self.context_matrix = None
        self.raw_data = None

    def _load_sparse(self, context_path, input_datatype, tfidf_path, matrix_path,
                     context_vector_path, test_mode):
        """Nạp tf-idf + ma trận ASR. Tự tắt ASR nếu dữ liệu không dùng được."""
        # Đọc corpus ASR thô để biết số dòng ma trận ĐÚNG phải là bao nhiêu.
        raw_data = semantic_extract.generate_raw_data(context_path, input_datatype)
        expected_rows = len(raw_data)
        logger.info("ASR corpus size (expected matrix rows): %d", expected_rows)

        if expected_rows == 0:
            self._disable(
                f"không đọc được câu ASR nào từ '{context_path}' "
                f"(thiếu thư mục dict/audio_ASR hoặc dict/audio_ARS)"
            )
            return

        cache_ok = self._cache_matches(matrix_path, tfidf_path, expected_rows)

        if not cache_ok:
            # Cache lệch. Chỉ build lại được khi thư mục GHI ĐƯỢC. Trên Kaggle
            # dict/ là symlink tới /kaggle/input (read-only) nên bước này sẽ
            # thất bại -> tắt ASR thay vì làm chết app.
            if not self._try_invalidate(tfidf_path, matrix_path, expected_rows):
                return

        try:
            load_file.__init__(
                self,
                clean_data_path=None,  # clean_data_path and context can't not be None at the same time
                save_tfids_object_path=context_vector_path,
                update=False,
                all_datatpye=["speech"],
                context_data=raw_data,
                ngram_range=(1, 3),
                input_datatype="json",
            )
        except OSError as e:
            self._disable(f"không build lại được cache ASR (thư mục chỉ đọc): {e}")
            return

        if not (os.path.exists(tfidf_path) and os.path.exists(matrix_path)):
            self._disable("thiếu file tf-idf/ma trận ASR sau khi build")
            return

        with open(tfidf_path, "rb") as f:
            tfidf_transform = pickle.load(f)
        context_matrix = scipy.sparse.load_npz(matrix_path)

        if context_matrix.shape[0] != expected_rows:
            # Đây là lỗi dữ liệu gốc của dataset, không sửa được từ phía code.
            self._disable(
                f"ma trận ASR lệch kích thước: {context_matrix.shape[0]} dòng "
                f"!= {expected_rows} câu ASR thô. Dữ liệu dataset hỏng, "
                f"không thể sửa từ phía code"
            )
            return

        self.tfidf_transform = tfidf_transform
        self.context_matrix = context_matrix
        self.raw_data = raw_data if test_mode else None
        self.available = True

    @staticmethod
    def _cache_matches(matrix_path, tfidf_path, expected_rows):
        if not (os.path.exists(tfidf_path) and os.path.exists(matrix_path)):
            return False
        try:
            return scipy.sparse.load_npz(matrix_path).shape[0] == expected_rows
        except Exception:
            logger.exception("Không đọc được ma trận ASR")
            return False

    def _try_invalidate(self, tfidf_path, matrix_path, expected_rows):
        """Thử xoá cache lệch để build lại. Trả False nếu không thể (read-only)."""
        try:
            rows = scipy.sparse.load_npz(matrix_path).shape[0]
        except Exception:
            rows = None

        logger.warning(
            "Ma trận ASR lệch kích thước (%s dòng, cần %d). Thử xoá cache và build lại.",
            rows,
            expected_rows,
        )
        for path in (tfidf_path, matrix_path):
            if not os.path.exists(path):
                continue
            try:
                os.remove(path)
            except OSError as e:
                # Kaggle: dict/ trỏ tới /kaggle/input, chỉ đọc.
                self._disable(
                    f"ma trận ASR lệch ({rows} dòng != {expected_rows}) và không "
                    f"xoá được cache để build lại (thư mục chỉ đọc): {e}"
                )
                return False
        return True

    def __call__(
        self,
        query: str,
        k: int = 3,
        index=None,
        semantic: bool = True,
        keyword: bool = True,
    ):
        if not self.available:
            # Dữ liệu ASR hỏng -> trả rỗng thay vì đụng vào ma trận lệch.
            logger.warning("Bỏ qua truy vấn ASR: %s", self.unavailable_reason)
            return np.array([]), np.array([], dtype="int64")

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
        if not self.available:
            return np.array([]), np.array([], dtype="int64")

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
    print("available:", obj.available, obj.unavailable_reason)
    print(obj("một người đàn ông đang đi bộ trên cầu", 3))
