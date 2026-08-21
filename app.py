import copy
import json
from typing import Any, Dict, Optional, Union

import numpy as np
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from utils.combine_utils import merge_searching_results_by_addition
from utils.context_encoding import VisualEncoding
from utils.faiss_processing import MyFaiss
from utils.logger_config import get_logger
from utils.models import (
    FeedbackRequest,
    PanelSearchRequest,
    TagRequest,
    TextSearchRequest,
    TranslateRequest,
)
from utils.parse_frontend import parse_data
from utils.search_utils import (
    _parse_keyframe_path,
    canonical_video_name,
    group_result_by_video,
    search_by_filter,
)
from utils.semantic_embed.tag_retrieval import tag_retrieval

logger = get_logger(__name__)

# Define paths
json_path = "dict/id2img.json"
audio_json_path = "dict/audio_id2img_id.json"
scene_path = "dict/scene_id2info.json"
bin_nomic_file = "dict/Nomic_cosine.bin"
bin_clipv2_file = "dict/CLIPv2_cosine.bin"
video_division_path = "dict/video_division_tag.json"
img2audio_json_path = "dict/img_id2audio_id.json"

# Initialize components
VisualEncoder = VisualEncoding()
CosineFaiss = MyFaiss(
    bin_nomic_file, bin_clipv2_file, json_path, audio_json_path, img2audio_json_path
)
TagRecommendation = tag_retrieval()
DictImagePath = CosineFaiss.id2img
TotalIndexList = np.array(list(range(len(DictImagePath)))).astype("int64")

with open(scene_path, "r") as f:
    Sceneid2info = json.load(f)

with open(video_division_path, "r") as f:
    VideoDivision = json.load(f)

with open("dict/video_id2img_id.json", "r") as f:
    Videoid2imgid = json.load(f)


def _build_video_key_aliases():
    """Map mọi cách viết video_id mà người dùng có thể gõ -> key thật trong
    dict/video_id2img_id.json.

    Key thật có dạng "L21_V001" (không có phần data_part hậu tố), trong khi UI
    lại hiển thị "L21_a_V001" (ghép từ đường dẫn keyframe
    /static/images/Keyframes/L21_a/V001/000000.jpg). Nếu chỉ tra trực tiếp thì
    ID copy từ UI luôn báo "Video không tồn tại".
    """
    aliases = dict()

    def put(name, canonical):
        if name:
            aliases.setdefault(name.strip().upper(), canonical)

    for canonical, idxs in Videoid2imgid.items():
        put(canonical, canonical)
        if not idxs:
            continue
        data_part, video_id, _ = _parse_keyframe_path(
            DictImagePath[idxs[0]]["image_path"]
        )
        # "L21_a" + "V001" -> "L21_a_V001" (đúng cái UI hiển thị)
        put(f"{data_part}_{video_id}", canonical)
        put(f"{data_part}/{video_id}", canonical)
        put(f"{data_part}_{video_id}".replace("_extract", "").replace("_extra", ""),
            canonical)
    return aliases


VideoKeyAliases = _build_video_key_aliases()


def resolve_video_key(video_id: str):
    """Chuẩn hoá video_id người dùng nhập về key thật, hoặc None nếu không có."""
    if not video_id:
        return None
    return VideoKeyAliases.get(video_id.strip().upper())


# Helper functions
def get_search_space(id):
    # id starting from 1 to N (N = số list_* trong dict/video_division_tag.json)
    search_space = []
    video_space = VideoDivision[f"list_{id}"]
    for video_id in video_space:
        if video_id in Videoid2imgid:
            search_space.extend(Videoid2imgid[video_id])
        else:
            logger.warning("search space: video '%s' không có trong video_id2img_id", video_id)
    return search_space


# Trước đây chỉ build 1..4 trong khi file có tới list_5 -> chọn Space=5 trên UI
# làm KeyError và trả 500. Giờ build đúng số list thật sự có.
N_SEARCH_SPACE = sum(1 for key in VideoDivision if key.startswith("list_"))
SearchSpace = dict()
for i in range(1, N_SEARCH_SPACE + 1):
    SearchSpace[i] = np.array(get_search_space(i)).astype("int64")
SearchSpace[0] = TotalIndexList


def get_search_space_index(search_space_index):
    """Trả về mảng index của search space, tự lùi về 0 (toàn bộ) nếu không hợp lệ."""
    if search_space_index not in SearchSpace:
        logger.warning(
            "search_space=%s không hợp lệ (hợp lệ: 0..%d), dùng 0 (toàn bộ)",
            search_space_index,
            N_SEARCH_SPACE,
        )
        return SearchSpace[0]
    return SearchSpace[search_space_index]


def get_near_frame(idx):
    image_info = DictImagePath[idx]
    scene_idx = image_info["scene_idx"].split("/")
    near_keyframes_idx = copy.deepcopy(
        Sceneid2info[scene_idx[0]][scene_idx[1]][scene_idx[2]][scene_idx[3]][
            "lst_keyframe_idxs"
        ]
    )
    return near_keyframes_idx


def get_related_ignore(ignore_index):
    total_ignore_index = []
    for idx in ignore_index:
        total_ignore_index.extend(get_near_frame(idx))
    return total_ignore_index


# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/data")
def index():
    pagefile = []
    for id, value in DictImagePath.items():
        if int(id) > 500:
            break
        pagefile.append({"imgpath": value["image_path"], "id": id})
    data = {"pagefile": pagefile}
    return data


@app.get("/imgsearch")
def image_search(k: int, imgid: int):
    logger.info("image search")
    lst_scores, list_ids, _, list_image_paths = CosineFaiss.image_search(imgid, k=k)
    data = group_result_by_video(lst_scores, list_ids, list_image_paths)
    return data


@app.post("/textsearch")
def text_search(request: TextSearchRequest):
    logger.info("text search")
    search_space_index = request.search_space
    k = request.k
    nomic = request.nomic
    clipv2 = request.clipv2
    text_query = request.textquery
    range_filter = request.range_filter

    logger.debug(
        "search_space_index: %s, k: %s, query: %s",
        search_space_index,
        k,
        text_query,
    )

    index = None
    if request.filter and request.id:  #
        index = np.array(request.id).astype("int64")
        k = min(k, len(index))
        logger.debug("using index")

    # Create list frames to list to keep (all frames minus ignore frames)
    keep_index = None
    ignore_index = None
    if request.ignore and request.ignore_idxs:
        ignore_index = get_related_ignore(np.array(request.ignore_idxs).astype("int64"))
        keep_index = np.delete(TotalIndexList, ignore_index)
        logger.debug("using ignore")

    if keep_index is not None:
        if index is not None:
            index = np.intersect1d(index, keep_index)
        else:
            index = keep_index

    if index is None:
        index = get_search_space_index(search_space_index)
    else:
        index = np.intersect1d(index, get_search_space_index(search_space_index))
    k = min(k, len(index))

    if nomic and clipv2:
        model_type = "both"
    elif nomic:
        model_type = "nomic"
    else:
        model_type = "clipv2"

    if (
        request.filtervideo != 0 and request.videos
    ):  # for temporal search, before or after specified keyframe
        logger.debug("filter video")
        mode = request.filtervideo
        prev_result = request.videos
        data = search_by_filter(
            prev_result,
            text_query,
            k,
            mode,
            model_type,
            range_filter,
            ignore_index,
            keep_index,
            Sceneid2info,
            DictImagePath,
            CosineFaiss,
        )
    else:
        if model_type == "both":
            scores_nomic, list_nomic_ids, _, _ = CosineFaiss.text_search(
                text_query, index=index, k=k, model_type="nomic"
            )
            scores_clipv2, list_clipv2_ids, _, _ = CosineFaiss.text_search(
                text_query, index=index, k=k, model_type="clipv2"
            )
            lst_scores, list_ids = merge_searching_results_by_addition(
                [scores_nomic, scores_clipv2], [list_nomic_ids, list_clipv2_ids]
            )
            infos_query = [
                CosineFaiss.id2img.get(idx)
                for idx in list_ids
                if CosineFaiss.id2img.get(idx)
            ]
            list_image_paths = [info["image_path"] for info in infos_query]
        else:
            lst_scores, list_ids, _, list_image_paths = CosineFaiss.text_search(
                text_query, index=index, k=k, model_type=model_type
            )
        logger.debug("List scores: %s", lst_scores)
        logger.debug("List ids: %s", list_ids)
        logger.debug("List image paths: %s", list_image_paths)
        data = group_result_by_video(lst_scores, list_ids, list_image_paths)

    return data


@app.post("/panel")
def panel(request: PanelSearchRequest):
    logger.info("panel search")
    k = request.k
    search_space_index = request.search_space

    index = None
    if request.useid and request.id:
        index = np.array(request.id).astype("int64")
        k = min(k, len(index))

    keep_index = None
    if request.ignore and request.ignore_idxs:
        ignore_index = get_related_ignore(np.array(request.ignore_idxs).astype("int64"))
        keep_index = np.delete(TotalIndexList, ignore_index)
        logger.debug("using ignore")

    if keep_index is not None:
        if index is not None:
            index = np.intersect1d(index, keep_index)
        else:
            index = keep_index

    if index is None:
        index = get_search_space_index(search_space_index)
    else:
        index = np.intersect1d(index, get_search_space_index(search_space_index))
    k = min(k, len(index))

    # Parse json input
    object_input = parse_data(request.model_dump(), VisualEncoder)
    ocr_input = None if request.ocr == "" else request.ocr
    asr_input = None if request.asr == "" else request.asr

    semantic = False
    keyword = True
    lst_scores, list_ids, _, list_image_paths, warnings = CosineFaiss.context_search(
        object_input=object_input,
        ocr_input=ocr_input,
        asr_input=asr_input,
        k=k,
        semantic=semantic,
        keyword=keyword,
        index=index,
        useid=request.useid,
    )

    videos = group_result_by_video(lst_scores, list_ids, list_image_paths)
    if warnings:
        logger.warning("panel search warnings: %s", warnings)
    # Trả về object thay vì list thuần để kèm được "warnings"; frontend chấp
    # nhận cả hai dạng (xem Panel.jsx) nên vẫn tương thích ngược.
    return {"videos": videos, "warnings": warnings}


# dict/tag/tag_list.txt lưu tag dạng có dấu cách ("race car") nhưng từ điển
# tf-idf của object retrieval lại dùng dạng gạch dưới ("race_car"). Nếu trả
# nguyên dạng có dấu cách, tag nhiều chữ được gợi ý sẽ không khớp gì khi bấm
# vào và gửi qua /panel.
TagVocab = set(CosineFaiss.object_retrieval.tag_corpus)


def _normalize_tag(tag: str) -> str:
    """Đưa tag về đúng dạng trong từ điển tf-idf (dạng gạch dưới)."""
    if tag in TagVocab:
        return tag
    underscored = tag.replace(" ", "_")
    return underscored if underscored in TagVocab else tag


@app.post("/getrec")
async def getrec(payload: Union[TagRequest, str, Dict[str, Any]] = Body(...)):
    """Gợi ý tag từ câu truy vấn.

    Frontend gửi thẳng một chuỗi JSON (vd. `"con mèo"`), nên endpoint chấp nhận
    cả chuỗi trần lẫn object {"text": "..."} để không trả 422.
    """
    logger.info("get tag recommendation")
    k = 50
    if isinstance(payload, TagRequest):
        text_query = payload.text
    elif isinstance(payload, dict):
        text_query = payload.get("text", "") or payload.get("textquery", "")
    else:
        text_query = payload or ""

    if not text_query.strip():
        return []
    if not getattr(TagRecommendation, "available", False):
        logger.warning(
            "Tag recommendation không khả dụng: %s",
            getattr(TagRecommendation, "unavailable_reason", ""),
        )
        return []
    return [_normalize_tag(tag) for tag in TagRecommendation(text_query, k)]


@app.get("/relatedimg")
def related_img(imgid: int):
    logger.info("related image")
    image_info = DictImagePath[imgid]
    image_path = image_info["image_path"]
    scene_idx = image_info["scene_idx"].split("/")

    video_info = copy.deepcopy(Sceneid2info[scene_idx[0]][scene_idx[1]])
    video_range = video_info[scene_idx[2]][scene_idx[3]]["shot_time"]

    near_keyframes = video_info[scene_idx[2]][scene_idx[3]]["lst_keyframe_paths"]
    near_keyframes.remove(image_path)

    data = {
        "video_range": video_range,
        "near_keyframes": near_keyframes,
    }
    return data


@app.get("/getvideoshot")
def get_video_shot(imgid: Optional[str] = None):
    logger.info("get video shot")

    if imgid == "undefined" or imgid is None:
        return {}

    id_query = int(imgid)
    image_info = DictImagePath[id_query]
    scene_idx = image_info["scene_idx"].split("/")
    shots = copy.deepcopy(Sceneid2info[scene_idx[0]][scene_idx[1]][scene_idx[2]])

    selected_shot = int(scene_idx[3])
    total_n_shots = len(shots)
    new_shots = dict()
    for select_id in range(
        max(0, selected_shot - 5), min(selected_shot + 6, total_n_shots)
    ):
        new_shots[str(select_id)] = shots[str(select_id)]
    shots = new_shots

    for shot_key in shots.keys():
        lst_keyframe_idxs = []
        for img_path in shots[shot_key]["lst_keyframe_paths"]:
            data_part, video_id, frame_id = _parse_keyframe_path(img_path)
            frame_id = int(frame_id)
            lst_keyframe_idxs.append(frame_id)
        shots[shot_key]["lst_idxs"] = shots[shot_key]["lst_keyframe_idxs"]
        shots[shot_key]["lst_keyframe_idxs"] = lst_keyframe_idxs

    data = {
        "collection": scene_idx[0],
        "video_id": scene_idx[1],
        # tên video chuẩn để nộp bài / tra FrameRange (vd. "L21_V001")
        "video_name": canonical_video_name(scene_idx[0], scene_idx[1]),
        "shots": shots,
        "selected_shot": scene_idx[3],
    }
    return data


@app.get("/framerange")
def frame_range(
    video_id: str,
    start: int,
    end: int,
    text_query: str = "",
    model_type: str = "nomic",
):
    """Trả về tất cả keyframe của video_id nằm trong khoảng [start, end].

    Nếu text_query được cung cấp, kết quả sẽ được xếp hạng theo mức khớp
    với text_query (dùng CosineFaiss.text_search) thay vì theo frame_id tăng dần.

    Args:
        video_id: ID video (ví dụ: "L21_V001")
        start: frame_id bắt đầu (bao gồm)
        end: frame_id kết thúc (bao gồm)
        text_query: (tuỳ chọn) query text để xếp hạng lại kết quả
        model_type: (tuỳ chọn) "nomic" | "clipv2" | "both" (mặc định "nomic")

    Returns:
        JSON với video_id, video_info (lst_keyframe_paths, lst_idxs, lst_keyframe_idxs), message.
    """
    logger.info("frame range: video_id=%s, start=%d, end=%d, text_query=%s, model_type=%s",
                video_id, start, end, text_query, model_type)

    canonical_id = resolve_video_key(video_id)
    all_idxs = Videoid2imgid.get(canonical_id) if canonical_id else None
    if all_idxs is None:
        return {
            "error": (
                f"Video '{video_id}' không tồn tại. "
                f"Định dạng hợp lệ: 'L21_a_V001' (như hiển thị trên UI) hoặc 'L21_V001'."
            ),
            "status_code": 404,
        }
    video_id = canonical_id

    # Filter keyframes within [start, end] range
    range_items = []  # list of (frame_id, idx, image_path)
    for idx in all_idxs:
        image_info = DictImagePath[idx]
        image_path = image_info["image_path"]
        _, _, frame_id_str = _parse_keyframe_path(image_path)
        frame_id = int(frame_id_str)
        if start <= frame_id <= end:
            range_items.append((frame_id, idx, image_path))

    if not range_items:
        return {
            "video_id": video_id,
            "video_info": {
                "lst_keyframe_paths": [],
                "lst_idxs": [],
                "lst_keyframe_idxs": [],
            },
            "message": "Không có keyframe nào trong khoảng này, thử mở rộng phạm vi",
        }

    if text_query and text_query.strip():
        # Re-rank by text similarity
        range_idxs = [item[1] for item in range_items]
        index = np.array(range_idxs).astype("int64")
        k = len(range_idxs)

        if model_type == "both":
            scores_nomic, list_nomic_ids, _, _ = CosineFaiss.text_search(
                text_query, index=index, k=k, model_type="nomic"
            )
            scores_clipv2, list_clipv2_ids, _, _ = CosineFaiss.text_search(
                text_query, index=index, k=k, model_type="clipv2"
            )
            lst_scores, list_ids = merge_searching_results_by_addition(
                [scores_nomic, scores_clipv2], [list_nomic_ids, list_clipv2_ids]
            )
            infos_query = [
                CosineFaiss.id2img.get(idx)
                for idx in list_ids
                if CosineFaiss.id2img.get(idx)
            ]
            list_image_paths = [info["image_path"] for info in infos_query]
        else:
            lst_scores, list_ids, _, list_image_paths = CosineFaiss.text_search(
                text_query, index=index, k=k, model_type=model_type
            )

        # Build response from re-ranked results
        result_paths = []
        result_idxs = []
        result_frame_ids = []
        for i, img_path in enumerate(list_image_paths):
            _, _, fid_str = _parse_keyframe_path(img_path)
            result_paths.append(img_path)
            result_idxs.append(int(list_ids[i]))
            result_frame_ids.append(int(fid_str))
    else:
        # Sort by frame_id ascending (temporal order)
        range_items.sort(key=lambda x: x[0])
        result_paths = [item[2] for item in range_items]
        result_idxs = [item[1] for item in range_items]
        result_frame_ids = [item[0] for item in range_items]

    return {
        "video_id": video_id,
        "video_info": {
            "lst_keyframe_paths": result_paths,
            "lst_idxs": result_idxs,
            "lst_keyframe_idxs": result_frame_ids,
        },
        "message": "",
    }


@app.post("/feedback")
def feed_back(request: FeedbackRequest):
    logger.info("feedback rerank")
    k = request.k
    prev_result = request.videos
    if isinstance(prev_result, dict):
        # client cũ có thể gửi {video_id: {...}}; đưa về dạng list thống nhất
        prev_result = list(prev_result.values())
    if not prev_result:
        return []
    lst_pos_vote_idxs = request.lst_pos_idxs
    lst_neg_vote_idxs = request.lst_neg_idxs
    lst_scores, list_ids, _, list_image_paths = CosineFaiss.reranking(
        prev_result, lst_pos_vote_idxs, lst_neg_vote_idxs, k
    )
    data = group_result_by_video(lst_scores, list_ids, list_image_paths)
    return data


@app.post("/translate")
def translate(request: TranslateRequest):
    text_query = request.textquery
    text_query_translated = CosineFaiss.translater(text_query)
    return text_query_translated


@app.get("/diagnostics")
def diagnostics():
    """Tự kiểm tra sức khoẻ hệ thống: kích thước các ma trận, mapping, search space.

    Dùng để phát hiện sớm lỗi lệch dữ liệu (vd. ma trận ASR/OCR không khớp số
    keyframe) trước khi thi đấu.
    """
    n_img = len(DictImagePath)
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    ocr_rows = CosineFaiss.ocr_retrieval.context_sparse_matrix_ocr.shape[0]
    add("ocr_matrix", ocr_rows == n_img, f"{ocr_rows} dòng / {n_img} keyframe")

    asr = CosineFaiss.asr_retrieval
    n_audio = len(CosineFaiss.audio_id2img_id)
    if getattr(asr, "available", False):
        asr_rows = asr.context_matrix.shape[0]
        add("asr_matrix", asr_rows == n_audio, f"{asr_rows} dòng / {n_audio} đoạn ASR")
    else:
        add("asr_matrix", False,
            f"ASR TẮT - {getattr(asr, 'unavailable_reason', 'không rõ lý do')}. "
            f"Các modality khác không bị ảnh hưởng.")

    add("tag_recommendation", getattr(TagRecommendation, "available", False),
        f"{TagRecommendation.index.ntotal} tag"
        if getattr(TagRecommendation, "available", False)
        else f"TẮT - {getattr(TagRecommendation, 'unavailable_reason', '')}")

    for name, matrix in CosineFaiss.object_retrieval.context_matrix.items():
        add(f"object_matrix_{name}", matrix.shape[0] == n_img,
            f"{matrix.shape[0]} dòng / {n_img} keyframe")

    add("faiss_nomic", CosineFaiss.index_nomic.ntotal == n_img,
        f"{CosineFaiss.index_nomic.ntotal} vector / {n_img} keyframe")
    add("faiss_clipv2", CosineFaiss.index_clipv2.ntotal == n_img,
        f"{CosineFaiss.index_clipv2.ntotal} vector / {n_img} keyframe")
    add("img_id2audio_id", len(CosineFaiss.img_id2audio_id) == n_img,
        f"{len(CosineFaiss.img_id2audio_id)} entry / {n_img} keyframe")

    missing_videos = [
        v for key in VideoDivision if key.startswith("list_")
        for v in VideoDivision[key] if v not in Videoid2imgid
    ]
    add("video_division", not missing_videos,
        f"{N_SEARCH_SPACE} search space (1..{N_SEARCH_SPACE}), "
        f"{len(missing_videos)} video thiếu mapping")

    # ASR / tag chỉ là modality phụ: tắt chúng KHÔNG làm hệ thống "không ok",
    # nhưng vẫn được liệt kê trong "degraded" để biết mà xử lý.
    optional = {"asr_matrix", "tag_recommendation"}
    degraded = [c["name"] for c in checks if not c["ok"] and c["name"] in optional]
    return {
        "ok": all(c["ok"] for c in checks if c["name"] not in optional),
        "degraded": degraded,
        "n_keyframes": n_img,
        "n_videos": len(Videoid2imgid),
        "asr_dir": getattr(CosineFaiss.asr_retrieval, "context_path", None),
        "example_video_id": next(iter(Videoid2imgid)),
        "checks": checks,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
