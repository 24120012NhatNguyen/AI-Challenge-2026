import copy
import json
from typing import Optional

import numpy as np
from fastapi import FastAPI
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


# Helper functions
def get_search_space(id):
    # id starting from 1 to 4
    search_space = []
    video_space = VideoDivision[f"list_{id}"]
    for video_id in video_space:
        search_space.extend(Videoid2imgid[video_id])
    return search_space


SearchSpace = dict()
for i in range(1, 5):
    SearchSpace[i] = np.array(get_search_space(i)).astype("int64")
SearchSpace[0] = TotalIndexList


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
        index = SearchSpace[search_space_index]
    else:
        index = np.intersect1d(index, SearchSpace[search_space_index])
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
        index = SearchSpace[search_space_index]
    else:
        index = np.intersect1d(index, SearchSpace[search_space_index])
    k = min(k, len(index))

    # Parse json input
    object_input = parse_data(request.model_dump(), VisualEncoder)
    ocr_input = None if request.ocr == "" else request.ocr
    asr_input = None if request.asr == "" else request.asr

    semantic = False
    keyword = True
    lst_scores, list_ids, _, list_image_paths = CosineFaiss.context_search(
        object_input=object_input,
        ocr_input=ocr_input,
        asr_input=asr_input,
        k=k,
        semantic=semantic,
        keyword=keyword,
        index=index,
        useid=request.useid,
    )

    data = group_result_by_video(lst_scores, list_ids, list_image_paths)
    return data


@app.post("/getrec")
def getrec(request: TagRequest):
    logger.info("get tag recommendation")
    k = 50
    text_query = request.text
    tag_outputs = TagRecommendation(text_query, k)
    return tag_outputs


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
        "shots": shots,
        "selected_shot": scene_idx[3],
    }
    return data


@app.post("/feedback")
def feed_back(request: FeedbackRequest):
    k = request.k
    prev_result = request.videos
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
