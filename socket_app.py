import json
import os

import socketio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from utils.logger_config import get_logger
from utils.models import QuestionNameRequest, UsernameRequest, UserRequest
from utils.search_utils import _parse_keyframe_path

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "https://*.ngrok-free.app",
        "http://localhost:3000",
        "http://localhost",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Create Socket.IO server
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=[
        "*",
        "https://*.ngrok-free.app",
        "http://localhost:3000",
        "http://localhost",
    ],
    ping_timeout=60,
    ping_interval=25,
)

###################### Initialize dict ########################
json_path = "dict/id2img.json"
back_up_folder = "back_up"
if not os.path.exists(back_up_folder):
    os.mkdir(back_up_folder)
with open(json_path, "r") as f:
    DictImagePath = json.load(f)
    DictImagePath = {int(k): v for k, v in DictImagePath.items()}

if os.path.exists(f"{back_up_folder}/answer.json"):
    with open(f"{back_up_folder}/answer.json", "r") as f:
        AnswerDict = json.load(f)
else:
    AnswerDict = dict()

if os.path.exists(f"{back_up_folder}/user.json"):
    with open(f"{back_up_folder}/user.json", "r") as f:
        UserDict = json.load(f)
else:
    UserDict = dict()

if os.path.exists(f"{back_up_folder}/reorder_status.json"):
    with open(f"{back_up_folder}/reorder_status.json", "r") as f:
        ReorderStatus = json.load(f)
else:
    ReorderStatus = dict()

if os.path.exists(f"{back_up_folder}/answer_ignore.json"):
    with open(f"{back_up_folder}/answer_ignore.json", "r") as f:
        AnswerIgnoreDict = json.load(f)
else:
    AnswerIgnoreDict = dict()
###############################################################


####################### Helper Utils ##########################
def store_answer():
    with open(f"{back_up_folder}/answer.json", "w") as f:
        json.dump(AnswerDict, f)


def store_user():
    with open(f"{back_up_folder}/user.json", "w") as f:
        json.dump(UserDict, f)


def store_status():
    with open(f"{back_up_folder}/reorder_status.json", "w") as f:
        json.dump(ReorderStatus, f)


def store_ignore():
    with open(f"{back_up_folder}/answer_ignore.json", "w") as f:
        json.dump(AnswerIgnoreDict, f)


def add_submit(ques_name, ques_idx):
    ques_idx = int(ques_idx)
    if not AnswerDict.get(ques_name, False):
        AnswerDict[ques_name] = [ques_idx]
    else:
        if ques_idx not in AnswerDict[ques_name]:
            AnswerDict[ques_name].append(ques_idx)
    if not ReorderStatus.get(ques_name, False):
        ReorderStatus[ques_name] = dict(status=False, owner="")
        store_status()
    store_answer()


def add_ignore(ques_name, ques_idx, autoIgnore):
    if not isinstance(ques_idx, list):
        ques_idx = [ques_idx]

    for i in range(len(ques_idx)):
        ques_idx[i] = int(ques_idx[i])

    if not AnswerIgnoreDict.get(ques_name, False):
        AnswerIgnoreDict[ques_name] = ques_idx
    else:
        for idx in ques_idx:
            if idx not in AnswerIgnoreDict[ques_name]:
                AnswerIgnoreDict[ques_name].append(idx)
            elif not autoIgnore:
                AnswerIgnoreDict[ques_name].remove(idx)
    store_ignore()


def add_user(user, ques_name):
    if not UserDict.get(user, False):
        UserDict[user] = [ques_name]
    else:
        if ques_name not in UserDict[user]:
            UserDict[user].append(ques_name)
            UserDict[user] = sorted(UserDict[user])
    store_user()


def clear_submit_helper(ques_name, ques_idx):
    if AnswerDict.get(ques_name, False):
        if ques_idx in AnswerDict[ques_name]:
            AnswerDict[ques_name].remove(ques_idx)
    else:
        logger.warning("Question name: %s not exist", ques_name)
    store_answer()


def clear_ignore_helper(ques_name, ques_idx):
    if AnswerIgnoreDict.get(ques_name, False):
        if ques_idx in AnswerIgnoreDict[ques_name]:
            AnswerIgnoreDict[ques_name].remove(ques_idx)
    else:
        logger.warning("Question name: %s not exist", ques_name)
    store_ignore()


def index2info(lst_idxs):
    info = {
        "lst_idxs": lst_idxs,
        "lst_keyframe_idxs": [],
        "lst_keyframe_paths": [],
        "lst_video_idxs": [],
    }
    for idx in lst_idxs:
        image_path = DictImagePath[idx]["image_path"]
        data_part, video_id, frame_id = _parse_keyframe_path(image_path)
        key = f"{data_part}_{video_id}".replace("_extract", "").replace("_extra", "")
        frame_id = int(frame_id)

        info["lst_keyframe_idxs"].append(frame_id)
        info["lst_keyframe_paths"].append(image_path)
        info["lst_video_idxs"].append(key)
    return info


def check_owned_all(username):
    # return all questions after getting checked ownership
    all_ques = sorted(list(AnswerDict.keys()))
    checked_ques = []

    if not UserDict.get(username, False):
        for ques in all_ques:
            checked_ques.append({"question": ques, "owned": False})
        return checked_ques

    # if user exists
    for ques in all_ques:
        if ques in UserDict[username]:
            checked_ques.append({"question": ques, "owned": True})
        else:
            checked_ques.append({"question": ques, "owned": False})

    logger.debug("res: %s", checked_ques)
    return checked_ques


def check_ignore():
    return list(AnswerIgnoreDict.keys())


##################### Submit ##################################
@sio.event
async def submit(sid, data):
    logger.info("submit")
    ques_name = data["questionName"]
    ques_idx = int(data["idx"])
    user = data["user"]
    add_submit(ques_name, ques_idx)
    add_user(user, ques_name)
    result = {"questionName": ques_name, "data": index2info(AnswerDict[ques_name])}
    await sio.emit("submit", result)


@sio.event
async def clearsubmit(sid, data):
    logger.info("clear submit")
    ques_name = data["questionName"]
    ques_idx = int(data["idx"])
    clear_submit_helper(ques_name, ques_idx)
    result = {"questionName": ques_name, "data": index2info(AnswerDict[ques_name])}
    await sio.emit("clearsubmit", result)


##################### Ignore #################################
@sio.event
async def ignore(sid, data):
    logger.info("ignore")
    ques_name = data["questionName"]
    ques_idx = data["idx"]
    add_ignore(ques_name, ques_idx, data["autoIgnore"])
    result = {"questionName": ques_name, "data": AnswerIgnoreDict[ques_name]}
    await sio.emit("ignore", result)


@sio.event
async def clearignore(sid, data):
    logger.info("clear ignore")
    ques_name = data["questionName"]
    ques_idx = data["idx"]
    clear_ignore_helper(ques_name, ques_idx)
    result = {
        "questionName": ques_name,
        "data": index2info(AnswerIgnoreDict[ques_name]),
    }
    await sio.emit("clearsubmit", result)


##################### Reorder #################################
@sio.event
async def reorder(sid, data):
    logger.info("re order")
    ques_name = data["questionName"]
    lst_idxs = data["data"]["lst_idxs"]
    if AnswerDict.get(ques_name, False):
        AnswerDict[ques_name] = lst_idxs
        store_answer()

    # Reset status of active_reorder once received reroder
    ReorderStatus[ques_name]["status"] = False
    ReorderStatus[ques_name]["owner"] = ""

    result = {"questionName": ques_name, "data": index2info(AnswerDict[ques_name])}
    await sio.emit("reorder", result)


@sio.event
async def activereorder(sid, data):
    logger.info("active reorder")
    ques_name = data["questionName"]
    user = data["user"]
    is_admin = data["isAdmin"]
    status = {
        "ques_name": ques_name,
        "user": user,
        "is_accepted": False,
    }
    # If is_admin: pass
    if ReorderStatus[ques_name]["status"] and not is_admin:
        logger.error("Reorder an active question error")
        # emit's first argument must be the same name as that of the channel on client-side
        await sio.emit("activereorder", status)
    else:
        ReorderStatus[ques_name]["status"] = True
        ReorderStatus[ques_name]["owner"] = user
        status["is_accepted"] = True
        await sio.emit("activereorder", status)


@sio.event
async def viewsubmitted(sid, data):
    logger.info("view submitted")
    ques_name = data["questionName"]
    if AnswerDict.get(ques_name, False):
        result = {"questionName": ques_name, "data": index2info(AnswerDict[ques_name])}
        await sio.emit("viewsubmitted", result)
    else:
        await sio.emit("viewsubmitted", {})


@app.get("/submit")
async def submit_get(item: str = "", frame: str = "", session: str = ""):
    """HTTP GET /submit for competition submission (FE calls with ?item=...&frame=...&session=...)."""
    if not item or not frame:
        return {"description": "Missing item or frame", "status": "error"}
    # Có thể lưu submission vào file/DB tại đây nếu cần
    return {
        "description": f"Submitted item={item}, frame={frame}, session={session}",
        "status": "success",
    }


@app.get("/getallques")
async def get_all_ques():
    all_ques = sorted(list(AnswerDict.keys()))
    return all_ques


@app.post("/getsubmitques")
async def get_submit_ques(request: UserRequest):
    user = request.user
    if UserDict.get(user, False):
        return UserDict[user]
    else:
        return []


@app.post("/getquestions")
async def get_questions(request: UsernameRequest):
    username = request.username
    checked_ques = check_owned_all(username)
    return checked_ques


@app.post("/getignoredquestions")
async def get_ignored_questions():
    return check_ignore()


@app.post("/getignore")
async def get_ignore(request: QuestionNameRequest):
    ques_name = request.questionName
    if AnswerIgnoreDict.get(ques_name, False):
        result = {"questionName": ques_name, "data": AnswerIgnoreDict[ques_name]}
    else:
        result = {"questionName": ques_name, "data": []}
    return result


# Create combined ASGI application
socket_app = socketio.ASGIApp(sio, app)

# Running app
if __name__ == "__main__":
    uvicorn.run(
        "socket_app:socket_app",
        host="0.0.0.0",
        port=8081,
        reload=True,
    )
