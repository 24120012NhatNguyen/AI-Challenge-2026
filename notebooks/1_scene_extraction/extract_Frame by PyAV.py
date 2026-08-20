import subprocess
import sys

try:
    import av
except ImportError:
    print("Thư viện 'av' chưa được cài đặt. Bắt đầu cài đặt...")
    subprocess.run([sys.executable, "-m", "pip", "install", "av", "-q"], check=True)
    import av

# ====== PHẦN 1: KHAI BÁO THƯ VIỆN ======
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from tqdm import tqdm

# ================== PHẦN 2: CẤU HÌNH ==================
# Các đường dẫn đầu vào và đầu ra cho pipeline
videos_dir = "/kaggle/input/batch-2"
scene_json_dirs = "/kaggle/input/scenes-segment/SceneJSON_batch2"
save_dir_all = "./Keyframes"
os.makedirs(save_dir_all, exist_ok=True)

# Lựa chọn các tập con dữ liệu để xử lý, giúp quản lý các lần chạy lớn
data_parts = ["K03", "K04"]

# ================== PHẦN 3: KHỞI TẠO MÔ HÌNH DEDUPLICATION ==================
# Sử dụng DINOv2 để tạo vector đặc trưng cho việc so sánh sự tương đồng về mặt ngữ nghĩa hình ảnh.
USE_DEDUP = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

try:
    # Tải mô hình DINOv2 pre-trained từ torch.hub
    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14", trust_repo=True
    ).to(device)
    model.eval()
    # Định nghĩa các bước tiền xử lý ảnh để phù hợp với đầu vào của DINOv2
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
except Exception as e:
    print(f"⚠️ Không thể tải mô hình DINOv2, tính năng deduplicate sẽ bị tắt. Lỗi: {e}")
    USE_DEDUP = False
    model, transform = None, None

# ================== PHẦN 4: CÁC HÀM CHỨC NĂNG ==================


def sample_frame_from_shot(start_idx, end_idx):
    """
    Lấy mẫu 5 khung hình theo phân phối đều từ một cảnh (shot).
    Chiến lược này đảm bảo bao quát được các diễn biến có thể xảy ra ở
    đầu, giữa và cuối cảnh.
    """
    s, e = int(start_idx), int(end_idx)
    if e < s:
        e = s
    return [
        s,
        int(s + (e - s) * 0.25),
        int(s + (e - s) * 0.5),
        int(s + (e - s) * 0.75),
        e,
    ]


def extract_features(image_paths):
    """
    Trích xuất vector đặc trưng (feature vectors) từ danh sách các ảnh bằng DINOv2.
    """
    feats = []
    for path in image_paths:
        try:
            with Image.open(path).convert("RGB") as img:
                tens = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = model(tens)
                feats.append(feat.squeeze().cpu().numpy())
        except Exception as ee:
            print(f"Lỗi khi đọc và xử lý ảnh {path}: {ee}")
    if not feats:
        return np.zeros((0, 384))  # Kích thước embedding của dinov2_vits14 là 384
    return np.vstack(feats)


def deduplicate_frames_in_folder(folder_path, threshold=0.9):
    """
    Loại bỏ các khung hình gần như trùng lặp trong một thư mục.
    Sử dụng cosine similarity trên vector đặc trưng DINOv2 để đo độ tương đồng.
    """
    image_files = sorted(
        [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
    )
    if len(image_files) <= 1:
        return

    print(
        f"Bắt đầu trích xuất đặc trưng cho {len(image_files)} ảnh trong: {folder_path}"
    )
    features = extract_features(image_files)
    if features.shape[0] <= 1:
        return

    print("🔍 Tính toán ma trận tương đồng cosine...")
    sim_matrix = cosine_similarity(features)
    to_delete = set()
    n = len(image_files)
    for i in range(n):
        if i in to_delete:
            continue
        for j in range(i + 1, n):
            if j in to_delete:
                continue
            if sim_matrix[i, j] >= threshold:
                to_delete.add(j)

    for idx in sorted(
        list(to_delete), reverse=True
    ):  # Xóa từ cuối để không ảnh hưởng chỉ số
        f = image_files[idx]
        try:
            os.remove(f)
            # print(f"🗑️ Đã xóa ảnh trùng lặp: {os.path.basename(f)}")
        except Exception as ee:
            print(f"⚠️ Lỗi không xóa được file {f}: {ee}")
    print(f"✅ Đã xóa {len(to_delete)} ảnh trùng lặp.")


def run_deduplication_pipeline(parent_folder, threshold=0.9):
    """
    Thực thi quy trình deduplication cho tất cả các thư mục con (mỗi thư mục của một video).
    """
    if not USE_DEDUP:
        print("⏭️ Bỏ qua bước deduplication do mô hình DINOv2 không khả dụng.")
        return
    video_folders = sorted(
        [
            os.path.join(parent_folder, d)
            for d in os.listdir(parent_folder)
            if os.path.isdir(os.path.join(parent_folder, d))
        ]
    )
    for folder in tqdm(video_folders, desc="🎞️ Đang loại bỏ trùng lặp các video"):
        deduplicate_frames_in_folder(folder, threshold=threshold)


def extract_keyframes_pyav(video_path, target_indices, out_dir):
    """
    Sử dụng PyAV để trích xuất hiệu quả các khung hình tại các chỉ số cụ thể.
    PyAV được chọn vì hỗ trợ codec tốt và hiệu năng cao.
    """
    os.makedirs(out_dir, exist_ok=True)
    target_set = set(int(x) for x in target_indices)
    if not target_set:
        return 0

    max_target = max(target_set)
    saved_count = 0
    frame_idx = 0

    try:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            # Tối ưu hóa việc decode nếu có thể
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                if frame_idx in target_set:
                    img = frame.to_ndarray(format="bgr24")
                    filename = os.path.join(out_dir, f"{frame_idx:06d}.jpg")
                    cv2.imwrite(filename, img)
                    saved_count += 1
                if frame_idx >= max_target:
                    # Tối ưu: dừng sớm khi đã lấy đủ các frame cần thiết
                    break
                frame_idx += 1
    except Exception as e:
        print(f"❌ Lỗi khi xử lý video ({video_path}) bằng PyAV: {e}")
        return 0
    return saved_count


# ================== PHẦN 5: VÒNG LẶP XỬ LÝ CHÍNH ==================
for data_part in data_parts:
    video_dir = os.path.join(videos_dir, f"Videos_{data_part}", "video")
    json_dir = os.path.join(scene_json_dirs, data_part)
    save_dir = os.path.join(save_dir_all, f"{data_part}_extract")
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.isdir(video_dir) or not os.path.isdir(json_dir):
        print(
            f"⚠️ Thư mục video hoặc JSON không tồn tại cho phần '{data_part}'. Bỏ qua."
        )
        continue

    video_files = sorted(
        [f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")]
    )
    video_ids = [os.path.splitext(vp)[0] for vp in video_files]

    print(
        f"\n🔎 Tìm thấy {len(video_ids)} video trong phần {data_part}. Bắt đầu xử lý..."
    )

    for video_id, video_file in tqdm(
        list(zip(video_ids, video_files)), desc=f"🎬 Đang trích xuất {data_part}"
    ):
        video_path = os.path.join(video_dir, video_file)
        # Ánh xạ tên file video sang tên file JSON (tùy theo quy ước đặt tên)
        json_id = video_id.split("_")[-1]
        video_scene_path = os.path.join(json_dir, f"{json_id}.json")

        if not os.path.exists(video_scene_path):
            print(f"⚠️ Không tìm thấy file JSON phân cảnh: {video_scene_path}")
            continue

        try:
            with open(video_scene_path, "r") as f:
                video_scenes = json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠️ Lỗi đọc file JSON {video_scene_path}: {e}")
            continue

        out_folder = os.path.join(save_dir, video_id)

        # Tổng hợp tất cả các chỉ số khung hình cần trích xuất từ các cảnh
        target_indices = []
        for shot in video_scenes:
            if isinstance(shot, (list, tuple)) and len(shot) >= 2:
                target_indices.extend(sample_frame_from_shot(shot[0], shot[1]))
        target_indices = sorted(list(set(target_indices)))

        # Thực hiện trích xuất
        extract_keyframes_pyav(video_path, target_indices, out_folder)

    # Sau khi trích xuất xong cho toàn bộ một phần, chạy pipeline deduplication
    print(f"✨ Hoàn tất trích xuất cho phần {data_part}. Bắt đầu loại bỏ trùng lặp...")
    run_deduplication_pipeline(save_dir, threshold=0.9)

print("\n✅ TẤT CẢ CÁC TÁC VỤ ĐÃ HOÀN THÀNH.")
