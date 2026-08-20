# 1. OCR
---
Extracting text from video frames using Easy OCR.

Input:
```bash
./Keyframes/
└── L01_extra/
    └── 001/
        ├── 000000.jpg  (Ảnh này chứa dòng chữ "BẢN TIN BUỔI SÁNG")
        ├── 000099.jpg  (Ảnh này không có chữ)
        ├── 000199.jpg  (Ảnh này có dòng chữ chạy ở dưới: "Thị trường chứng khoán biến động")
        └── 000299.jpg  (Ảnh này có hai dòng chữ: "MC: Thu Hoài" và "Biên tập: Minh Anh")
    ├── 002/
        ├── 000000.jpg
        ├── 000099.jpg
        └── ...
```

Output:
```bash
./ocr/
└── L01_extra/
    ├── 001.json
    ├── 002.json
    └── ...
```
The contents inside `001.json` as will be follow:
```json
[
    [
        "BẢN TIN BUỔI SÁNG"
    ],
    [],
    [
        "Thị trường chứng khoán biến động"
    ],
    [
        "MC: Thu Hoài",
        "Biên tập: Minh Anh"
    ]
]
```

# 2. Metadata Extraction
---
## 2.1 Object Detection

Object detection on video frames using YOLOv8 pre-trained on Open COCO dataset.
Spatial Locality Encoding is used to enhance the detection results.
Divide the image using 7x7 grid lines, the detected objects will be calculated by the intersection of the bbox and the grid_bbox.
This not only helps identify the subject but also determines where the subject is in the frame.
Also extracts the number of classes and the types of classes.

Input: Keyframes extracted


## 2.2 Colors encode

Encode colors in the video frames using grid-based encoding.
Similar to object detection, the image is divided into a 7x7 grid.
The colors in each grid cell are encoded, providing a spatial representation of the color distribution in the frame.

Input: Keyframes extracted

## 2.3. Tag
Using RAM (Reccognize Anything Model) for tagging objects in the video frames.


Input: Keyframes extracted


---
### Output:
```bash
./context_encoded/
└── bboxes_encoded/
    └── L21_a/
        ├── V001.txt
        ├──...
└── classes_encoded/
    └── L21_a/
        ├── V001.txt
        ├──...
└── number_encoded/
    └── L21_a/
        ├── V001.txt
        ├──...
└── tags_encoded/
    └── L21_a/
        ├── V001.txt
        ├──...
```