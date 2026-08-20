# 1. Audio extraction
---
Using the following notebook `audio_extraction.ipynb` to extract audio from videos and perform audio-related tasks.
Output in .wav file format.
```bash
./audio/
└── L21_a/
    ├── V001.wav
    ├── V002.wav
    └── ...
├── L22_a/
    ├── V001.wav
    ├── V002.wav
    └── ...
```

Data extracted in this link: [Kaggle](https://www.kaggle.com/datasets/danielway17/audio-extracted-data)
# 2. Audio detection
---
- Detecting human voices, speech, etc.
- Using VAD (Voice Activity Detection) to filter out non-speech segments.

Input: `.wav` files from the audio extraction step.

Output:
```bash
./audio_detection/
└── L01_extra/
    ├── V001.json
    └── V002.json
```
The contents of the V001.json file will be following:
```json
[
    [2.5134, 8.9521], # segment of voice detected
    [10.2005, 15.6788],
    [22.1, 23.504]
]
```

# 3. Audio recognition
---
- Using Model supported Vietnamese ASR (Automatic Speech Recognition) to transcribe audio like PhoWhisper, Whisper-Large-V3, etc.

Input: includes `.wav` files from the audio extraction step and `.json` files from the audio detection step.

Output:
```bash
./audio/
└── L01_extra/
    └── V001.json
```
The contents of the V001.json file will be following:
```json
[
    "xin chào các bạn hôm nay chúng ta sẽ cùng tìm hiểu về", #  coressponding to the first segment of voice detected
    "trí tuệ nhân tạo là một lĩnh vực rất thú vị nó bao gồm học máy và học sâu"
]
```
