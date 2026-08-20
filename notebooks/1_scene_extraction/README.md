# 1. Scenes segmentation
Using notebook `scene_segmentation.ipynb` to segment the video into scenes.

Scenes extracted in this link: [Kaggle](https://www.kaggle.com/datasets/danielway17/scenes-segment)

# 2. Keyframes extraction
Using notebook `keyframes_extraction.ipynb` to extract keyframes from the segmented scenes.
---
- Input directory structure:
    ```bash
    |- news-event-retrieval-video-data 
        |- Videos_L21_a
        |- Videos_L22_a
        |- ...
    ```
- Output directory structure: 
    ```bash
    |- SceneJSON 
        |- L01
            |- V001.json # list of scenes in L01, ex: [[0, 120], [121, 240], ...]
            |- V002.json
            |- ...
        |- L02
        |- ...
    |- Keyframes
        |- L01_extra
        |- L02_extra
        |- ...
    ```
