#!/bin/bash

IMAGE_DATASETS=("cc12m""ai2d" "chartqa" "mathvision" "coco_train2017" "gqa" "ocr_vqa" "redcaps12m" "textvqa" "vg")

for IMAGE_DATASET in "${IMAGE_DATASETS[@]}"
do
    autofaiss build_index \
        --embeddings="embeddings_folder/${IMAGE_DATASET}/img_emb" \
        --index_path="index_folder/${IMAGE_DATASET}/knn.index" \
        --index_infos_path="index_folder/${IMAGE_DATASET}/infos.json" \
        --metric_type="ip" \
        --max_index_query_time_ms=10 \
        --max_index_memory_usage="16GB"
done