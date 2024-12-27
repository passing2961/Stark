#!/bin/bash


clip-retrieval inference \
    --input_dataset "data/textvqa/train_images" \
    --output_folder embeddings_folder/textvqa \
    --clip_model ViT-L/14@336px

clip-retrieval inference \
    --input_dataset "data/vg/VG_100K" \
    --output_folder embeddings_folder/vg \
    --clip_model ViT-L/14@336px

clip-retrieval inference \
    --input_dataset "data/ocr_vqa/images" \
    --output_folder embeddings_folder/ocr_vqa \
    --clip_model ViT-L/14@336px

clip-retrieval inference \
    --input_dataset "data/gqa/images" \
    --output_folder embeddings_folder/gqa \
    --clip_model ViT-L/14@336px

clip-retrieval inference \
    --input_dataset "data/coco/train2017" \
    --output_folder embeddings_folder/coco_train2017 \
    --clip_model ViT-L/14@336px

clip-retrieval inference \
    --input_dataset "cc12m/{00000..01242}.tar" \
    --output_folder embeddings_folder/cc12m \
    --input_format webdataset \
    --clip_model ViT-L/14@336px \
    --enable_metadata True \
    --output_partition_count 1243

clip-retrieval inference \
    --input_dataset "data/redcaps/redcaps12m_shards/{00000..00180}.tar" \
    --output_folder embeddings_folder/redcaps12m \
    --input_format webdataset \
    --output_partition_count 181 \
    --clip_model ViT-L/14@336px \
    --enable_metadata True


clip-retrieval inference \
    --input_dataset "data/ai2d/images" \
    --output_folder embeddings_folder/ai2d \
    --clip_model ViT-L/14@336px

clip-retrieval inference \
    --input_dataset "data/ChartQA Dataset/train/png" \
    --output_folder embeddings_folder/chartqa \
    --clip_model ViT-L/14@336px

clip-retrieval inference \
    --input_dataset "data/mathvision/images" \
    --output_folder embeddings_folder/mathvision \
    --clip_model ViT-L/14@336px