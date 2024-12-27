#!/bin/bash

wget https://storage.googleapis.com/conceptual_12m/cc12m.tsv

sed -i '1s/^/url\tcaption\n/' cc12m.tsv

img2dataset --url_list cc12m.tsv --input_format "tsv" \
         --url_col "url" --caption_col "caption" --output_format webdataset \
           --output_folder cc12m --processes_count 16 --thread_count 64 --image_size 256 \
             --enable_wandb False