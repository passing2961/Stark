#!/bin/bash


# python generate_sonny_dataset.py --run-id sonny_v1 \
#    --diffusion-model-id jialuliluka/selma-xl \
#    --runner-name image \
#    --cache-dir pretrained_diffusion_model \
#    --model gpt-3.5-turbo-0125	\
#    --persona-seed-num demo

#stabilityai/stable-diffusion-xl-base-1.0 \

# python generate_sonny_dataset.py --run-id sonny_v4 \
#    --model gpt-3.5-turbo-0125	\
#    --temperature 0.9 \
#    --top-p 1.0 \
#    --frequency-penalty .0 \
#    --presence-penalty 0. \
#    --max-tokens 4096 \
#    --runner-name dialogue \
#    --persona-seed-num 0 \
#    --do-parse-filter

   # --debug \
   #--debug-sample-num 10 \
   #--shard-num 1 \
   
# python generate_sonny_dataset.py --run-id sonny_v4 \
#    --model gpt-3.5-turbo-0125	\
#    --temperature 0.9 \
#    --top-p 1.0 \
#    --frequency-penalty 0.4 \
#    --presence-penalty 0.4 \
#    --max-tokens 1024 \
#    --runner-name face-image \
#    --debug \
#    --debug-sample-num 100 \
#    --shard-num 1 \
#    --persona-seed-num 10 \
   #--do-parse-filter

# python generate_sonny_dataset.py --run-id sonny_v2 \
#    --diffusion-model-id jialuliluka/selma-xl \
#    --runner-name album-image \
#    --cache-dir pretrained_diffusion_model \
#    --model gpt-3.5-turbo-0125 \
#    --do-parse-filter









# python generate_stark_dialogue.py --run-id stark_v1 \
#     --model gpt-3.5-turbo-0125	\
#     --temperature 0.9 \
#     --top-p 1.0 \
#     --frequency-penalty 0.0 \
#     --presence-penalty 0.0 \
#     --max-tokens 2048 \
#     --runner-name persona-attr \
#     --shard-num 1 \
#     --debug \
#     --debug-sample-num 10 
    
# python generate_stark_dialogue.py --run-id stark_v1 \
#     --model gpt-3.5-turbo-0125	\
#     --temperature 0.9 \
#     --top-p 1.0 \
#     --frequency-penalty 0.0 \
#     --presence-penalty 0.0 \
#     --max-tokens 2048 \
#     --runner-name persona-attr \
#     --shard-num 1 \
#     --do-parse-filter
    

# python generate_stark_dialogue.py --run-id stark_v1 \
#     --model gpt-3.5-turbo-0125	\
#     --temperature 0.9 \
#     --top-p 1.0 \
#     --frequency-penalty 0.0 \
#     --presence-penalty 0.0 \
#     --max-tokens 2048 \
#     --runner-name face \
#     --persona-seed-num 0 \
#     --shard-num 1

# python generate_stark_dialogue.py --run-id stark_v1 \
#     --model gpt-3.5-turbo-0125	\
#     --temperature 0.9 \
#     --top-p 1.0 \
#     --frequency-penalty 0.0 \
#     --presence-penalty 0.0 \
#     --max-tokens 2048 \
#     --runner-name face \
#     --persona-seed-num 0 \
#     --shard-num 1 \
#     --do-parse-filter



# python generate_stark_dialogue.py --run-id stark_v1 \
#    --model gpt-3.5-turbo-0125	\
#    --temperature 0.9 \
#    --top-p 1.0 \
#    --frequency-penalty .0 \
#    --presence-penalty .0 \
#    --max-tokens 1024 \
#    --runner-name commonsense \
#    --persona-seed-num 0 \
#    --debug \
#    --debug-sample-num 20

# python generate_stark_dialogue.py --run-id stark_v1 \
#    --model gpt-3.5-turbo-0125	\
#    --temperature 0.9 \
#    --top-p 1.0 \
#    --frequency-penalty .0 \
#    --presence-penalty .0 \
#    --max-tokens 1024 \
#    --runner-name commonsense \
#    --persona-seed-num 0 \
#    --do-parse-filter




# python generate_stark_dialogue.py --run-id stark_v1 \
#    --model gpt-3.5-turbo-0125 \
#    --temperature 0.9 \
#    --top-p 0.95 \
#    --frequency-penalty 1.0 \
#    --presence-penalty 0.6 \
#    --max-tokens 2048 \
#    --runner-name narrative \
#    --debug \
#    --debug-sample-num 20 \
#    --persona-seed-num 0

# python generate_stark_dialogue.py --run-id stark_v1 \
#    --model gpt-3.5-turbo-0125 \
#    --temperature 0.9 \
#    --top-p 0.95 \
#    --frequency-penalty 1.0 \
#    --presence-penalty 0.6 \
#    --max-tokens 2048 \
#    --runner-name narrative \
#    --debug \
#    --debug-sample-num 20 \
#    --persona-seed-num 0 \
#    --do-parse-filter



# python generate_stark_dialogue.py --run-id stark_v1 \
#    --model gpt-3.5-turbo-0125 \
#    --temperature 0.9 \
#    --top-p 1.0 \
#    --frequency-penalty 0. \
#    --presence-penalty 0. \
#    --max-tokens 4096 \
#    --runner-name event \
#    --debug \
#    --debug-sample-num 20 \
#    --persona-seed-num 0

# python generate_stark_dialogue.py --run-id stark_v1 \
#    --model gpt-3.5-turbo-0125 \
#    --temperature 0.9 \
#    --top-p 1.0 \
#    --frequency-penalty 0. \
#    --presence-penalty 0. \
#    --max-tokens 4096 \
#    --runner-name event \
#    --debug \
#    --debug-sample-num 20 \
#    --persona-seed-num 0 \
#    --do-parse-filter




# python generate_stark_dialogue.py --run-id stark_v1 \
#     --model gpt-3.5-turbo-0125	\
#     --temperature 0.9 \
#     --top-p 1.0 \
#     --frequency-penalty 0.0 \
#     --presence-penalty 0.0 \
#     --max-tokens 1024 \
#     --runner-name album \
#     --debug \
#     --debug-sample-num 20 \
#     --persona-seed-num 0 

# python generate_stark_dialogue.py --run-id stark_v1 \
#     --model gpt-3.5-turbo-0125	\
#     --temperature 0.9 \
#     --top-p 1.0 \
#     --frequency-penalty 0.0 \
#     --presence-penalty 0.0 \
#     --max-tokens 1024 \
#     --runner-name album \
#     --debug \
#     --debug-sample-num 20 \
#     --persona-seed-num 0 \
#     --do-parse-filter


# for session_num in {1..6}; do
#    python generate_stark_dialogue.py --run-id stark_v1 \
#       --model gpt-3.5-turbo-0125	\
#       --temperature 0.9 \
#       --top-p 1.0 \
#       --frequency-penalty 0.0 \
#       --presence-penalty 0.0 \
#       --max-tokens 4096 \
#       --runner-name dialogue \
#       --debug \
#       --debug-sample-num 20 \
#       --persona-seed-num 0 \
#       --target-session-num "$session_num" \
      
#    python generate_stark_dialogue.py --run-id stark_v1 \
#       --model gpt-3.5-turbo-0125	\
#       --temperature 0.9 \
#       --top-p 1.0 \
#       --frequency-penalty 0.0 \
#       --presence-penalty 0.0 \
#       --max-tokens 4096 \
#       --runner-name dialogue \
#       --debug \
#       --debug-sample-num 20 \
#       --persona-seed-num 0 \
#       --target-session-num "$session_num" \
#       --do-parse-filter
# done

# python make_final_dataset.py

# python postprocess_final_dataset.py

# python generate_face_image.py \
#     --start-idx 0 \
#     --end-idx 1 \
#     --device cuda:0

# python plan_runner.py \
#    --start-idx 0 \
#    --end-idx 1

# python plan_runner.py \
#    --start-idx 0 \
#    --end-idx 1 \
#    --do-planner

# python execute_photomaker.py \
#    --start-idx 0 \
#    --end-idx 1 

# python execute_sdxl.py \
#    --start-idx 0 \
#    --end-idx 1 

# python execute_retrieval.py \
#    --start-idx 0 \
#    --end-idx 1 

# python execute_web_search.py \
#    --start-idx 0 \
#    --end-idx 1 