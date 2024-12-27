import os
import random
import argparse

from runner import (
    PersonaRunner, 
    CommonsenseRunner,
    NarrativeRunner,
    AlignmentRunner,
    EventRunner,
    DialogueRunner,
    #ImageRunner,
    AlbumRunner,
    AlbumImageRunner,
    FaceRunner,
    FaceImageRunner
)


RUNNER_MAP = {
    'persona-attr': PersonaRunner,
    'commonsense': CommonsenseRunner,
    'narrative': NarrativeRunner,
    'event': EventRunner,
    'dialogue': DialogueRunner,
    'album': AlbumRunner,
    'face': FaceRunner,
}

def main(args):
    random.seed(42)

    runner = RUNNER_MAP[args.runner_name](args)
    runner.run()    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for generating multi-modal dialogues using LLM')
    parser.add_argument('--run-id',
                        type=str,
                        default='vanilla',
                        help='the name of the directory where the output will be dumped')
    parser.add_argument('--model',
                        type=str,
                        default='gpt-3.5-turbo-1106',
                        help='which LLM to use')
    parser.add_argument('--temperature',
                        type=float,
                        default=0.9,
                        help="control randomness: lowering results in less random completion")
    parser.add_argument('--top-p',
                        type=float,
                        default=0.95,
                        help="nucleus sampling")
    parser.add_argument('--frequency-penalty',
                        type=float,
                        default=1.0,
                        help="decreases the model's likelihood to repeat the same line verbatim")
    parser.add_argument('--presence-penalty',
                        type=float,
                        default=0.6,
                        help="increases the model's likelihood to talk about new topics")
    parser.add_argument('--max-tokens',
                        type=int,
                        default=1024,
                        help='maximum number of tokens to generate')
    parser.add_argument('--split',
                        type=str,
                        default=None,
                        help='Specify the dataset split (i.e., train, validation, test).')
    parser.add_argument('--runner-name',
                        type=str,
                        default=None,
                        help='Specify the runner name (e.g., persona-attribute)')
    parser.add_argument('--do-parse-filter',
                        action='store_true',
                        help='do parsing and filtering based on llm-generated results')
    parser.add_argument('--diffusion-model-id',
                        type=str,
                        default=None,
                        help='Specify the diffusion model.')
    parser.add_argument('--cache-dir',
                        type=str,
                        default=None,
                        help='Cache dir for downloading pre-trained diffusion model.')
    parser.add_argument('--debug',
                        action='store_true',
                        help='do debugging for generating small number of sampels.')
    parser.add_argument('--debug-sample-num',
                        type=int,
                        default=None,
                        help="Number of sample for debug.")
    parser.add_argument('--shard-num',
                        type=int,
                        default=200,
                        help='Number of sharded files.')
    parser.add_argument('--persona-seed-num',
                        type=int,
                        default=None,
                        help="Persona seed number.")
    parser.add_argument('--target-session-num',
                        type=int,
                        default=None,
                        help="Target dialogue session number.")
    args = parser.parse_args()
    main(args)