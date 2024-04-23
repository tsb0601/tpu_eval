import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import multiprocessing as mp
import pandas as pd
import os
import json
from tqdm import tqdm
import shortuuid

from PIL import Image
from io import BytesIO
import base64
import math

import model_eval.model_interface as model_interface

def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))

all_options = ['A', 'B', 'C', 'D']

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def setup(rank, world_size, port=12355):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def concatenate_outputs(world_size, base_filename):
    with open(base_filename, 'w') as outfile:
        for rank in range(world_size):
            chunk_filename = '.'.join(base_filename.split('.')[:-1]) + f'_chunk{world_size}-{rank}.' + base_filename.split('.')[-1]
            with open(chunk_filename, 'r') as infile:
                for line in infile:
                    outfile.write(line)


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


def evaluate_model(rank, world_size, config):
    setup(rank, world_size, config["port"])
    torch.cuda.set_device(rank)
    # Model
    model = getattr(model_interface, config['model_module'])(config['model_path'], device_map="cuda:{}".format(rank))
    # model = getattr(model_interface, config['model_module'])(config['model_path'], device_map="auto")
    questions = pd.read_table(os.path.expanduser(config['question_file']))
    questions = get_chunk(questions, world_size, rank)
    answers_file = os.path.expanduser(config['answers_file'])
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    answers_file_chunk = '.'.join(answers_file.split('.')[:-1])+'_chunk{}-{}.'.format(world_size, rank)+answers_file.split('.')[-1]
    ans_file = open(answers_file_chunk, "w")

    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if config['all_rounds']:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            image = load_image_from_base64(row['image'])
            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question
            
            if config['single_pred_prompt']:
                if config['lang'] == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                else:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

            
            outputs = model.inference(image, qs, temperature=config['temperature'], top_p=config['top_p'], num_beams=config['num_beams'], max_length=config['max_length'])

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "round_id": round_idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "options": options,
                                    "option_char": cur_option_char,
                                    "answer_id": ans_id,
                                    "model_id": config['model_path'],
                                    "metadata": {}}) + "\n")
            ans_file.flush()

            # rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]
    ans_file.close()

def main(config):
    world_size = config['world_size']
    
    mp.set_start_method('spawn', force=True)

    processes = []
    for rank in range(world_size):
        p = mp.Process(target=evaluate_model, args=(rank, world_size, config))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # combine the output chunks
    concatenate_outputs(world_size, config['answers_file'])