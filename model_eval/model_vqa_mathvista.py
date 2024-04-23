import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import multiprocessing as mp
import os
import json
from tqdm import tqdm
import shortuuid

from PIL import Image
import math

from datasets import load_dataset

import model_eval.model_interface as model_interface

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
    output_samples = dict()
    for rank in range(world_size):
        chunk_filename = '.'.join(base_filename.split('.')[:-1]) + f'_chunk{world_size}-{rank}.' + base_filename.split('.')[-1]
        output_samples_chunk = json.load(open(chunk_filename))
        for k,v in output_samples_chunk.items():
            output_samples[k] = v
    with open(base_filename, 'w') as f:
        json.dump(output_samples, f)
    
def evaluate_model(rank, world_size, config):
    setup(rank, world_size, config["port"])
    torch.cuda.set_device(rank)
    
    # Model
    model = getattr(model_interface, config['model_module'])(config['model_path'], device_map="cuda:{}".format(rank))
    # model = getattr(model_interface, config['model_module'])(config['model_path'], device_map="auto")
    dataset = load_dataset(config['data_path'])

    questions = [sample for sample in dataset[config['split']]]

    questions = get_chunk(questions, world_size, rank)

    answers_file = os.path.expanduser(config['answers_file'])
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    answers_file_chunk = '.'.join(answers_file.split('.')[:-1])+'_chunk{}-{}.'.format(world_size, rank)+answers_file.split('.')[-1]
    out_samples = dict()
    for i, sample in enumerate(tqdm(questions)):

        qs = sample["query"]
        image_file = sample['image']
        idx = sample['pid']

        image = Image.open(os.path.join(config['image_folder'], image_file)).convert('RGB')

        outputs = model.inference(image, qs, temperature=config['temperature'], top_p=config['top_p'], num_beams=config['num_beams'], max_length=config['max_length'])
        
        sample['prediction'] = outputs

        del sample['decoded_image']
        out_samples[idx] = sample

    with open(answers_file_chunk, 'w') as f:
        json.dump(out_samples, f)


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