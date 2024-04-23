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

from datasets import load_dataset, concatenate_datasets

import model_eval.model_interface as model_interface
from model_eval.MMMU_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG, parse_multi_choice_response, parse_open_response

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
    save_json(base_filename, output_samples)
    
def evaluate_model(rank, world_size, config):
    setup(rank, world_size, config["port"])
    torch.cuda.set_device(rank)
    
    # Model
    model = getattr(model_interface, config['model_module'])(config['model_path'], device_map="cuda:{}".format(rank))
    # model = getattr(model_interface, config['model_module'])(config['model_path'], device_map="auto")
    # run for each subject
    sub_dataset_list = []
    for subject in CAT_SHORT2LONG.values():
        sub_dataset = load_dataset(config['data_path'], subject, split=config['split'])
        sub_dataset_list.append(sub_dataset)

    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)
    questions = [process_single_sample(sample) for sample in dataset]

    questions = get_chunk(questions, world_size, rank)

    answers_file = os.path.expanduser(config['answers_file'])
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    answers_file_chunk = '.'.join(answers_file.split('.')[:-1])+'_chunk{}-{}.'.format(world_size, rank)+answers_file.split('.')[-1]
    out_samples = dict()
    for i, sample in enumerate(tqdm(questions)):
        sample = construct_prompt(sample, config)

        qs = sample["final_input_prompt"]
        image = sample['image'].convert('RGB')
        idx = sample['id']

        outputs = model.inference(image, qs, temperature=config['temperature'], top_p=config['top_p'], num_beams=config['num_beams'], max_length=config['max_length'])

        if sample['question_type'] == 'multiple-choice':
                pred_ans = parse_multi_choice_response(outputs, sample['all_choices'], sample['index2ans'])
        else:  # open question
            pred_ans = outputs
        out_samples[sample['id']] = pred_ans

    save_json(answers_file_chunk, out_samples)


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