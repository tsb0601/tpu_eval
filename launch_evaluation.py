import json
import os
import subprocess
import copy
import torch
import numpy as np
import random


from model_eval.model_vqa_eval import main as eval_main
from model_eval.model_vqa_science import main as eval_main_science
from model_eval.model_vqa_mmbench import main as eval_main_mmbench
from model_eval.model_vqa_MMMU import main as eval_main_MMMU
from model_eval.model_vqa_mathvista import main as eval_main_mathvista
from model_eval.model_vqa_realworldqa import main as eval_main_realworldqa
from model_eval.model_vqa_mmstar import main as eval_main_mmstar
from postprocess_scripts.convert_mmbench_for_submission import convert_mmbench_for_submission


"""TODO:
- extract the performance metrics (or upload file paths) from the evaluation scripts, collect them all in a single place, and save them to a file
    - maybe we can write to a file incrementally. and then skip the evaluation if the file already exists, to allow resuming the evaluation
- save the files that need to be uploaded with the benchmark in the title to avoid dupes
"""


def print_green(text):
    print(f"\033[32m{text}\033[0m")

def print_red(text):
    print(f"\033[31m{text}\033[0m")

def create_config(config_all, benchmark_name):
    config = copy.deepcopy(config_all['benchmarks'][benchmark_name])
    config['model_module'] = config_all['model_module']
    config['model_path'] = config_all['model_path']
    config['data_folder'] = config_all['data_folder']
    config['output_name'] = config_all['output_name']
    config['world_size'] = config_all['world_size']
    config['port'] = config_all['port']
    
    config['temperature'] = config.get('temperature', config_all['temperature'])
    config['top_p'] = config.get('top_p', config_all['top_p'])
    config['num_beams'] = config.get('num_beams', config_all['num_beams'])
    config['max_length'] = config.get('max_length', config_all['max_length'])
    return config

def run_mmbench(config):
    answers_file = './results/mmbench/answers/{}/{}.jsonl'.format(config['split'], config['output_name'])
    config['answers_file'] = answers_file
    config['question_file'] = os.path.join(config['data_folder'], "mmbench/{}.tsv".format(config['split']))  
    eval_main_mmbench(config)
    # result = subprocess.run(['bash', 'postprocess_scripts/mmbench.sh', config['data_folder'], config['answers_file'], config['output_name'], config['split']], stdout=None, stderr=None)
    """
    python3 postprocess_scripts/convert_mmbench_for_submission.py \
    --annotation-file $1/mmbench/$4.tsv \
    --result-dir ./results/mmbench/answers/$4 \
    --upload-dir ./results/mmbench/answers_upload/$4 \
    --experiment $3
    """
    anno_file = os.path.join(config['data_folder'], f"mmbench/{config['split']}.tsv")
    result_dir = f"./results/mmbench/answers/{config['split']}"
    upload_dir = f"./results/mmbench/answers_upload/{config['split']}"
    experiment = config['output_name']
    xlsx_path = convert_mmbench_for_submission(anno_file, result_dir, upload_dir, experiment)
    print_green(f"Please submit the results at {xlsx_path} to the server at https://mmbench.opencompass.org.cn/mmbench-submission")


def run_mmbench_cn(config):
    answers_file = './results/mmbench/answers/{}/{}.jsonl'.format(config['split'], config['output_name'])
    config['answers_file'] = answers_file
    config['question_file'] = os.path.join(config['data_folder'], "mmbench/{}.tsv".format(config['split']))  
    eval_main_mmbench(config)
    result = subprocess.run(['bash', 'postprocess_scripts/mmbench_cn.sh', config['data_folder'], config['answers_file'], config['output_name'], config['split']], stdout=None, stderr=None)


def run_vqav2(config):
    answers_file = './results/vqav2/answers/{}/{}.jsonl'.format(config['split'], config['output_name'])
    config['answers_file'] = answers_file
    config['question_file'] = os.path.join(config['data_folder'], "vqav2/{}.jsonl".format(config['split'])) 
    config['image_folder'] = os.path.join(config['data_folder'], "vqav2/test2015") 
    eval_main(config)
    result = subprocess.run(['bash', 'postprocess_scripts/vqav2.sh', config['data_folder'], config['answers_file'], config['output_name'], config['split']], stdout=None, stderr=None)

def run_gqa(config):
    answers_file = './results/gqa/answers/{}.jsonl'.format(config['output_name'])
    config['answers_file'] = answers_file
    config['question_file'] = os.path.join(config['data_folder'], "gqa/llava_gqa_testdev_balanced.jsonl") 
    config['image_folder'] = os.path.join(config['data_folder'], "gqa/data/images") 
    eval_main(config)
    result = subprocess.run(['bash', 'postprocess_scripts/gqa.sh', config['data_folder'], config['answers_file'], config['output_name']], stdout=None, stderr=None)

def run_vizwiz(config):
    answers_file = './results/vizwiz/answers/{}.jsonl'.format(config['output_name'])
    config['answers_file'] = answers_file
    config['question_file'] = os.path.join(config['data_folder'], "vizwiz/llava_test.jsonl") 
    config['image_folder'] = os.path.join(config['data_folder'], "vizwiz/test") 
    eval_main(config)
    result = subprocess.run(['bash', 'postprocess_scripts/vizwiz.sh', config['data_folder'], config['answers_file'], config['output_name']], stdout=None, stderr=None)

def run_scienceqa(config):
    answers_file = './results/scienceqa/answers/{}.jsonl'.format(config['output_name'])
    config['answers_file'] = answers_file
    config['question_file'] = os.path.join(config['data_folder'], "scienceqa/llava_test_CQM-A.json") 
    config['image_folder'] = os.path.join(config['data_folder'], "scienceqa/images/test") 
    eval_main_science(config)
    result = subprocess.run(['bash', 'postprocess_scripts/sqa.sh', config['data_folder'], config['answers_file'], config['output_name']], stdout=None, stderr=None)

def run_textvqa(config):
    answers_file = './results/textvqa/answers/{}.jsonl'.format(config['output_name'])
    config['answers_file'] = answers_file
    config['question_file'] = os.path.join(config['data_folder'], "textvqa/llava_textvqa_val_v051_ocr.jsonl") 
    config['image_folder'] = os.path.join(config['data_folder'], "textvqa/train_images") 
    eval_main(config)
    result = subprocess.run(['bash', 'postprocess_scripts/textvqa.sh', config['data_folder'], config['answers_file'], config['output_name']], stdout=None, stderr=None)

def run_pope(config):
    answers_file = './results/pope/answers/{}.jsonl'.format(config['output_name'])
    config['answers_file'] = answers_file
    config['question_file'] = os.path.join(config['data_folder'], "pope/llava_pope_test.jsonl") 
    config['image_folder'] = os.path.join(config['data_folder'], "pope/val2014") 
    eval_main(config)
    result = subprocess.run(['bash', 'postprocess_scripts/pope.sh', config['data_folder'], config['answers_file'], config['output_name']], stdout=None, stderr=None)

def run_mme(config):
    answers_file = './results/MME/answers/{}.jsonl'.format(config['output_name'])
    config['answers_file'] = answers_file
    config['question_file'] = os.path.join(config['data_folder'], "MME/llava_mme.jsonl") 
    config['image_folder'] = os.path.join(config['data_folder'], "MME/MME_Benchmark_release_version") 
    eval_main(config)
    result = subprocess.run(['bash', 'postprocess_scripts/mme.sh', config['data_folder'], config['answers_file'], config['output_name']], stdout=None, stderr=None)

def run_llava_bench(config):
    answers_file = './results/llava-bench-in-the-wild/answers/{}.jsonl'.format(config['output_name'])
    config['answers_file'] = answers_file
    config['question_file'] = os.path.join(config['data_folder'], "llava-bench-in-the-wild/questions.jsonl") 
    config['image_folder'] = os.path.join(config['data_folder'], "llava-bench-in-the-wild/images") 
    eval_main(config)
    result = subprocess.run(['bash', 'postprocess_scripts/llavabench.sh', config['data_folder'], config['answers_file'], config['output_name'], config['openai_api_key']], stdout=None, stderr=None)

def run_mmvet(config):
    answers_file = './results/mm-vet/answers/{}.jsonl'.format(config['output_name'])
    config['answers_file'] = answers_file
    config['question_file'] = os.path.join(config['data_folder'], "mm-vet/llava-mm-vet.jsonl") 
    config['image_folder'] = os.path.join(config['data_folder'], "mm-vet/images") 
    eval_main(config)
    result = subprocess.run(['bash', 'postprocess_scripts/mmvet.sh', config['data_folder'], config['answers_file'], config['output_name']], stdout=None, stderr=None)

def run_seed_bench(config):
    answers_file = './results/seed_bench/answers/{}.jsonl'.format(config['output_name'])
    config['answers_file'] = answers_file
    config['question_file'] = os.path.join(config['data_folder'], "seed_bench/SEED-Bench.jsonl") 
    config['image_folder'] = os.path.join(config['data_folder'], "seed_bench") 
    eval_main(config)
    result = subprocess.run(['bash', 'postprocess_scripts/seed.sh', config['data_folder'], config['answers_file'], config['output_name']], stdout=None, stderr=None)

def run_MMMU(config):
    answers_file = './results/MMMU/answers/{}.json'.format(config['output_name'])
    config['answers_file'] = answers_file
    eval_main_MMMU(config)
    result = subprocess.run(['bash', 'postprocess_scripts/MMMU.sh', config['data_folder'], config['answers_file'], config['output_name']], stdout=None, stderr=None)

def run_realworldqa(config):
    answers_file = './results/realworldqa/answers/{}.json'.format(config['output_name'])
    config['answers_file'] = answers_file
    eval_main_realworldqa(config)
    result = subprocess.run(['bash', 'postprocess_scripts/realworldqa.sh', config['data_folder'], config['answers_file'], config['output_name']], stdout=None, stderr=None)

def run_mmstar(config):
    answers_file = './results/mmstar/answers/{}.json'.format(config['output_name'])
    config['answers_file'] = answers_file
    eval_main_mmstar(config)
    result = subprocess.run(['bash', 'postprocess_scripts/mmstar.sh', config['data_folder'], config['answers_file'], config['output_name']], stdout=None, stderr=None)

def run_chartQA(config):
    answers_file = './results/ChartQA/answers/{}.jsonl'.format(config['output_name'])
    config['answers_file'] = answers_file
    config['question_file'] = os.path.join(config['data_folder'], "ChartQA/test_questions.jsonl") 
    config['image_folder'] = os.path.join(config['data_folder'], "ChartQA/test/png") 
    eval_main(config)
    result = subprocess.run(['bash', 'postprocess_scripts/chartQA.sh', config['data_folder'], config['answers_file'], config['output_name']], stdout=None, stderr=None)

def run_AI2D(config):
    answers_file = './results/AI2D/answers/{}.jsonl'.format(config['output_name'])
    config['answers_file'] = answers_file
    config['question_file'] = os.path.join(config['data_folder'], "AI2D/test_questions.jsonl") 
    config['image_folder'] = os.path.join(config['data_folder'], "AI2D/ai2d/images") 
    eval_main(config)
    result = subprocess.run(['bash', 'postprocess_scripts/AI2D.sh', config['data_folder'], config['answers_file'], config['output_name']], stdout=None, stderr=None)

def run_MMVP(config):
    answers_file = './results/MMVP/answers/{}.jsonl'.format(config['output_name'])
    config['answers_file'] = answers_file
    config['question_file'] = os.path.join(config['data_folder'], "MMVP/test_questions.jsonl") 
    config['image_folder'] = os.path.join(config['data_folder'], "MMVP") 
    eval_main(config)
    result = subprocess.run(['bash', 'postprocess_scripts/MMVP.sh', config['data_folder'], config['answers_file'], config['output_name']], stdout=None, stderr=None)

def run_vstar(config):
    answers_file = './results/vstar/answers/{}.jsonl'.format(config['output_name'])
    config['answers_file'] = answers_file
    config['question_file'] = os.path.join(config['data_folder'], "vstar/test_questions.jsonl") 
    config['image_folder'] = os.path.join(config['data_folder'], "vstar") 
    eval_main(config)
    result = subprocess.run(['bash', 'postprocess_scripts/vstar.sh', config['data_folder'], config['answers_file'], config['output_name']], stdout=None, stderr=None)

def run_mathvista(config):
    answers_file = './results/mathvista/answers/{}.json'.format(config['output_name'])
    config['answers_file'] = answers_file
    config['image_folder'] = os.path.join(config['data_folder'], "mathvista") 
    eval_main_mathvista(config)
    result = subprocess.run(['bash', 'postprocess_scripts/mathvista.sh', config['data_folder'], config['answers_file'], config['output_name'], config['openai_api_key']], stdout=None, stderr=None)

def run_docvqa(config):
    answers_file = './results/docvqa/answers/{}.jsonl'.format(config['output_name'])
    config['answers_file'] = answers_file
    config['question_file'] = os.path.join(config['data_folder'], "DocVQA/test_questions.jsonl") 
    config['image_folder'] = os.path.join(config['data_folder'], "DocVQA") 
    eval_main(config)
    result = subprocess.run(['bash', 'postprocess_scripts/docvqa.sh', config['data_folder'], config['answers_file'], config['output_name']], stdout=None, stderr=None)

def run_infovqa(config):
    answers_file = './results/infovqa/answers/{}.jsonl'.format(config['output_name'])
    config['answers_file'] = answers_file
    config['question_file'] = os.path.join(config['data_folder'], "InfoVQA/test_questions.jsonl") 
    config['image_folder'] = os.path.join(config['data_folder'], "InfoVQA/images") 
    eval_main(config)
    result = subprocess.run(['bash', 'postprocess_scripts/infovqa.sh', config['data_folder'], config['answers_file'], config['output_name']], stdout=None, stderr=None)

def run_ocrbench(config):
    answers_file = './results/ocrbench/answers/{}.jsonl'.format(config['output_name'])
    config['answers_file'] = answers_file
    config['question_file'] = os.path.join(config['data_folder'], "OCRBench/test_questions.jsonl") 
    config['image_folder'] = os.path.join(config['data_folder'], "OCRBench/OCRBench_Images") 
    eval_main(config)
    result = subprocess.run(['bash', 'postprocess_scripts/ocrbench.sh', config['data_folder'], config['answers_file'], config['output_name']], stdout=None, stderr=None)

def run_stvqa(config):
    answers_file = './results/stvqa/answers/{}.jsonl'.format(config['output_name'])
    config['answers_file'] = answers_file
    config['question_file'] = os.path.join(config['data_folder'], "STVQA/test_questions.jsonl") 
    config['image_folder'] = os.path.join(config['data_folder'], "STVQA") 
    eval_main(config)
    result = subprocess.run(['bash', 'postprocess_scripts/stvqa.sh', config['data_folder'], config['answers_file'], config['output_name']], stdout=None, stderr=None)


def main(config: dict, benchmarks: dict, seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    #   torch.use_deterministic_algorithms = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # MMbench
    if benchmarks['mmbench']['eval']:
        print_green("Start the evaluation on MMbench")
        run_mmbench(create_config(config, 'mmbench'))
        print_green("Evaluation on MMbench finished. Please submit the results under {} to the server at {}.".format('results/mmbench/answers_upload/{}'.format(benchmarks['mmbench']['split']), 'https://mmbench.opencompass.org.cn/mmbench-submission'))

    # MMbench_CN
    if benchmarks['mmbench_cn']['eval']:
        print_green("Start the evaluation on MMbench_CN")
        run_mmbench_cn(create_config(config, 'mmbench_cn'))
        print_green("Evaluation on MMbench_CN finished. Please submit the results under {} to the server at {}.".format('results/mmbench/answers_upload/{}'.format(benchmarks['mmbench_cn']['split']), 'https://mmbench.opencompass.org.cn/mmbench-submission'))

    # GQA
    if benchmarks['gqa']['eval']:
        print_green("Start the evaluation on GQA")
        run_gqa(create_config(config, 'gqa'))
        print_green("Evaluation on GQA finished.")

    # VisWiz
    if benchmarks['vizwiz']['eval']:
        print_green("Start the evaluation on VizWiz")
        run_vizwiz(create_config(config, 'vizwiz'))
        print_green("Evaluation on VisWiz finished. Please submit the results under {} to the server at {}.".format('results/vizwiz/answers_upload', 'https://eval.ai/web/challenges/challenge-page/2185/overview'))

    # ScienceQA
    if benchmarks['scienceqa']['eval']:
        print_green("Start the evaluation on ScienceQA")
        run_scienceqa(create_config(config, 'scienceqa'))
        print_green("Evaluation on ScienceQA finished.")

    # TextVQA
    if benchmarks['textvqa']['eval']:
        print_green("Start the evaluation on TextVQA")
        run_textvqa(create_config(config, 'textvqa'))
        print_green("Evaluation on TextVQA finished.")

    # POPE
    if benchmarks['pope']['eval']:
        print_green("Start the evaluation on POPE")
        run_pope(create_config(config, 'pope'))
        print_green("Evaluation on POPE finished.")

    # MME
    if benchmarks['mme']['eval']:
        print_green("Start the evaluation on MME")
        run_mme(create_config(config, 'mme'))
        print_green("Evaluation on MME finished.")

    # LLaVA_Bench
    if benchmarks['llava_bench']['eval']:
        assert len(benchmarks['llava_bench']['openai_api_key']) > 0
        print_green("Start the evaluation on LLaVA_Bench")
        run_llava_bench(create_config(config, 'llava_bench'))
        print_green("Evaluation on LLaVA_Bench finished.")


    # MM-Vet
    if benchmarks['mmvet']['eval']:
        print_green("Start the evaluation on MM-Vet")
        run_mmvet(create_config(config, 'mmvet'))
        print_green("Evaluation on MM-Vet finished. Please submit the results under {} to the server at {}.".format('results/mm-vet/results', 'https://huggingface.co/spaces/whyu/MM-Vet_Evaluator'))

    # SEED-Bench
    if benchmarks['seed_bench']['eval']:
        print_green("Start the evaluation on SEED-Bench")
        run_seed_bench(create_config(config, 'seed_bench'))
        print_green("Evaluation on SEED-Bench finished.")


    # MMMU
    if benchmarks['MMMU']['eval']:
        print_green("Start the evaluation on MMMU")
        run_MMMU(create_config(config, 'MMMU'))
        print_green("Evaluation on MMMU finished.")

    # ChartQA
    if benchmarks['chartQA']['eval']:
        print_green("Start the evaluation on ChartQA")
        run_chartQA(create_config(config, 'chartQA'))
        print_green("Evaluation on ChartQA finished.")

    # AI2D
    if benchmarks['AI2D']['eval']:
        print_green("Start the evaluation on AI2D")
        run_AI2D(create_config(config, 'AI2D'))
        print_green("Evaluation on AI2D finished.")

    # realworldqa
    if benchmarks['realworldqa']['eval']:
        print_green("Start the evaluation on realworldqa")
        run_realworldqa(create_config(config, 'realworldqa'))
        print_green("Evaluation on realworldqa finished.")

    # mmstar
    if benchmarks['mmstar']['eval']:
        print_green("Start the evaluation on mmstar")
        run_mmstar(create_config(config, 'mmstar'))
        print_green("Evaluation on mmstar finished.")

    # docvqa
    if benchmarks['docvqa']['eval']:
        print_green("Start the evaluation on docvqa")
        run_docvqa(create_config(config, 'docvqa'))
        print_green("Evaluation on docvqa finished. Please submit the results under {} to the server at {}.".format('results/docvqa/answers_upload', 'https://rrc.cvc.uab.es/?ch=17&com=mymethods&task=1'))

    # infovqa
    if benchmarks['infovqa']['eval']:
        print_green("Start the evaluation on infovqa")
        run_infovqa(create_config(config, 'infovqa'))
        print_green("Evaluation on infovqa finished. Please submit the results under {} to the server at {}.".format('results/infovqa/answers_upload', 'https://rrc.cvc.uab.es/?ch=17&com=mymethods&task=3'))
        

    # ocrbench
    if benchmarks['ocrbench']['eval']:
        print_green("Start the evaluation on ocrbench")
        run_ocrbench(create_config(config, 'ocrbench'))
        print_green("Evaluation on ocrbench finished.")

    # stvqa
    if benchmarks['stvqa']['eval']:
        print_green("Start the evaluation on stvqa")
        run_stvqa(create_config(config, 'stvqa'))
        print_green("Evaluation on stvqa finished. Please submit the results under {} to the server at {}.".format('results/stvqa/answers_upload', 'https://rrc.cvc.uab.es/?ch=11&com=mymethods&task=3'))


    # MMVP
    if benchmarks['MMVP']['eval']:
        print_green("Start the evaluation on MMVP")
        run_MMVP(create_config(config, 'MMVP'))
        print_green("Evaluation on MMVP finished.")

    # vstar
    if benchmarks['vstar']['eval']:
        print_green("Start the evaluation on vstar")
        run_vstar(create_config(config, 'vstar'))
        print_green("Evaluation on vstar finished.")

    # VQAv2
    if benchmarks['vqav2']['eval']:
        print_green("Start the evaluation on VQAv2")
        run_vqav2(create_config(config, 'vqav2'))
        print_green("Evaluation on VQAv2 finished. Please submit the results under {} to the server at {}.".format('results/vqav2/answers_upload', 'https://eval.ai/web/challenges/challenge-page/830/overview'))

    # MathVista
    if benchmarks['mathvista']['eval']:
        assert len(benchmarks['mathvista']['openai_api_key']) > 0
        print_green("Start the evaluation on MathVista")
        run_mathvista(create_config(config, 'mathvista'))
        print_green("Evaluation on MathVista finished.")




if __name__ == '__main__':
    # Add command-line arguments for the parameters you want to overwrite
    import argparse
    parser = argparse.ArgumentParser(description='Launch evaluation on multiple benchmarks')
    parser.add_argument('--config_file', type=str, default='config.json', help='Path to the config file')
    parser.add_argument('--model_module', type=str, help='Name of the model module')
    parser.add_argument('--model_path', type=str, help='Path to the model')
    parser.add_argument('--data_folder', type=str, help='Path to the data folder')
    parser.add_argument('--output_name', type=str, default="default", help='Name of the output file. If not provided (or `default`), the name of the model will be used.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--port', type=int, default=12355, help='Port number for the distributed training')
    # Add more arguments as needed

    # Parse the command-line arguments
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config_file = json.load(f)

    # Update the configuration with the command-line arguments
    num_gpus = torch.cuda.device_count()
    # num_gpus = 1
    config_file['world_size'] = num_gpus
    config_file['port'] = args.port
    benchmarks = config_file['benchmarks']
    if args.model_path:
        config_file['model_path'] = args.model_path
    if args.data_folder:
        config_file['data_folder'] = args.data_folder
    if args.output_name:
        output_name = args.output_name
        if output_name == "default":
            model_path = config_file['model_path']
            if model_path.endswith('/'):
                model_path = model_path[:-1]
            output_name = model_path.split('/')[-1]
            print(f"Passed output_name=default. Parsed {output_name} from model_path.")
        config_file['output_name'] = output_name

    # Call the main function
    print_red(f"Running evaluation on {num_gpus} GPUs")
    main(config_file, benchmarks, args.seed)
