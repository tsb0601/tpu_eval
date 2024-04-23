import os
import json
import argparse

import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, required=True)
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--upload-dir", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)

    return parser.parse_args()


def convert_mmbench_for_submission(annotation_file, result_dir, upload_dir, experiment) -> str:

    df = pd.read_table(annotation_file)

    cur_df = df.copy()
    cur_df = cur_df.drop(columns=['hint', 'category', 'source', 'image', 'comment', 'l2-category'])
    cur_df.insert(6, 'prediction', None)
    for pred in open(os.path.join(result_dir, f"{experiment}.jsonl")):
        pred = json.loads(pred)
        if len(pred['text']) > 1 and pred['text'][0] in ['A', 'B', 'C', 'D']:
            pred['text'] = pred['text'][0]
        cur_df.loc[df['index'] == pred['question_id'], 'prediction'] = pred['text']

    path = os.path.join(upload_dir, f"{experiment}.xlsx")
    cur_df.to_excel(path, index=False, engine='openpyxl')
    return path


if __name__ == "__main__":
    args = get_args()

    convert_mmbench_for_submission(args.annotation_file, args.result_dir, args.upload_dir, args.experiment)
