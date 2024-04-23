import os
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str, required=True)
    parser.add_argument('--result-upload-file', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    os.makedirs(os.path.dirname(args.result_upload_file), exist_ok=True)

    results = []
    for line_idx, line in enumerate(open(args.result_file)):
        data = json.loads(line)
        results.append({'questionId':data['question_id'], 'answer':data['text']})

    with open(args.result_upload_file, 'w') as f:
        json.dump(results, f)