import os
import argparse
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    correct = []

    all_answers = dict()
    for line in open(args.result_file):
        data = json.loads(line)
        question_id = data['question_id']
        text = data['text'].rstrip('.').lower()
        if len(text) > 1:
            if text[:3] in ['(a)', '(b)', '(c)', '(d)']:
                text = text[1]
        all_answers[question_id] = text


    for line in open(args.annotation_file):
        data = json.loads(line)
        question_id = data['question_id']
        label = data['label'].rstrip('.').lower()

        correct.append(label==all_answers[question_id])

    correct2 = []
    for i in range(0, len(correct), 2):
        correct2.append(correct[i] and correct[i+1])

    # print("{} correct in total {}. Accuracy {}".format(sum(correct), len(correct), sum(correct)/len(correct)))

    print("{} correct in total {}. Accuracy {}".format(sum(correct2), len(correct2), sum(correct2)/len(correct2)))