import os
import argparse
import json


def cal_relaxed_accuracy(pred, gt):
    return 1 if abs(pred-gt) <= abs(gt)*0.05 else 0

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    category2total_number = dict()
    category2correct_number = dict()

    all_answers = dict()
    for line in open(args.result_file):
        data = json.loads(line)
        question_id = data['question_id']
        text = data['text'].rstrip('.').lower()
        all_answers[question_id] = text


    for line in open(args.annotation_file):
        data = json.loads(line)
        question_id = data['question_id']
        label = data['label'].rstrip('.').lower()
        category = data['category']

        if is_number(label):
            if not is_number(all_answers[question_id]):
                correct = 0
            else:
                correct = cal_relaxed_accuracy(float(all_answers[question_id]), float(label))
        else:
            correct = 1 if label == all_answers[question_id] else 0

        if category not in category2total_number:
            category2total_number[category] = 0
            category2correct_number[category] = 0
        category2total_number[category] += 1
        category2correct_number[category] += correct

    accuracy_list = []

    for category in category2total_number.keys():
        accuracy = category2correct_number[category]/category2total_number[category]
        print("Accuray for category {} is {}.".format(category, accuracy))
        accuracy_list.append(accuracy)

    print("Averaged Accuracy is: {}.".format(sum(accuracy_list)/len(accuracy_list)))
