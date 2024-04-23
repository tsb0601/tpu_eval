import os
import argparse
import json

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--annotation-file', type=str)
	parser.add_argument('--result-file', type=str)
	return parser.parse_args()


OCRBench_score = {
		'Regular Text Recognition': 0,
		'Irregular Text Recognition': 0,
		'Artistic Text Recognition': 0,
		'Handwriting Recognition': 0,
		'Digit String Recognition': 0,
		'Non-Semantic Text Recognition': 0,
		'Scene Text-centric VQA': 0,
		'Doc-oriented VQA': 0,
		'Key Information Extraction': 0,
		'Handwritten Mathematical Expression Recognition': 0
	}

if __name__ == "__main__":
	args = get_args()

	
	all_answers = []
	for line in open(args.result_file):
		data = json.loads(line)
		text = data['text']
		all_answers.append(text)

	i = 0
	for line in open(args.annotation_file):
		data = json.loads(line)
		predict = all_answers[i]
		i += 1
		answers = data['label']
		
		if not type(answers)==list:
			answers = [answers]
		
		category = data["category"]
		dataset_name = data["dataset_name"]
		success = 0
		if dataset_name == "HME100k":
			for j in range(len(answers)):
				answer = answers[j].strip().replace('\n', ' ').replace(' ', '')
				predict = predict.strip().replace('\n', ' ').replace(' ', '')
				if answer in predict:
					OCRBench_score[category] += 1
					success = 1
					break
		else:
			for j in range(len(answers)):
				answer = answers[j].lower().strip().replace('\n', ' ')
				predict = predict.lower().strip().replace('\n', ' ')
				if answer in predict:
					OCRBench_score[category] += 1
					success = 1
					break

	recognition_score = OCRBench_score['Regular Text Recognition']+OCRBench_score['Irregular Text Recognition']+OCRBench_score['Artistic Text Recognition']+OCRBench_score['Handwriting Recognition']+OCRBench_score['Digit String Recognition']+OCRBench_score['Non-Semantic Text Recognition']
	Final_score = recognition_score+OCRBench_score['Scene Text-centric VQA']+OCRBench_score['Doc-oriented VQA']+OCRBench_score['Key Information Extraction']+OCRBench_score['Handwritten Mathematical Expression Recognition']
	print(f"Text Recognition(Total 300):{recognition_score}")
	print("------------------Details of Recognition Score-------------------")
	print(f"Regular Text Recognition(Total 50): {OCRBench_score['Regular Text Recognition']}")
	print(f"Irregular Text Recognition(Total 50): {OCRBench_score['Irregular Text Recognition']}")
	print(f"Artistic Text Recognition(Total 50): {OCRBench_score['Artistic Text Recognition']}")
	print(f"Handwriting Recognition(Total 50): {OCRBench_score['Handwriting Recognition']}")
	print(f"Digit String Recognition(Total 50): {OCRBench_score['Digit String Recognition']}")
	print(f"Non-Semantic Text Recognition(Total 50): {OCRBench_score['Non-Semantic Text Recognition']}")
	print("----------------------------------------------------------------")
	print(f"Scene Text-centric VQA(Total 200): {OCRBench_score['Scene Text-centric VQA']}")
	print("----------------------------------------------------------------")
	print(f"Doc-oriented VQA(Total 200): {OCRBench_score['Doc-oriented VQA']}")
	print("----------------------------------------------------------------")
	print(f"Key Information Extraction(Total 200): {OCRBench_score['Key Information Extraction']}")
	print("----------------------------------------------------------------")
	print(f"Handwritten Mathematical Expression Recognition(Total 100): {OCRBench_score['Handwritten Mathematical Expression Recognition']}")
	print("----------------------Final Score-------------------------------")
	print(f"Final Score(Total 1000): {Final_score}")


