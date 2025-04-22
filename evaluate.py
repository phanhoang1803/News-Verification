import pandas as pd
import json
import os
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

captions = []
ground_truth = []
predicted = []
confidence_scores = []
inference_time_list = []
candidates = []
entities = []

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', type=str, default='result')
parser.add_argument('--skip_non_candidates', '-s', action='store_true')
args = parser.parse_args()
result_dir = args.result_dir

result_json_list = os.listdir(result_dir)

incorrect_index = []
correct_index = []
empty_evidence_count = 0
unmatch_result_idx = []
non_evidence_idx = []
for item in result_json_list:
    try:
        result_json_dir = os.path.join(result_dir, item)
        with open(result_json_dir, 'r', encoding='utf-8') as f:
            result_json = json.load(f)
            index = result_json_dir.split('_')[-1].split('.')[0]

            if result_json['check_result']['evidences'] == []:
                empty_evidence_count += 1
                non_evidence_idx.append(int(index))
                if args.skip_non_candidates:
                    continue
            
            # if result_json['check_result']['check_type'] != 'high quality evidences':
            #     continue
            
            captions.append(result_json['caption'])
            ground_truth.append(result_json['ground_truth'])
            predicted.append(1 if result_json['final_result']['OOC'] else 0)
            confidence_scores.append(result_json['final_result']['confidence_score'])
            inference_time_list.append(result_json['inference_time'])
            
            if result_json['ground_truth'] != result_json['final_result']['OOC']:
                incorrect_index.append(int(index))
            else:
                correct_index.append(int(index))
    except Exception as e:
        print(item)
        print(f"ERROR: {e}")
        raise

# save incorrect_index to a file
incorrect_index.sort()
with open('src/incorrect_index.txt', 'w') as f:
    for index in incorrect_index:
            f.write(str(index) + '\n')

print(f"Empty evidence count: {empty_evidence_count}")
print(len(captions))
print(len(ground_truth))
print(len(predicted))
print(len(inference_time_list))

df = pd.DataFrame({
    'caption': captions,
    'ground_truth': ground_truth,
    'predicted': predicted,
    'confidence_score': confidence_scores,
    'inference_time': inference_time_list,
})
df['adjusted_predicted'] = df.apply(
    lambda row: row['predicted'] if row['confidence_score'] >= 8 else 1 - row['predicted'], axis=1
)
adjusted_predicted = df['adjusted_predicted'].tolist()
df[(df['ground_truth'] != df['predicted'])]


class_names = ["NOOC", "OOC"]

# Generate Classification Report
report = classification_report(ground_truth, predicted, target_names=class_names)
print("\nClassification Report:")
print(report)
print("################################")

# calculate per-class accuracy
cm = confusion_matrix(ground_truth, predicted)
class_accuracies = cm.diagonal() / cm.sum(axis=1)
print("Per-Class Accuracy:")
for class_name, acc in zip(class_names, class_accuracies):
    print(f"{class_name}: {acc:.4f}")
print("################################")

# Calculate average inference time
average_time = sum(inference_time_list) / len(inference_time_list)
print(f"Average Inference Time: {average_time:.6f} seconds")
