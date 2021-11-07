import argparse
import json
from pathlib import Path
from os import listdir

def evaluate_vqa_answers(data_path, output_path):
    with open(data_path / "true_VQA.json", "r") as f:
        true_answers = json.load(f)
    with open(output_path / "prediction_VQA.json", "r") as f:
        predicted_answers = json.load(f)

    accuracy = 0.
    for i in range(len(true_answers)):
        if true_answers[str(i)]["answer"] == \
                predicted_answers[str(i)]["answer"]:
            accuracy += 1.

    accuracy /= len(true_answers)
    print(f"VQA simple accuracy: {accuracy: .3f}")

def parse_args():
    parser = \
        argparse.ArgumentParser("Evaluation answers script for Fusion Brain")

    parser.add_argument("--data_path", default="", required=True, type=str,
                        help="Path to the TRUE fusion brain answers")
    parser.add_argument("--output_path", default="./output", type=str,
                        help="Path to the PREDICTED fusion brain answers")

    return parser.parse_args()

def main(args):
    for filename in listdir(args.output_path):
        if filename == "prediction_VQA.json":
            evaluate_vqa_answers(Path(args.data_path), Path(args.output_path))

if __name__ == "__main__":
    main(parse_args())
