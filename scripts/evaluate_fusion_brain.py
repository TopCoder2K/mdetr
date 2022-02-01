import argparse
import json
from pathlib import Path
from os import listdir
from collections import defaultdict

from util.evaluate_zsOD_fb import detection_evaluate
from util.evaluate_vqa2_fb import vqa_evaluate


def evaluate_vqa_answers(output_path):
    with open(output_path / "VQA/true_VQA.json", "r") as f:
        true_answers = json.load(f)
    with open(output_path / "VQA/prediction_VQA.json", "r") as f:
        predicted_answers = json.load(f)

    print("VQA")
    print(len(true_answers), len(predicted_answers))
    print(f"simple accuracy: {vqa_evaluate(true_answers, predicted_answers): .2f}%")
    print("===================================================================")


def evaluate_zsOD(output_path):
    with open(output_path / "zsOD/prediction_zsOD.json", "r") as f:
        translated_detections = json.load(f)

    # Уберём скоры уверенности детекций
    translated_detections_without_scores = {}
    for filename, img_requests in translated_detections.items():
        translated_detections_without_scores[filename] = defaultdict(list)
        for detected_obj in img_requests.items():
            for occurience in detected_obj[1]:
                if occurience[1] >= 0.0:
                    translated_detections_without_scores[filename][detected_obj[0]].append(occurience[0])
                else:
                    translated_detections_without_scores[filename][detected_obj[0]]  # просто создаём пустой список

    with open(output_path / "zsOD/true_zsOD.json", "r") as f:
        true_detections = json.load(f)

    print("zsOD")
    print(len(true_detections), len(translated_detections_without_scores))
    print(f"F1-score: {detection_evaluate(true_detections, translated_detections_without_scores): .4f}")
    print("===================================================================")


def parse_args():
    parser = \
        argparse.ArgumentParser("Evaluation answers script for Fusion Brain")

    parser.add_argument("--output_path", default="./output", type=str,
                        help="Path to the predicted and true fusion brain answers")

    return parser.parse_args()


def main(args):
    for folder in listdir(args.output_path):
        if folder == "VQA":
            evaluate_vqa_answers(Path(args.output_path))
        if folder == "zsOD":
            evaluate_zsOD(Path(args.output_path))


if __name__ == "__main__":
    main(parse_args())
