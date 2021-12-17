from collections import defaultdict
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from deep_translator import GoogleTranslator


def simple_detect_lang(text):
    if len(set("абвгдежзийклмнопрстуфхцчшщъыьэюяё").intersection(
            text.lower())) > 0:
        return "ru"
    if len(set("abcdefghijklmnopqrstuvwxyz").intersection(
            text.lower())) > 0:
        return "en"


def translate_questions_from_ru_into_en(data_path):
    # ruen_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    # ruen_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    # ruen_translator = GoogleTranslator(source='ru', target='en')

    ruen_tokenizer = AutoTokenizer.from_pretrained(
        "Helsinki-NLP/opus-mt-ru-en",
        cache_dir="./checkpoints/"
    )
    ruen_model = AutoModelForSeq2SeqLM.from_pretrained(
        "Helsinki-NLP/opus-mt-ru-en",
        cache_dir="./checkpoints/"
    )

    with open(data_path / "questions.json", "r") as f:
        questions = json.load(f)

    translated_ques = defaultdict(dict)
    for item in tqdm(questions.items()):
        translated_ques[item[0]] = item[1]
        translated_ques[item[0]]["original_lang"] = "en"

        if simple_detect_lang(translated_ques[item[0]]["question"]) == "ru":
            translated = ruen_model.generate(**ruen_tokenizer(
                translated_ques[item[0]]["question"],
                return_tensors="pt",
                padding=True
            ))
            translated_ques[item[0]]["question"] = \
                ruen_tokenizer.decode(translated.squeeze(),
                                      skip_special_tokens=True)
            translated_ques[item[0]]["original_lang"] = "ru"

    with open(data_path / "translated_questions.json", "w") as f:
        json.dump(translated_ques, f)


def translate_answers_from_en_into_ru(data_path, output_path, online=False):
    # enru_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
    # enru_model = \
    #     AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru")

    with open(data_path / "VQA/translated_questions.json", "r") as f:
        translated_questions = json.load(f)
    with open(output_path / "VQA/prediction_VQA_untranslated.json", "r") as f:
        untranslated_answers = json.load(f)
    translated_ans = {}

    if not online:
        enru_tokenizer = AutoTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-en-ru",
            cache_dir="./checkpoints/"
        )
        enru_model = AutoModelForSeq2SeqLM.from_pretrained(
            "Helsinki-NLP/opus-mt-en-ru",
            cache_dir="./checkpoints/"
        )

        for i in tqdm(range(len(untranslated_answers))):
            translated_ans[str(i)] = untranslated_answers[str(i)]

            if translated_questions[str(i)]["original_lang"] == "ru":
                translated = enru_model.generate(**enru_tokenizer(
                    translated_ans[str(i)],
                    return_tensors="pt",
                    padding=True
                ))
                translated_ans[str(i)] = enru_tokenizer.decode(
                    translated.squeeze(), skip_special_tokens=True
                )
    else:
        print("Running online translation on answers...")
        # print(translated_questions["0"], untranslated_answers["0"])
        enru_translator = GoogleTranslator(source='en', target='ru')

        for i in tqdm(range(len(untranslated_answers))):
            translated_ans[str(i)] = untranslated_answers[str(i)]

            if translated_questions[str(i)]["original_lang"] == "ru" and not untranslated_answers[str(i)].isdigit():
                translated_ans[str(i)] = enru_translator.translate(untranslated_answers[str(i)])

    with open(output_path / "VQA/prediction_VQA.json", "w") as f:
        json.dump(translated_ans, f)


def parse_args():
    parser = argparse.ArgumentParser("Translation script")

    parser.add_argument("--questions", action="store_true",
                        help="Whether you want to translate questions")
    parser.add_argument("--answers", action="store_true",
                        help="Whether you want to translate answers")

    parser.add_argument("--online", action="store_true",
                        help="Whether to use online translation")

    parser.add_argument("--data_path", default="", required=True, type=str,
                        help="Path to the vqa dataset")
    parser.add_argument("--output_path", default="./output", type=str,
                        help="Path to the output produced by the model")

    return parser.parse_args()


def main(args):
    if args.questions and args.answers:
        raise RuntimeError("Use only one option.")

    with torch.no_grad():
        if args.questions:
            translate_questions_from_ru_into_en(Path(args.data_path))
        else:
            translate_answers_from_en_into_ru(
                Path(args.data_path), Path(args.output_path), args.online
            )


if __name__ == "__main__":
    main(parse_args())
