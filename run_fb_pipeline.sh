#!/bin/bash
# whoami
# pwd

# Перевод вопросов
python scripts/translate.py --questions --data_path input/VQA/
# Генерация coco формата
python scripts/fine-tuning/vqa_coco_format.py --data_path input/VQA/ --fusion_brain --use_translated
# Перемещение asnwer2id[_by_type].json в папку с данными для консистентности
mv vqa2_answer2id.json ./input/VQA/
mv vqa2_answer2id_by_type.json ./input/VQA/

# Выставляем детерминистичность
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# В коде fusion_brain.py также указываем, где cache torch
# resnet101:__init__() -> _resnet() -> load_state_dict_from_url() -> torch.hub.load_state_dict_from_url() ->
# model_dir = get_dir() -> _get_torch_home()
# И запускаем модель
python fusion_brain.py --dataset_config configs/vqa2_fusion_brain.json --do_qa --load ~/Internship/MDETR/mdetr/checkpoints/BEST_checkpoint.pth --batch_size 4 --inference

# Перевод ответов
python scripts/translate.py --answers --data_path input/VQA/ --output_path output/

# Запуск оценивания
# pip install -r eval_scripts/VQA/requirements.txt
# python eval_scripts/VQA/evaluate.py --ref_path ./output/true_VQA.json --pred_path ./output/prediction_VQA.json ---> 0.23375