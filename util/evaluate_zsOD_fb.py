import itertools
import numpy as np


def xywh_to_xyxy(bbox):
    """
    Из стандартного coco формата [xmin, ymin, w, h] -> [xmin, ymin, xmax, ymax]
    """
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def iou(box1, box2):
    """
    Считает IoU между двумя bbox, каждый box [xmin, ymin, xmax, ymax].
    Args:
      box1: (list) bbox, sized (4,).
      box2: (list) bbox, sized (4,).
    Return:
      float: iou
    """
    lt = np.zeros(2)
    rb = np.zeros(2)  # get inter-area left_top/right_bottom
    for i in range(2):
        if box1[i] > box2[i]:
            lt[i] = box1[i]
        else:
            lt[i] = box2[i]
        if box1[i + 2] < box2[i + 2]:
            rb[i] = box1[i + 2]
        else:
            rb[i] = box2[i + 2]
    wh = rb - lt
    wh[wh < 0] = 0  # if no overlapping
    inter = wh[0] * wh[1]
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = inter / (area1 + area2 - inter)
    if np.isnan(iou):
        iou = 0
    return iou


def detection_evaluate(true_annot, predictions):
    """Основная функция для подсчета метрик.
       Файл с правильными ответами имеет следующую структуру (это словарь) : {img_name1 : {label1:[[xmin,ymin,w,h],[xmin,ymin,w,h]], label2:[[xmin,ymin,w,h]]}, ....}

       0. создаем пустые переменные для статистик TP,FP,FN.
       1. Пробегаемся по каждому изображению из true (ключи true - имена файлов изображений).
       2. Внутри каждого изображения пробгеаемся по всем лейблам (ключи true[img_name] - названия лейблов).
       3. true = true_annot[img][label] - правильные боксы для данного лейбла на данном изображении. соответственно
          pred = predictions[img][label] - предсказанные боксы для данного лейбла на данном изображении.
       4. если len(true)==0 (то есть это был негативный лейбл в запросах), но модель участника сделала предсказание для него,
       увеличиваем FP на один за каждое предсказание
       5. если len(true)>0, но модель участника не сделала ни одного предсказания, то
       увеличиваем FN за каждый не найденные правильный бокс
       6. если len(true)>0 и если len(pred)>0:

       для начала смотрим пересечение по IoU true боксов со всеми pred боксами. Допустим предсказаний для изображения было 3, правильных бокса - 2. Тогда для каждого правильного бокса смотрим пересечение по IoU со всеми предсказанными, и если нет пересечений выше >0.5 ни с одним боксом, то мы считаем это за FN для каждого true бокса

       Затем мы пробегаемся по всем предсказаниям. Если у данного предсказания нет пересечения по IoU выше 0.5 ни с одним правильным боксом, мы считаем это предсказание за FP. Если же есть хоть с одним - считаем за TP.
       7. Затем считаются метрики precision и recall, и на их основе - финальный F1_score.


    Args:
      predictions: (string) path to pred.json
      true_annot: (string) path to true.json
    Return:
      float: f1_score
    """
    assert list(true_annot.keys()) == list(predictions.keys()), 'Not all images have predictions!'

    fp = 0
    fn = 0
    tp = 0

    for img in true_annot:
        for label in true_annot[img]:
            true = true_annot[img][label]
            assert label in predictions[
                img], f'There are no prediction for label "{label}" for image {img} in requests!'

            pred = predictions[img][label]

            if pred == [[]]:
                pred = []

            if len(pred) == 0 and len(true) == 0:
                continue
            elif len(pred) > 0 and len(true) == 0:
                fp += len(pred)
            elif len(pred) == 0 and len(true) > 0:
                fn += len(true)

            else:

                pairs = list(itertools.product(true, pred))
                pairs_iou = [(el[0], el[1], iou(xywh_to_xyxy(el[0]), xywh_to_xyxy(el[1]))) for el in pairs]

                for _, group in itertools.groupby(pairs_iou, key=lambda x: x[0]):  # true
                    if np.all(np.array([i for _, _, i in group]) < 0.5):
                        fn += 1

                for _, group in itertools.groupby(sorted(pairs_iou, key=lambda x: x[1]), key=lambda x: x[1]):  # pred
                    if np.all(np.array([i for _, _, i in group]) < 0.5):
                        fp += 1
                    else:
                        tp += 1

    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0

    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    if precision > 0 or recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0

    print(f"tp={tp}, fp={fp}, fn={fn}, precision={precision: .5f}, recall={recall: .5f}")
    return f1_score
