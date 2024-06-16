import random
import pandas as pd
from nltk.tokenize import word_tokenize
import ast
import json
import numpy as np


def augment_data(train_dataset):
    """функция для дополнения данных со случайным изменением чисел"""

    with open("data/utils_data.json", encoding="UTF-8") as f:
        replacement_list = json.load(f)

    augmented_df = pd.DataFrame()
    for _, row in train_dataset.iterrows():
        if row["label"] != "{}\n":
            tokens = word_tokenize(row["text"])
            new_line = {}
            for k, v in ast.literal_eval(row["label"]).items():
                if k == "B-value":
                    if (len(v) == 1) & (
                        tokens[int(v[0])] in replacement_list["b_values_to_replace"]
                    ):
                        tokens[int(v[0])] = random.choice(replacement_list["b_values_random"])
                    elif len(v) > 1:
                        for i in v:
                            if (
                                tokens[int(i)]
                                in replacement_list["b_values_to_replace"]
                            ):
                                tokens[int(i)] = random.choice(
                                    replacement_list["b_values_random"]
                                )
            new_line = {
                "text": " ".join(tokens),
                "label": row["label"],
                "target": row["target"],
            }
            augmented_df = pd.concat([augmented_df, pd.DataFrame([new_line])])
    return augmented_df


def split_to_chunk(text, label, txt_idx, CHUNK_SIZE=400):
    """функция для разбивки текстов на чанки"""
    chunk_text_list = []  # текст чанка
    chunk_range_list = (
        []
    )  # тупл номеров токенов начала и конца чанка в оригинальной строки
    chunk_label, chunk_label_shifted = (
        [],
        [],
    )
    orig_text = []  # оригинальные тексты
    orig_text_num = (
        []
    )
    text_list = word_tokenize(text)
    curr_l = len(text_list)
    if curr_l > CHUNK_SIZE:
        for beg in range(0, curr_l, CHUNK_SIZE):
            chunk_text = " ".join(text_list[beg:beg + CHUNK_SIZE])
            chunk_text_list.append(chunk_text)
            if beg - CHUNK_SIZE > 0:
                chunk_range_list.append((beg, beg + CHUNK_SIZE))
            else:
                chunk_range_list.append((beg, curr_l))
            # составим два словаря попавших в чанк токенов
            curr_label, curr_label_shifted = {}, {}
            for k, v in ast.literal_eval(label).items():
                if len([pos for pos in v if pos >= beg and pos < beg + CHUNK_SIZE]) > 0:
                    curr_label[k] = [
                        pos for pos in v if pos >= beg and pos < beg + CHUNK_SIZE
                    ]
                    curr_label_shifted[k] = [
                        pos - beg for pos in v if pos >= beg and pos < beg + CHUNK_SIZE
                    ]

            chunk_label.append(
                str(curr_label)
            )  # для общности будем хранить словари как строки
            chunk_label_shifted.append(str(curr_label_shifted))
            orig_text.append(text)
            orig_text_num.append(txt_idx)
    else:
        orig_text.append(text)
        orig_text_num.append(txt_idx)
        chunk_text_list.append(text)
        chunk_range_list.append((0, curr_l))
        chunk_label.append(label)
        chunk_label_shifted.append(label)
    return (
        orig_text,
        orig_text_num,
        chunk_text_list,
        chunk_range_list,
        chunk_label,
        chunk_label_shifted,
    )


def create_chunked_df(dataset, chunk_size):
    """функция для создания датасета из чанков"""
    df_chunk = pd.DataFrame()
    for idx, row in dataset.iterrows():
        (
            orig_text,
            orig_text_num,
            chunk_text_list,
            chunk_range_list,
            chunk_label,
            chunk_label_shifted,
        ) = split_to_chunk(
            text=row["text"],
            label=row["label"],
            txt_idx=idx,
            CHUNK_SIZE=chunk_size,
        )

        df_chunk = pd.concat(
            [
                df_chunk,
                pd.DataFrame(
                    {
                        "orig_text": orig_text,
                        "orig_text_num": orig_text_num,
                        "chunk_text": chunk_text_list,
                        "chunk_range": chunk_range_list,
                        "chunk_label": chunk_label,
                        "chunk_label_shifted": chunk_label_shifted,
                    }
                ),
            ]
        )
    df_chunk = df_chunk.reset_index(drop=True)
    return df_chunk


def preprocess_text(example):
    """функция для препроцессинга данных для обработки NER"""
    words = []
    tags = []
    new_dict = {}
    labels_dict = ast.literal_eval(example["chunk_label_shifted"])
    tags_index_list = [i[0] for i in labels_dict.values()]
    for key, value in labels_dict.items():
        for i in value:
            new_dict[i] = key
    for index, i in enumerate(word_tokenize(example["chunk_text"])):
        if index in tags_index_list:
            if new_dict[index] == "B-discount":
                tags.append(1)
                words.append(i)
            elif new_dict[index] == "B-value":
                tags.append(2)
                words.append(i)
            else:
                tags.append(3)
                words.append(i)
        else:
            tags.append(0)
            words.append(i)

    example["ner_tag"] = tags
    example["tokens"] = words
    return example


def tokenize_and_align_labels(examples, tokenizer):
    """функция для обработки служебных токенов при токенизации"""
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tag"]):
        word_ids = tokenized_inputs.word_ids(
            batch_index=i
        )  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif (
                word_idx != previous_word_idx
            ):  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def prepare_compute_metrics(label_list, seqeval):
    """функция для расчета метрики при обучении модели"""
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions,
                                  references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_metrics


def text_naive_classifier(text, disc_substr, mu_substr):
    """функция для классифкатора по содержанию слов"""
    assert isinstance(text, str)
    assert isinstance(disc_substr, str)
    assert isinstance(mu_substr, str)
    disc_substr = 1 if disc_substr in text else 0
    mu_ind = 1 if mu_substr in text else 0
    return disc_substr or mu_ind


def split_to_chunk_inference(text, CHUNK_SIZE=400):
    """функция для разбивки на чанки в инференсе"""
    chunk_text_list = []  # текст чанка
    text_list = word_tokenize(text)
    curr_l = len(text_list)
    if curr_l > CHUNK_SIZE:
        for beg in range(0, curr_l, CHUNK_SIZE):
            chunk_text = " ".join(text_list[beg:beg + CHUNK_SIZE])
            chunk_text_list.append(chunk_text)
    else:
        chunk_text_list.append(text)
    return chunk_text_list
