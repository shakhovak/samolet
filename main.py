import logging
from tqdm import tqdm
from utils import (
    text_naive_classifier,
    split_to_chunk_inference
)
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import pandas as pd
from nltk.tokenize import word_tokenize
import argparse
import nltk

parser = argparse.ArgumentParser()
parser.add_argument(
    "test_file_path",
    nargs="?",
    type=str,
    default="data/test.csv",
    help="file to rework with model",
)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

nltk.download("punkt")

def create_dataset_tags(path_to_dataset):
    logger.info("Reading and augementing datasets...")
    list_df = []
    with open(path_to_dataset, encoding="utf-8") as f:
        for row in f:
            text, labels = row.split(",", maxsplit=1)[0], row.split(",", maxsplit=1)[1]
            list_df.append([text, labels])
    df = pd.DataFrame(
        list_df[1:], columns=["text", "label"]
    )  # удаляем строку с заголовками

    logger.info("Text classification started...")
    DISC_SUBST = "скидк"
    MU_SUBSTR = "проце"

    token_class_model = AutoModelForTokenClassification.from_pretrained(
        "Shakhovak/samolet"
    )
    token_class_tokenizer = AutoTokenizer.from_pretrained("Shakhovak/samolet")
    # Determine the device
    device = torch.device("cpu")
    token_class_model.to(device)

    df_with_answers = pd.DataFrame()
    for _, row in tqdm(df.iterrows()):
        class_result = text_naive_classifier(row['text'],
                                             disc_substr=DISC_SUBST,
                                             mu_substr=MU_SUBSTR)
        if class_result == 1:
            chunks = split_to_chunk_inference(row['text'])
            predictions = []
            for chunk in chunks:
                tokens = word_tokenize(chunk)
                tokenized_inputs = token_class_tokenizer(
                    tokens,
                    truncation=True,
                    is_split_into_words=True,
                    return_tensors='pt'
                )
                token_class_model.eval()
                with torch.no_grad():
                    outputs = token_class_model(**tokenized_inputs)
                    prediction = torch.argmax(outputs.logits, dim=2)
                    predicted_token_class = [
                        token_class_model.config.id2label[t.item()] for t in prediction[0]
                    ]
                    tokens = token_class_tokenizer.convert_ids_to_tokens(
                        tokenized_inputs["input_ids"].squeeze().tolist()
                    )
                    wp_preds = list(zip(tokens, predicted_token_class))
                    word_level_predictions = []
                    for pair in wp_preds:
                        if (pair[0].startswith("##")) or (pair[0] in ['[CLS]', '[SEP]', '[PAD]']):
                            # skip prediction
                            continue
                        else:
                            word_level_predictions.append(pair[1])
                    predictions.append(word_level_predictions)
            new_line = {
                "processed_text": row["text"],
                "label": [label for sublist in predictions for label in sublist],
            }
        else:
            new_line = {
                "processed_text": row["text"],
                "label": ["O"] * len(word_tokenize(row["text"])),
            }
        df_with_answers = pd.concat([df_with_answers, pd.DataFrame([new_line])])
    return df_with_answers


create_dataset_tags(args.test_file_path).to_csv("data/gt_test.csv", index=False)
logger.info("All done and file saved!")
