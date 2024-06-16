from flask import Flask, render_template, request, send_file
from tqdm import tqdm
from utils import text_naive_classifier, split_to_chunk_inference
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk

app = Flask(__name__)

nltk.download("punkt")

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    # Check if the POST request has a file part
    if "file" not in request.files:
        return "No file part"

    file = request.files["file"]

    # If the user submits an empty form
    if file.filename == "":
        return "No selected file"

    # Check if the file is a text file
    if file and file.filename.endswith("csv"):
        file.save(file.filename)
        list_df = []
        with open(file.filename, encoding="utf-8") as f:
            for row in f:
                text, labels = row.split(",", maxsplit=1)[0], row.split(",", maxsplit=1)[1]
                list_df.append([text, labels])
        df = pd.DataFrame(
            list_df[1:], columns=["text", "label"]
        )  # удаляем строку с заголовками
        DISC_SUBST = "скидк"
        MU_SUBSTR = "проце"

        token_class_model = AutoModelForTokenClassification.from_pretrained(
            "./results/token_checkpoint-last"
        )
        token_class_tokenizer = AutoTokenizer.from_pretrained("./results/token_checkpoint-last")
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
            df_with_answers.to_csv("gt_test.csv", index=False)

        return render_template("result.html")

    else:
        return "Invalid file format. Please upload a text file."


@app.route("/process_template", methods=["POST"])
def process_template():
    list_df = []
    with open("data/test.csv", encoding="utf-8") as f:
        for row in f:
            text, labels = (
                row.split(",", maxsplit=1)[0],
                row.split(",", maxsplit=1)[1],
            )
            list_df.append([text, labels])
    df = pd.DataFrame(
        list_df[1:], columns=["text", "label"]
    )  # удаляем строку с заголовками
    DISC_SUBST = "скидк"
    MU_SUBSTR = "проце"

    token_class_model = AutoModelForTokenClassification.from_pretrained(
        "./results/token_checkpoint-last"
    )
    token_class_tokenizer = AutoTokenizer.from_pretrained(
        "./results/token_checkpoint-last"
    )
    # Determine the device
    device = torch.device("cpu")
    token_class_model.to(device)

    df_with_answers = pd.DataFrame()
    for _, row in tqdm(df.iterrows()):
        class_result = text_naive_classifier(
            row["text"], disc_substr=DISC_SUBST, mu_substr=MU_SUBSTR
        )
        if class_result == 1:
            chunks = split_to_chunk_inference(row["text"])
            predictions = []
            for chunk in chunks:
                tokens = word_tokenize(chunk)
                tokenized_inputs = token_class_tokenizer(
                    tokens,
                    truncation=True,
                    is_split_into_words=True,
                    return_tensors="pt",
                )
                token_class_model.eval()
                with torch.no_grad():
                    outputs = token_class_model(**tokenized_inputs)
                    prediction = torch.argmax(outputs.logits, dim=2)
                    predicted_token_class = [
                        token_class_model.config.id2label[t.item()]
                        for t in prediction[0]
                    ]
                    tokens = token_class_tokenizer.convert_ids_to_tokens(
                        tokenized_inputs["input_ids"].squeeze().tolist()
                    )
                    wp_preds = list(zip(tokens, predicted_token_class))
                    word_level_predictions = []
                    for pair in wp_preds:
                        if (pair[0].startswith("##")) or (
                            pair[0] in ["[CLS]", "[SEP]", "[PAD]"]
                        ):
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
    df_with_answers.to_csv("gt_test.csv", index=False)

    return render_template("result.html")

@app.route("/download/<file_name>")
def download(file_name):
    file_name = f"{file_name}"
    return send_file(file_name, as_attachment=True, download_name=f"{file_name}")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
