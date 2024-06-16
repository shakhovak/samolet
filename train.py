import pandas as pd
import logging
import argparse
import warnings
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    AutoTokenizer,
)
import evaluate
from sklearn.model_selection import train_test_split
from utils import (
    augment_data,
    create_chunked_df,
    preprocess_text,
    tokenize_and_align_labels,
    prepare_compute_metrics,
)

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('seed', nargs='?', type=int, default=15, help="seed to split train and test")
parser.add_argument("chunk_size", nargs='?', type=int, default=360, help="chunk in words to split data")
parser.add_argument(
    "model_name",
    nargs='?',
    type=str,
    default="ai-forever/ruBert-base",
    help="model to be used",
)
parser.add_argument("train_data_path",
                    nargs='?',
                    type=str,
                    default='data/train_data.csv',
                    help="path to train data")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()
# Препроцессинг данных
logger.info("Reading and augementing datasets...")
list_df = []
with open(args.train_data_path, encoding="utf-8") as f:
    for row in f:
        text, labels = row.split(",", maxsplit=1)[0], row.split(",", maxsplit=1)[1]
        list_df.append([text, labels])


df = pd.DataFrame(
    list_df[1:], columns=["text", "label"]
)

df["label"] = df["label"].str.replace('''"''', "")
df["target"] = df["label"].apply(lambda x: 0 if x == "{}\n" else 1)

df = df.drop(labels=[966, 188, 1398, 2089], axis="index")
df = df.reset_index(drop=True)

train, test = train_test_split(df,
                               test_size=0.01,
                               random_state=args.seed)
train_pos = train[train['target'] == 1]
train_neg = train[train["target"] == 0]
sample_neg = train_neg.sample(frac=0.75, replace=False)
df_augmented = pd.concat([train_pos, sample_neg, augment_data(train[train["target"] == 1])])
df_augmented = df_augmented.reset_index(drop=True)

train_chunked = create_chunked_df(df_augmented, args.chunk_size)
test_chunked = create_chunked_df(test, args.chunk_size)

final_ds = DatasetDict(
    {
        "train": Dataset.from_pandas(train_chunked[["chunk_text", "chunk_label_shifted"]]),
        "test": Dataset.from_pandas(
            test_chunked[["chunk_text", "chunk_label_shifted"]]
        ),
    }
)
final_ds = final_ds.shuffle()

entity_groups = ["O", "B-discount", "B-value", "I-value"]
id2label = {i: label for i, label in enumerate(entity_groups)}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)
label_list = list(label2id.keys())

processed_dataset = final_ds.map(
    preprocess_text,
    remove_columns=["chunk_text", "chunk_label_shifted"],
)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

tokenized_dataset = processed_dataset.map(
    tokenize_and_align_labels,
    fn_kwargs={"tokenizer": tokenizer},
    batched=True,
    remove_columns=["tokens", "ner_tag"],
)

logger.info("Loading model and starting training...")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

model = AutoModelForTokenClassification.from_pretrained(
    args.model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)
seqeval = evaluate.load("seqeval")

compute_metrics = prepare_compute_metrics(label_list=label_list,
                                          seqeval=seqeval)

training_args = TrainingArguments(
    output_dir="token_class_model",
    learning_rate=5e-05,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    num_train_epochs=8,
    weight_decay=0.1,
    evaluation_strategy="epoch",
    push_to_hub=False,
    save_strategy="no",
    group_by_length=True,
    warmup_ratio=0.1,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# Save the fine-tuned model locally
model.save_pretrained("./results/token_checkpoint-last")
tokenizer.save_pretrained("./results/token_checkpoint-last")

logger.info("Model trained and saved!")
