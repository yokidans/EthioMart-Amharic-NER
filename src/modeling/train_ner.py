# src/modeling/train_ner.py
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
import datasets
import numpy as np
from sklearn.metrics import classification_report

def load_data():
    dataset = datasets.load_dataset('json', 
                                 data_files={
                                     'train': 'data/labeled/train.jsonl',
                                     'val': 'data/labeled/val.jsonl'
                                 })
    return dataset

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["text"], 
                                truncation=True, 
                                padding='max_length',
                                max_length=128,
                                is_split_into_words=True)
    
    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
        
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    return classification_report(true_labels, true_predictions, output_dict=True)

def train():
    dataset = load_data()
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    
    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True
    )
    
    model = AutoModelForTokenClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=len(label_list)
    )
    
    training_args = TrainingArguments(
        output_dir="models/fine_tuned",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    trainer.save_model("models/fine_tuned/ethiomart_ner")

if __name__ == "__main__":
    train()