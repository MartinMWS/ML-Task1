import pandas as pd
from datasets import Dataset, load_dataset, DatasetDict, load_from_disk
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score
import torch  # To check GPU availability

# Check if GPU is available and print
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load tokenized datasets
#train_df = pd.read_csv('train_dataset.csv')
#validation_df = pd.read_csv('validation_dataset.csv')
#test_df = pd.read_csv('test_dataset.csv')

# Load
train_dataset = load_from_disk('train_dataset')
validation_dataset = load_from_disk('validation_dataset')
test_dataset = load_from_disk('test_dataset')

# Load the pre-trained BERT model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=85437)

# Initialize and load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased',clean_up_tokenization_spaces=True)

# Move the model to the GPU
model.to(device)



# Set up training arguments with GPU configuration
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=500,
    eval_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    metric_for_best_model="accuracy",
    save_strategy="epoch",
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    
    # Enable GPU usage
    report_to="none",  # to avoid unnecessary logging
    fp16=True,  # Use mixed precision training
)

# Define metrics for evaluation
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {'accuracy': accuracy_score(p.label_ids, preds)}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate(test_dataset)
print(f"Evaluation results: {eval_results}")

# Save the fine-tuned model and tokenizer
model.save_pretrained('fine_tuned_bert')
tokenizer.save_pretrained('fine_tuned_bert') 
