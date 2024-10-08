import pandas as pd
from datasets import Dataset, load_dataset, DatasetDict, load_from_disk
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, get_cosine_schedule_with_warmup, BartTokenizer, BartForSequenceClassification
from sklearn.metrics import accuracy_score
import torch  # To check GPU availability

# Check if GPU is available and print
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(f"Using device: {device}")
#device = 'cuda'
# Load
train_dataset = load_from_disk('train_dataset-bart')
validation_dataset = load_from_disk('validation_dataset-bart')
test_dataset = load_from_disk('test_dataset-bart')


device = "cuda:0"
# Initialize and load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base', clean_up_tokenization_spaces=True)


# Load the pre-trained BART model
model = BartForSequenceClassification.from_pretrained(
    'facebook/bart-base', 
    num_labels=85437,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map='cuda' # Move the model to the GPU for some reason this is with device_map='cuda' for BART ??
    )

# Set up training arguments with GPU configuration
training_args = TrainingArguments(
    output_dir='./results-bart',
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
    #fp16=True,  # Use mixed precision training, doesnt work on BART? 
)

# Define metrics for evaluation
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {'accuracy': accuracy_score(p.label_ids, preds)}

# Initialize the Trainer
with torch.autocast(device_type='cuda', dtype=torch.bfloat16): #no idea why this works and does not throw errors
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
model.save_pretrained('fine_tuned_bart')
tokenizer.save_pretrained('fine_tuned_bart')
