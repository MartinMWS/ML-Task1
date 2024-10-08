import pandas as pd
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer
import multiprocessing
num_cores = multiprocessing.cpu_count()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)

def sanitize_text(example):
    # Check if 'abstract' exists and is a non-empty string
    if example.get('abstract') and isinstance(example['abstract'], str):
        example['abstract'] = example['abstract'].replace('\n', ' ').strip()
    return example

def tokenize_function(examples):
    
    tokens = tokenizer(
        examples['abstract'], 
        padding='max_length', 
        truncation=True, 
        max_length=512,
    )
    # Ensure 'labels' are included in the tokenized output
    tokens['labels'] = examples['categories']
    return tokens

if __name__ == '__main__':
    # Step 1: Load your JSON dataset
    dataset = load_dataset('json', data_files='arxiv-metadata-oai-snapshot.json')
    #print(dataset.features)
    
    # Step 2: Keep only 'abstract' and 'categories'
    dataset = dataset.map(lambda x: {'abstract': x['abstract'], 'categories': x['categories']})

    
    
    # Step 3: Apply sanitization to 'abstract'
    sanitized_dataset = dataset.map(sanitize_text)

    del dataset # Free memory
    print('dataset unloaded')

    # Step 4: Automatically convert 'categories' to numerical labels
    # `ClassLabel` automatically maps categories to integers
    sanitized_dataset = sanitized_dataset.class_encode_column('categories')


    # Step 5: Tokenize the sanitized dataset
    tokenized_dataset = sanitized_dataset.map(tokenize_function, batched=True, num_proc=num_cores)

    del sanitized_dataset  # Free memory
    print('unloaded sanitized_dataset')

    # Step 6: Split the dataset into train, validation, and test sets
    train_test_split = tokenized_dataset['train'].train_test_split(test_size=0.2)
    train_validation_split = train_test_split['train'].train_test_split(test_size=0.2)

    del tokenized_dataset  # Free memory
    print('unloaded tokenized_dataset')

    train_dataset = train_validation_split['train']
    train_dataset.save_to_disk('train_dataset')
    del train_dataset # Free memory

    validation_dataset = train_validation_split['test']
    validation_dataset.save_to_disk('validation_dataset')
    del validation_dataset # Free memory

    test_dataset = train_test_split['test']
    test_dataset.save_to_disk('test_dataset')
    del test_dataset # Free memory
    
    print("Preprocessed and tokenized datasets have been saved as Arrow files!")
