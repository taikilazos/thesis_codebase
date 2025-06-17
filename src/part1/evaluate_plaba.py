import json
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from medreadme import MedReadmeDataset, calculate_metrics, extract_entities
import argparse
from tqdm import tqdm

def load_model(model_path, device):
    """Load a trained model from path"""
    # Use 8 classes to match the checkpoint
    model = AutoModelForTokenClassification.from_pretrained("roberta-base", num_labels=2)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def map_labels_to_medreadme(label):
    """Map PLABA labels to MedReadme 7-class format"""
    label = label.lower()
    if 'google easy' in label:
        return 'GOOGLE_EASY'
    elif 'google hard' in label:
        return 'GOOGLE_HARD'
    elif 'medical name' in label:
        return 'MEDICAL_NAME'
    elif 'medical abbreviation' in label:
        return 'MEDICAL_ABBR'
    elif 'general abbreviation' in label:
        return 'GENERAL_ABBR'
    elif 'general complex' in label:
        return 'GENERAL_COMPLEX'
    else:
        return 'O'

def process_plaba_data(data_path, tokenizer, classification_type='7-cls'):
    """Process PLABA data into MedReadme format"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    processed_data = []
    for item in data:
        tokens = item['tokens']
        entities = item['entities']
        
        # Initialize labels
        labels = ['O'] * len(tokens)
        
        # Fill in entity labels
        for entity in entities:
            text = entity['text']
            label = entity['label']
            start = None
            end = None
            
            # Find the start and end indices of the entity
            for i in range(len(tokens)):
                if tokens[i] in text:
                    if start is None:
                        start = i
                    end = i + 1
            
            if start is not None:
                # Map label to MedReadme format
                mapped_label = map_labels_to_medreadme(label)
                if mapped_label != 'O':
                    labels[start] = f'B-{mapped_label}'
                    for i in range(start + 1, end):
                        labels[i] = f'I-{mapped_label}'
        
        # Tokenize
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=250,
            return_tensors='pt'
        )
        
        # Create label mapping for 7-class classification
        label2id = {
            'O': 0,
            'B-GOOGLE_EASY': 1, 'I-GOOGLE_EASY': 1,
            'B-GOOGLE_HARD': 2, 'I-GOOGLE_HARD': 2,
            'B-MEDICAL_NAME': 3, 'I-MEDICAL_NAME': 3,
            'B-MEDICAL_ABBR': 4, 'I-MEDICAL_ABBR': 4,
            'B-GENERAL_ABBR': 5, 'I-GENERAL_ABBR': 5,
            'B-GENERAL_COMPLEX': 6, 'I-GENERAL_COMPLEX': 6
        }
        
        # Align labels with subwords
        word_ids = encoding.word_ids()
        label_ids = []
        prev_word_id = None
        
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(label2id.get(labels[word_id], 0))
            else:
                if labels[word_id].startswith(('B-', 'I-')):
                    label = 'I-' + labels[word_id].split('-')[1]
                    label_ids.append(label2id.get(label, 0))
                else:
                    label_ids.append(label2id.get('O', 0))
            prev_word_id = word_id
        
        processed_data.append({
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'labels': torch.tensor(label_ids)
        })
    
    return processed_data

def evaluate(model, data, device):
    """Evaluate model on data"""
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for item in tqdm(data, desc='Evaluating'):
            input_ids = item['input_ids'].unsqueeze(0).to(device)
            attention_mask = item['attention_mask'].unsqueeze(0).to(device)
            labels = item['labels']
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            predictions = torch.argmax(outputs.logits, dim=2)[0]
            
            # Get actual length of sequence (ignore padding)
            mask = attention_mask[0].bool()
            length = mask.sum().item()
            
            # Only keep predictions and labels for actual tokens
            pred_seq = predictions[:length].cpu().tolist()
            label_seq = labels[:length].tolist()
            
            all_predictions.append(pred_seq)
            all_labels.append(label_seq)
    
    # Calculate metrics for 7-class classification
    token_results = calculate_metrics(all_predictions, all_labels, level='token', num_labels=7)
    entity_results = calculate_metrics(all_predictions, all_labels, level='entity', num_labels=7)
    
    return {
        'token': token_results,
        'entity': entity_results
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to PLABA data")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Base model name")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, device)
    # Initialize tokenizer with add_prefix_space=True for RoBERTa
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)
    
    # Process data
    print("Processing data...")
    data = process_plaba_data(args.data_path, tokenizer, classification_type='7-cls')
    
    # Evaluate
    print("Evaluating...")
    results = evaluate(model, data, device)
    
    # Print results
    print("\nToken-level results:")
    print(f"F1: {results['token']['overall']['f1']:.2f}")
    print(f"Precision: {results['token']['overall']['precision']:.2f}")
    print(f"Recall: {results['token']['overall']['recall']:.2f}")
    
    print("\nEntity-level results:")
    print(f"F1: {results['entity']['overall']['f1']:.2f}")
    print(f"Precision: {results['entity']['overall']['precision']:.2f}")
    print(f"Recall: {results['entity']['overall']['recall']:.2f}")
    
    # Print per-class metrics
    print("\nPer-class metrics:")
    for class_id in range(1, 7):  # Skip class 0 (O)
        class_name = {
            1: "Google Easy",
            2: "Google Hard",
            3: "Medical Name",
            4: "Medical Abbreviation",
            5: "General Abbreviation",
            6: "General Complex"
        }[class_id]
        
        print(f"\n{class_name}:")
        print(f"F1: {results['token'][f'class_{class_id}']['f1']:.2f}")
        print(f"Precision: {results['token'][f'class_{class_id}']['precision']:.2f}")
        print(f"Recall: {results['token'][f'class_{class_id}']['recall']:.2f}")

if __name__ == "__main__":
    main()