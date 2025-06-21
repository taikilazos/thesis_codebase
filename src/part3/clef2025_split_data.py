import json
import os
import re
import argparse
from math import ceil

def split_data(input_file, num_parts=4):
    # Extract task number from filename
    match = re.search(r'task(\d+)', input_file)
    if not match:
        raise ValueError("Input filename must contain 'taskXX' where XX is the task number")
    task_num = match.group(1)
    
    # Determine output directory based on task number
    if task_num == '11':
        split_dir = "splits_snt"
    elif task_num == '12':
        split_dir = "splits_doc"
    else:
        raise ValueError(f"Unexpected task number: {task_num}. Only task11 and task12 are supported.")
    
    # Create output directory if it doesn't exist
    os.makedirs(f"data/CLEF2025/{split_dir}", exist_ok=True)
    
    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Calculate split sizes
    total_size = len(data)
    base_size = ceil(total_size / num_parts)
    
    # Split and save data
    for i in range(num_parts):
        start_idx = i * base_size
        end_idx = min((i + 1) * base_size, total_size)
        split_data = data[start_idx:end_idx]
        
        output_file = f"data/CLEF2025/{split_dir}/simpletext25_task{task_num}_test_part{i+1}.json"
        with open(output_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"Part {i+1}: {len(split_data)} entries saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split a CLEF2025 dataset into multiple parts')
    parser.add_argument('input_file', help='Path to the input JSON file (e.g., data/CLEF2025/simpletext25_task11_test.json)')
    parser.add_argument('--parts', type=int, default=4, help='Number of parts to split the data into (default: 4)')
    
    args = parser.parse_args()
    split_data(args.input_file, args.parts) 