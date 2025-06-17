import json
import os
import argparse
from glob import glob

def combine_results(results_dir: str, run_id: str, output_file: str = "combined_results.json"):
    """
    Combine multiple JSON files from a directory into one JSON file.
    Args:
        results_dir: Directory containing the JSON files to combine
        run_id: ID for the run to set in all entries
        output_file: Name of the output combined JSON file
    """
    print(f"Looking for result files in {results_dir}...")
    
    # Find all JSON files in the directory
    json_files = glob(os.path.join(results_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return
    
    print(f"Found {len(json_files)} files to combine:")
    for f in sorted(json_files):
        print(f"  - {os.path.basename(f)}")
    
    # Combine all results
    all_results = []
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r') as f:
                results = json.load(f)
                for entry in results:
                    entry['run_id'] = run_id
                print(f"Loaded and updated {len(results)} entries from {os.path.basename(json_file)}")
                all_results.extend(results)
        except Exception as e:
            print(f"Error loading {json_file}: {str(e)}")
            continue
    
    # Save combined results
    try:
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSuccessfully combined and updated {len(all_results)} total entries into {output_file}")
    except Exception as e:
        print(f"Error saving combined results: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Combine split result files')
    parser.add_argument('results_dir', help='Directory containing the result files')
    parser.add_argument('--run-id', required=True, help='Run ID to set for all entries')
    parser.add_argument('--output', default='combined_results.json', help='Output file name')
    
    args = parser.parse_args()
    combine_results(args.results_dir, args.run_id, args.output)

if __name__ == "__main__":
    main() 