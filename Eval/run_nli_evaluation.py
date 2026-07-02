"""
Script to run NLI evaluation on TT results.

This script:
1. Converts JSONL results to NLI format
2. Creates CSV files from test data
3. Provides command to run NLI evaluation
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def convert_results_for_nli(jsonl_file, test_data_file, output_dir, shared_csv_dir=None):
    """Convert JSONL results to NLI format using the conversion script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    convert_script = os.path.join(script_dir, 'convert_results_for_nli.py')
    
    cmd = [
        sys.executable, convert_script,
        jsonl_file,
        test_data_file,
        output_dir
    ]
    
    # Add shared CSV directory if provided
    if shared_csv_dir:
        cmd.append(shared_csv_dir)
    
    print(f"Running conversion script...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error during conversion:")
        print(result.stderr)
        return None
    
    print(result.stdout)
    
    # Return paths to created files
    nli_file = os.path.join(output_dir, 'nli_formatted.json')
    linking_file = os.path.join(output_dir, 'nli_linking.json')
    csv_dir = os.path.join(output_dir, 'all_csv')
    
    return {
        'nli_file': nli_file,
        'linking_file': linking_file,
        'csv_dir': csv_dir,
        'output_dir': output_dir
    }


def main():
    if len(sys.argv) < 4:
        print("Usage: python run_nli_evaluation.py <jsonl_file> <test_data> <nli_model_path>")
        print("\nArguments:")
        print("  jsonl_file: Path to JSONL results file (e.g., output/res/TT_GPT4_direct_output_one_shot.jsonl)")
        print("  test_data: Path to test data JSON file (e.g., TT-data/L2T_700.json)")
        print("  nli_model_path: Path to NLI model .pt file")
        print("\nExample:")
        print("  python run_nli_evaluation.py \\")
        print("    ../output/res/TT_GPT4_direct_output_one_shot.jsonl \\")
        print("    ../TT-data/L2T_700.json \\")
        print("    LogicNLG-master/NLI_models/model_ep4.pt")
        sys.exit(1)
    
    jsonl_file = os.path.abspath(sys.argv[1])
    test_data = os.path.abspath(sys.argv[2])
    nli_model_path = sys.argv[3]
    
    # Check if files exist
    if not os.path.exists(jsonl_file):
        print(f"Error: JSONL file not found: {jsonl_file}")
        sys.exit(1)
    
    if not os.path.exists(test_data):
        print(f"Error: Test data file not found: {test_data}")
        sys.exit(1)
    
    if not os.path.exists(nli_model_path):
        print(f"Error: NLI model not found: {nli_model_path}")
        print("\nPlease provide the correct path to your NLI model file.")
        print("The model should be a .pt file (e.g., model_ep4.pt)")
        sys.exit(1)
    
    # Create output directory
    file_basename = os.path.splitext(os.path.basename(jsonl_file))[0]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'LogicNLG-master', 'outputs', file_basename + '_nli')
    
    # Check for shared CSV directory
    shared_csv_dir = os.path.join(script_dir, 'LogicNLG-master', 'shared_csv')
    if not os.path.exists(shared_csv_dir):
        shared_csv_dir = None
        print("Note: Shared CSV directory not found. Will create CSV files in output directory.")
    else:
        print(f"Using shared CSV directory: {shared_csv_dir}")
    
    print(f"Converting results to NLI format...")
    print(f"  Input: {jsonl_file}")
    print(f"  Test data: {test_data}")
    print(f"  Output directory: {output_dir}\n")
    
    # Convert to NLI format
    result = convert_results_for_nli(jsonl_file, test_data, output_dir, shared_csv_dir)
    
    if not result:
        print("Conversion failed!")
        sys.exit(1)
    
    # Prepare NLI command
    logicnlg_dir = os.path.join(script_dir, 'LogicNLG-master')
    
    # Make paths relative to LogicNLG-master directory
    rel_nli_file = os.path.relpath(result['nli_file'], logicnlg_dir)
    rel_linking_file = os.path.relpath(result['linking_file'], logicnlg_dir)
    rel_csv_dir = os.path.relpath(result['csv_dir'], logicnlg_dir)
    
    # If model path is not absolute, make it relative to LogicNLG-master
    if os.path.isabs(nli_model_path):
        rel_model_path = os.path.relpath(nli_model_path, logicnlg_dir)
    else:
        rel_model_path = nli_model_path
    
    print(f"\n{'='*80}")
    print("NLI Evaluation Setup Complete!")
    print(f"{'='*80}\n")
    print("To run NLI evaluation, execute the following command:")
    print(f"\ncd {logicnlg_dir}")
    print(f"\npython NLI.py \\")
    print(f"  --model bert-base-multilingual-uncased \\")
    print(f"  --do_verify \\")
    print(f"  --encoding gnn \\")
    print(f"  --load_from {rel_model_path} \\")
    print(f"  --verify_file {rel_nli_file} \\")
    print(f"  --verify_linking {rel_linking_file} \\")
    print(f"  --csv_path {rel_csv_dir} \\")
    print(f"  --fp16")
    print(f"\n{'='*80}\n")
    
    # Ask if user wants to run it now
    response = input("Do you want to run NLI evaluation now? (y/n): ").strip().lower()
    if response == 'y':
        print("\nRunning NLI evaluation...")
        nli_script = os.path.join(logicnlg_dir, 'NLI.py')
        
        cmd = [
            sys.executable, nli_script,
            '--model', 'bert-base-multilingual-uncased',
            '--do_verify',
            '--encoding', 'gnn',
            '--load_from', rel_model_path,
            '--verify_file', rel_nli_file,
            '--verify_linking', rel_linking_file,
            '--csv_path', rel_csv_dir,
            '--fp16'
        ]
        
        result_cmd = subprocess.run(cmd, cwd=logicnlg_dir)
        
        if result_cmd.returncode == 0:
            print("\nNLI evaluation completed successfully!")
        else:
            print("\nNLI evaluation failed. Check the error messages above.")
    else:
        print("\nYou can run the evaluation later using the command shown above.")


if __name__ == '__main__':
    main()
