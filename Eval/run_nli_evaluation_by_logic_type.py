"""
Run NLI evaluation for each logic type separately.

This script:
1. Processes split JSONL files from output/res/split_by_logic_type/
2. Converts each logic-type-specific file to NLI format
3. Runs NLI evaluation for each logic type separately
4. Organizes outputs by model and logic type (matching parser evaluation structure)
"""

import os
import sys
import json
import subprocess
import re
from pathlib import Path
from collections import defaultdict
import argparse


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
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR during conversion:")
        print(result.stderr)
        return None
    
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


def run_nli_evaluation(nli_file, linking_file, csv_dir, nli_model_path, logicnlg_dir, output_predictions=None):
    """Run NLI evaluation and parse accuracy from output."""
    # Make paths relative to LogicNLG-master directory
    rel_nli_file = os.path.relpath(nli_file, logicnlg_dir)
    rel_linking_file = os.path.relpath(linking_file, logicnlg_dir)
    rel_csv_dir = os.path.relpath(csv_dir, logicnlg_dir)
    
    # If model path is not absolute, make it relative to LogicNLG-master
    if os.path.isabs(nli_model_path):
        rel_model_path = os.path.relpath(nli_model_path, logicnlg_dir)
    else:
        rel_model_path = nli_model_path
    
    # Handle output_predictions path
    rel_output_predictions = None
    if output_predictions:
        if os.path.isabs(output_predictions):
            rel_output_predictions = os.path.relpath(output_predictions, logicnlg_dir)
        else:
            rel_output_predictions = output_predictions
    
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
    
    # Add output_predictions if specified
    if rel_output_predictions:
        cmd.extend(['--output_predictions', rel_output_predictions])
    
    result = subprocess.run(cmd, cwd=logicnlg_dir, capture_output=True, text=True)
    
    # Parse accuracy from output
    accuracy = None
    detailed_results = None
    if result.returncode == 0:
        # Look for "the final accuracy is X" in stdout
        for line in result.stdout.split('\n'):
            if 'the final accuracy is' in line.lower():
                try:
                    # Extract number from line like "the final accuracy is 0.85"
                    match = re.search(r'the final accuracy is\s+([\d.]+)', line, re.IGNORECASE)
                    if match:
                        accuracy = float(match.group(1))
                        break
                except (ValueError, AttributeError):
                    pass
        
        # Load detailed results if output file was created
        if output_predictions and os.path.exists(output_predictions):
            try:
                with open(output_predictions, 'r', encoding='utf-8') as f:
                    detailed_results = json.load(f)
            except Exception as e:
                print(f"  WARNING: Could not load detailed results: {e}")
    
    return result.returncode == 0, result.stdout, result.stderr, accuracy, detailed_results


def process_single_logic_type(jsonl_file, test_data, nli_model_path, shared_csv_dir, output_base_dir, skip_existing=False):
    """Process a single logic-type-specific JSONL file through NLI evaluation."""
    jsonl_path = Path(jsonl_file)
    
    if not jsonl_path.exists():
        print(f"ERROR: JSONL file not found: {jsonl_file}")
        return False, None, None
    
    # Extract model name and logic type from filename
    # Format: TT_model_name_logic_type.jsonl
    # Example: TT_GPT4_direct_output_one_shot_aggregation.jsonl
    filename = jsonl_path.stem  # Without .jsonl extension
    
    # Logic types to check
    logic_types = ['aggregation', 'comparative', 'count', 'majority', 'ordinal', 'superlative', 'unique']
    logic_type = None
    for lt in logic_types:
        if filename.endswith(f'_{lt}'):
            logic_type = lt
            model_name = filename[:-len(f'_{lt}')]
            break
    
    if logic_type is None:
        print(f"ERROR: Cannot identify logic type from filename: {filename}")
        return False, None, None
    
    # Create output directory: output_base_dir/model_name_logic_type
    # This matches the structure expected by compare_nli_with_parser_formatted.py
    output_dir = os.path.join(output_base_dir, f"{model_name}_{logic_type}")
    
    # Check if already processed
    nli_result_file = os.path.join(output_dir, 'nli_result.json')
    if skip_existing and os.path.exists(nli_result_file):
        print(f"  SKIP: Already processed (found {nli_result_file})")
        # Try to load accuracy from existing result file
        accuracy = None
        try:
            with open(nli_result_file, 'r') as f:
                existing_result = json.load(f)
                accuracy = existing_result.get('accuracy')
        except:
            pass
        return True, logic_type, accuracy
    
    print(f"  Model: {model_name}")
    print(f"  Logic Type: {logic_type}")
    print(f"  Output: {output_dir}")
    
    # Step 1: Convert to NLI format
    print(f"  [1/2] Converting to NLI format...")
    result = convert_results_for_nli(str(jsonl_file), test_data, output_dir, shared_csv_dir)
    
    if not result:
        print(f"  ERROR: Conversion failed")
        return False, logic_type, None
    
    # Check if conversion created valid files
    if not os.path.exists(result['nli_file']):
        print(f"  ERROR: NLI formatted file not created")
        return False, logic_type, None
    
    # Check if there are any claims (file might be empty if all claims were filtered)
    with open(result['nli_file'], 'r', encoding='utf-8') as f:
        nli_data = json.load(f)
    
    if not nli_data:
        print(f"  WARNING: No valid claims found (all were empty/invalid)")
        print(f"  Skipping NLI evaluation...")
        return True, logic_type, None  # Not an error, just no data to process
    
    print(f"  ✓ Conversion complete: {len(nli_data)} tables with claims")
    
    # Step 2: Run NLI evaluation
    print(f"  [2/2] Running NLI evaluation...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logicnlg_dir = os.path.join(script_dir, 'LogicNLG-master')
    
    # Set up output file for detailed predictions
    detailed_predictions_file = os.path.join(output_dir, 'nli_detailed_predictions.json')
    
    success, stdout, stderr, accuracy, detailed_results = run_nli_evaluation(
        result['nli_file'],
        result['linking_file'],
        result['csv_dir'],
        nli_model_path,
        logicnlg_dir,
        output_predictions=detailed_predictions_file
    )
    
    if success:
        if accuracy is not None:
            print(f"  ✓ NLI evaluation completed successfully (Accuracy: {accuracy:.4f})")
        else:
            print(f"  ✓ NLI evaluation completed successfully")
        
        # Save per-sample results (all samples, not just refused)
        per_sample_file = os.path.join(output_dir, 'nli_per_sample_results.json')
        if detailed_results and 'detailed_results' in detailed_results:
            # Extract all per-sample results
            all_per_sample = []
            for table_id, table_data in detailed_results['detailed_results'].items():
                for result_entry in table_data.get('results', []):
                    all_per_sample.append({
                        'table_id': table_id,
                        'claim': result_entry.get('statement', ''),
                        'prediction': result_entry.get('prediction', 'Unknown'),
                        'is_entailed': result_entry.get('prediction_label', 0) == 1,
                        'entailed_probability': result_entry.get('entailed_probability', 0.0),
                        'refused_probability': result_entry.get('refused_probability', 0.0)
                    })
            
            per_sample_data = {
                'model': model_name,
                'logic_type': logic_type,
                'accuracy': accuracy if accuracy is not None else 0.0,
                'total_samples': len(all_per_sample),
                'per_sample_results': all_per_sample
            }
            
            with open(per_sample_file, 'w', encoding='utf-8') as f:
                json.dump(per_sample_data, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Per-sample results saved to: {per_sample_file}")
            print(f"    Total samples: {len(all_per_sample)}")
        
        # Also save refused samples by table_id (for backward compatibility)
        refused_samples_file = os.path.join(output_dir, 'nli_refused_samples.json')
        if detailed_results and 'refused_samples_by_table_id' in detailed_results:
            refused_samples = detailed_results['refused_samples_by_table_id']
            # Convert to simpler format: {table_id: [list of refused claims]}
            simplified_refused = {}
            for table_id, claims in refused_samples.items():
                simplified_refused[table_id] = [
                    {
                        'statement': claim['statement'],
                        'refused_probability': claim['refused_probability'],
                        'entailed_probability': claim['entailed_probability']
                    }
                    for claim in claims
                ]
            
            with open(refused_samples_file, 'w', encoding='utf-8') as f:
                json.dump(simplified_refused, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Refused samples saved to: {refused_samples_file}")
            print(f"    Total tables with refused samples: {len(simplified_refused)}")
            total_refused = sum(len(claims) for claims in simplified_refused.values())
            print(f"    Total refused claims: {total_refused}")
        
        # Save a marker file to indicate completion
        result_data = {
            'status': 'completed',
            'tables': len(nli_data),
            'model': model_name,
            'logic_type': logic_type
        }
        if accuracy is not None:
            result_data['accuracy'] = accuracy
        if detailed_results:
            result_data['total_samples'] = detailed_results.get('total_samples')
            result_data['entailed_samples'] = detailed_results.get('entailed_samples')
            result_data['refused_samples'] = detailed_results.get('refused_samples')
        with open(nli_result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2)
    else:
        # Evaluation failed - show error details
        print(f"  ✗ NLI evaluation failed")
        if stderr:
            print(f"  Error output (last 500 chars):")
            print(f"  {stderr[-500:]}")
        if stdout:
            # Look for any error messages in stdout
            error_lines = [line for line in stdout.split('\n') if 'error' in line.lower() or 'exception' in line.lower() or 'traceback' in line.lower()]
            if error_lines:
                print(f"  Error messages found:")
                for line in error_lines[-5:]:  # Show last 5 error lines
                    print(f"    {line}")
        return False, logic_type, None


def process_all_logic_types(split_base_dir, test_data, nli_model_path, shared_csv_dir, output_base_dir, skip_existing=False):
    """Process all models and logic types from split_by_logic_type directory."""
    split_path = Path(split_base_dir)
    
    if not split_path.exists():
        print(f"ERROR: Split base directory does not exist: {split_base_dir}")
        return
    
    # Find all model directories
    model_dirs = [d for d in split_path.iterdir() if d.is_dir()]
    
    if not model_dirs:
        print(f"ERROR: No model directories found in {split_base_dir}")
        return
    
    # Logic types to process
    logic_types = ['aggregation', 'comparative', 'count', 'majority', 'ordinal', 'superlative', 'unique']
    
    results = defaultdict(dict)
    total_files = 0
    successful = 0
    failed = 0
    
    print(f"\n{'='*80}")
    print(f"NLI Evaluation by Logic Type")
    print(f"{'='*80}")
    print(f"Split base directory: {split_base_dir}")
    print(f"Test data: {test_data}")
    print(f"NLI model: {nli_model_path}")
    print(f"Output base directory: {output_base_dir}")
    print(f"Found {len(model_dirs)} model(s) to process")
    print(f"{'='*80}\n")
    
    # Process each model
    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        # Process each logic type for this model
        for logic_type in logic_types:
            jsonl_file = model_dir / f"{model_name}_{logic_type}.jsonl"
            
            if not jsonl_file.exists():
                print(f"\n  Logic Type: {logic_type}")
                print(f"  SKIP: File not found: {jsonl_file.name}")
                results[model_name][logic_type] = {"status": "skipped", "reason": "file_not_found"}
                continue
            
            total_files += 1
            print(f"\n  Logic Type: {logic_type}")
            success, detected_logic_type, accuracy = process_single_logic_type(
                str(jsonl_file),
                test_data,
                nli_model_path,
                shared_csv_dir,
                output_base_dir,
                skip_existing=skip_existing
            )
            
            if success:
                successful += 1
                result_entry = {"status": "success"}
                if accuracy is not None:
                    result_entry["accuracy"] = accuracy
                results[model_name][logic_type] = result_entry
            else:
                failed += 1
                results[model_name][logic_type] = {"status": "failed"}
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Summary")
    print(f"{'='*80}")
    print(f"Total files processed: {total_files}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {total_files - successful - failed}")
    
    print(f"\nResults by model:")
    for model_name in sorted(results.keys()):
        model_results = results[model_name]
        model_success = sum(1 for r in model_results.values() if r.get('status') == 'success')
        model_total = len(model_results)
        
        # Calculate average accuracy for this model
        accuracies = [r.get('accuracy') for r in model_results.values() if r.get('status') == 'success' and 'accuracy' in r]
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else None
        
        acc_str = f" (Avg Accuracy: {avg_accuracy:.4f})" if avg_accuracy is not None else ""
        print(f"  {model_name}: {model_success}/{model_total} logic types successful{acc_str}")
        
        # Print per-logic-type accuracy
        for lt in sorted(model_results.keys()):
            lt_result = model_results[lt]
            if lt_result.get('status') == 'success' and 'accuracy' in lt_result:
                print(f"    - {lt}: {lt_result['accuracy']:.4f}")
    
    print(f"{'='*80}\n")
    
    # Save comprehensive summary
    summary_file = os.path.join(output_base_dir, 'nli_evaluation_summary.json')
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Calculate overall statistics
    all_accuracies = []
    for model_results in results.values():
        for lt_result in model_results.values():
            if lt_result.get('status') == 'success' and 'accuracy' in lt_result:
                all_accuracies.append(lt_result['accuracy'])
    
    summary_data = {
        'total_files': total_files,
        'successful': successful,
        'failed': failed,
        'skipped': total_files - successful - failed,
        'overall_statistics': {
            'average_accuracy': sum(all_accuracies) / len(all_accuracies) if all_accuracies else None,
            'min_accuracy': min(all_accuracies) if all_accuracies else None,
            'max_accuracy': max(all_accuracies) if all_accuracies else None,
            'total_with_accuracy': len(all_accuracies)
        },
        'results_by_model': {}
    }
    
    # Organize results by model with statistics
    for model_name in sorted(results.keys()):
        model_results = results[model_name]
        model_accuracies = [r.get('accuracy') for r in model_results.values() 
                           if r.get('status') == 'success' and 'accuracy' in r]
        
        summary_data['results_by_model'][model_name] = {
            'logic_types': model_results,
            'statistics': {
                'successful_logic_types': sum(1 for r in model_results.values() if r.get('status') == 'success'),
                'total_logic_types': len(model_results),
                'average_accuracy': sum(model_accuracies) / len(model_accuracies) if model_accuracies else None,
                'min_accuracy': min(model_accuracies) if model_accuracies else None,
                'max_accuracy': max(model_accuracies) if model_accuracies else None
            }
        }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run NLI evaluation separately for each logic type",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all models and logic types
  python run_nli_evaluation_by_logic_type.py output/res/split_by_logic_type TT-data/L2T_700.json LogicNLG-master/NLI_models/model_ep4.pt
  
  # Skip already processed files
  python run_nli_evaluation_by_logic_type.py output/res/split_by_logic_type TT-data/L2T_700.json LogicNLG-master/NLI_models/model_ep4.pt --skip-existing
  
  # Custom output directory
  python run_nli_evaluation_by_logic_type.py output/res/split_by_logic_type TT-data/L2T_700.json LogicNLG-master/NLI_models/model_ep4.pt --output-dir Eval/LogicNLG-master/outputs/logic_type_evaluation_NLI
        """
    )
    parser.add_argument('split_base_dir', type=str,
                        help='Base directory containing split JSONL files (e.g., output/res/split_by_logic_type)')
    parser.add_argument('test_data', type=str,
                        help='Path to test data JSON file (e.g., TT-data/L2T_700.json)')
    parser.add_argument('nli_model_path', type=str,
                        help='Path to NLI model .pt file (e.g., LogicNLG-master/NLI_models/model_ep4.pt)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Base output directory (default: Eval/LogicNLG-master/outputs/logic_type_evaluation_NLI)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip files that have already been processed')
    parser.add_argument('--shared-csv-dir', type=str, default=None,
                        help='Shared CSV directory (default: LogicNLG-master/shared_csv if exists)')
    
    args = parser.parse_args()
    
    # Convert paths to absolute
    split_base_dir = os.path.abspath(args.split_base_dir)
    test_data = os.path.abspath(args.test_data)
    nli_model_path = os.path.abspath(args.nli_model_path)
    
    # Check if files exist
    if not os.path.exists(split_base_dir):
        print(f"ERROR: Split base directory not found: {split_base_dir}")
        sys.exit(1)
    
    if not os.path.exists(test_data):
        print(f"ERROR: Test data file not found: {test_data}")
        sys.exit(1)
    
    if not os.path.exists(nli_model_path):
        print(f"ERROR: NLI model not found: {nli_model_path}")
        sys.exit(1)
    
    # Setup output directory
    if args.output_dir:
        output_base_dir = os.path.abspath(args.output_dir)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_base_dir = os.path.join(script_dir, 'LogicNLG-master', 'outputs', 'logic_type_evaluation_NLI')
    
    # Setup shared CSV directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.shared_csv_dir:
        shared_csv_dir = os.path.abspath(args.shared_csv_dir)
    else:
        shared_csv_dir = os.path.join(script_dir, 'LogicNLG-master', 'shared_csv')
        if not os.path.exists(shared_csv_dir):
            shared_csv_dir = None
            print("Note: Shared CSV directory not found. Will create CSV files in output directories.")
        else:
            print(f"Using shared CSV directory: {shared_csv_dir}")
    
    process_all_logic_types(
        split_base_dir,
        test_data,
        nli_model_path,
        shared_csv_dir,
        output_base_dir,
        skip_existing=args.skip_existing
    )


if __name__ == '__main__':
    main()
