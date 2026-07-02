"""
Run TAPEX evaluation for Task 4 (nl_claim_new) for each logic type separately.

This script:
1. Processes split JSONL files from output/res/round1_split_by_logic_type/
2. Extracts nl_claim_new from task4_more_claims.nl_claim_new
3. Converts each logic-type-specific file to parser format
4. Runs TAPEX evaluation for each logic type separately
5. Organizes outputs by model and logic type
"""

import os
import sys
import json
import shutil
from pathlib import Path
from collections import defaultdict
import argparse

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOG_DIR = os.path.join(SCRIPT_DIR, '..', 'PLOG')


def convert_to_parser_format_task4(jsonl_file, output_dir):
    """Convert a single JSONL file to parser format for Task 4."""
    sys.path.insert(0, SCRIPT_DIR)
    from convert_jsonl_to_parser_format_task4 import convert_jsonl_to_parser_format_task4
    
    jsonl_path = Path(jsonl_file)
    jsonl_name = jsonl_path.stem
    
    # Create output file path
    output_file = Path(output_dir) / f"{jsonl_name}_parser_formatted.json"
    
    try:
        # Convert the JSONL file directly
        formatted_data, id_mapping = convert_jsonl_to_parser_format_task4(str(jsonl_file), str(output_file))
        
        if not formatted_data:
            print(f"  Warning: No data was converted from {jsonl_file}")
            return None, None
        
        print(f"  Converted {len(formatted_data)} tables, {sum(len(claims) for claims in formatted_data.values())} claims")
        
        # Save ID mapping
        mapping_file = output_file.parent / f"{output_file.stem}_id_mapping.json"
        id_mapping_str = {f"{k[0]}_{k[1]}": v for k, v in id_mapping.items()}
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(id_mapping_str, f, indent=2, ensure_ascii=False)
        
        return str(output_file) if output_file.exists() else None, id_mapping
        
    except Exception as e:
        print(f"  ERROR during parser format conversion: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def run_tapex_evaluation(parser_file, csv_dir, tapex_model_path, device=-1, batch_size=12, output_dir=None, short_affix=None):
    """Run TAPEX evaluation using the evaluation function directly."""
    script_path = os.path.join(PLOG_DIR, 'scripts', 'run_tapex_evaluation_by_logic_type.py')
    
    if not os.path.exists(script_path):
        print(f"ERROR: TAPEX evaluation script not found: {script_path}")
        return None, None
    
    # Import the evaluation function
    sys.path.insert(0, os.path.join(PLOG_DIR, 'scripts'))
    from run_tapex_evaluation_by_logic_type import evaluate_single_file_with_tapex
    
    # Use output_dir if provided, otherwise create temp
    if output_dir:
        temp_base = output_dir
        os.makedirs(temp_base, exist_ok=True)
    else:
        temp_base = os.path.join(os.path.dirname(parser_file), 'temp_tapex')
        os.makedirs(temp_base, exist_ok=True)
    
    # Fix csv_dir: TAPEX script appends 'all_csv/' to data_dir
    # If user passed the all_csv directory itself, use its parent
    csv_dir_path = Path(csv_dir).resolve()
    if csv_dir_path.name == 'all_csv':
        tapex_data_dir = str(csv_dir_path.parent)
    else:
        tapex_data_dir = str(csv_dir_path)
    
    # Use a short affix to avoid Windows path length issues
    # If not provided, create a short one from the parser file name
    if short_affix is None:
        # Use just the logic type or a hash
        parser_stem = Path(parser_file).stem.replace('_parser_formatted', '')
        # Try to extract just the logic type part
        logic_types = ['aggregation', 'comparative', 'count', 'majority', 'ordinal', 'superlative', 'unique']
        logic_type_short = {
            'aggregation': 'agg',
            'comparative': 'comp',
            'count': 'cnt',
            'majority': 'maj',
            'ordinal': 'ord',
            'superlative': 'sup',
            'unique': 'uni'
        }
        for lt in logic_types:
            if parser_stem.endswith(f'_{lt}'):
                short_affix = f"t4_{logic_type_short.get(lt, lt[:4])}"  # t4 = task4, very short logic type
                break
        if short_affix is None:
            # Fallback: use first 10 chars
            short_affix = parser_stem[:10] if len(parser_stem) > 10 else parser_stem
    
    result = evaluate_single_file_with_tapex(
        test_file=parser_file,
        data_dir=tapex_data_dir,
        output_dir=temp_base,
        affix=short_affix,
        model_path=tapex_model_path,
        device=device,
        batch_size=batch_size,
        local_files_only=True
    )
    
    if result is None:
        return None, None
    
    accuracy = result.get('acc')  # TAPEX returns 'acc' not 'accuracy'
    detailed_results = result.get('detailed_results')
    
    # Also try to load per-sample results from the prediction JSON file
    # TAPEX creates nested structure: output_dir/tapex_{affix}/{affix}/{affix}_predict_results.json
    prediction_json_paths = [
        Path(temp_base) / f'tapex_{short_affix}' / short_affix / f"{short_affix}_predict_results.json",  # Standard path
        Path(temp_base) / short_affix / f"{short_affix}_predict_results.json",  # Without tapex_ prefix
        Path(temp_base) / f"{short_affix}_predict_results.json",  # Direct in output_dir
        Path(temp_base) / f'tapex_{short_affix}' / f"{short_affix}_predict_results.json",  # One level less
    ]
    
    prediction_json = None
    for path in prediction_json_paths:
        if path.exists():
            prediction_json = path
            break
    
    if prediction_json:
        try:
            with open(prediction_json, 'r', encoding='utf-8') as f:
                prediction_data = json.load(f)
                # Update accuracy if available
                if 'accuracy' in prediction_data:
                    accuracy = prediction_data.get('accuracy')
                # Get per-sample results if available
                if 'per_sample_results' in prediction_data:
                    detailed_results = prediction_data['per_sample_results']
        except Exception as e:
            print(f"Warning: Could not load prediction JSON from {prediction_json}: {e}")
    else:
        # Debug: list what files actually exist
        print(f"Warning: Could not find prediction JSON file. Checked paths:")
        for path in prediction_json_paths:
            print(f"  - {path} (exists: {path.exists()})")
        # List directory contents for debugging
        if Path(temp_base).exists():
            print(f"  Contents of {temp_base}:")
            try:
                for item in sorted(Path(temp_base).iterdir()):
                    item_type = 'dir' if item.is_dir() else 'file'
                    print(f"    - {item.name} ({item_type})")
                    # If it's a directory, list its contents too
                    if item.is_dir():
                        try:
                            for subitem in sorted(item.iterdir()):
                                subitem_type = 'dir' if subitem.is_dir() else 'file'
                                print(f"      - {subitem.name} ({subitem_type})")
                        except Exception:
                            pass
            except Exception as e:
                print(f"    Could not list directory: {e}")
    
    return accuracy, detailed_results


def process_single_logic_type(jsonl_file, tapex_model_path, csv_dir, output_base_dir, device=-1, batch_size=12, skip_existing=False):
    """Process a single logic-type-specific JSONL file through TAPEX evaluation for Task 4."""
    jsonl_path = Path(jsonl_file)
    
    if not jsonl_path.exists():
        print(f"ERROR: JSONL file not found: {jsonl_file}")
        return False, None, None
    
    # Extract model name and logic type from filename
    # Format: TT_model_name_logic_type.jsonl
    # Example: TT_deepseek-v3.2_output_one_shot_with_operations_nested_aggregation.jsonl
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
    
    # Create output directory with shorter names to avoid Windows path length issues
    # Extract a shorter model name (e.g., "deepseek-v3.2" from "TT_deepseek-v3.2_output_one_shot_with_operations_nested")
    short_model_name = model_name
    # Remove "TT_" prefix if present
    if short_model_name.startswith('TT_'):
        short_model_name = short_model_name[3:]
    # Extract just the model identifier part (before first underscore after model name)
    # Examples:
    # "deepseek-v3.2_output_one_shot_with_operations_nested" -> "deepseek-v3.2"
    # "qwen3-32b_output_one_shot_with_operations_nested" -> "qwen3-32b"
    # "GPT4_direct_output_one_shot" -> "GPT4"
    parts = short_model_name.split('_')
    # Try to find a version number pattern (e.g., v3.2, 3.2, 32b, etc.)
    short_parts = []
    for part in parts:
        if any(char.isdigit() for char in part) or part.lower() in ['gpt4', 'gpt5']:
            short_parts.append(part)
            break
        if part and not part.lower() in ['output', 'direct', 'one', 'shot', 'with', 'operations', 'nested']:
            short_parts.append(part)
            if len(short_parts) >= 2:  # Take at most 2 parts
                break
    
    if short_parts:
        short_model = '-'.join(short_parts)
    else:
        # Fallback: use first part
        short_model = parts[0] if parts else short_model_name[:20]
    
    output_dir = os.path.join(output_base_dir, f"{short_model}-{logic_type}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full model name for reference
    model_mapping_file = os.path.join(output_dir, '_model_name.txt')
    with open(model_mapping_file, 'w', encoding='utf-8') as f:
        f.write(model_name)
    
    # Check if already processed
    tapex_result_file = os.path.join(output_dir, 'tapex_result.json')
    if skip_existing and os.path.exists(tapex_result_file):
        print(f"  SKIP: Already processed (found {tapex_result_file})")
        # Try to load accuracy from existing result file
        accuracy = None
        try:
            with open(tapex_result_file, 'r') as f:
                existing_result = json.load(f)
                accuracy = existing_result.get('accuracy')
        except:
            pass
        return True, logic_type, accuracy
    
    print(f"  Model: {model_name}")
    print(f"  Logic Type: {logic_type}")
    print(f"  Output: {output_dir}")
    
    # Step 1: Convert to parser format
    print(f"  [1/2] Converting to parser format (Task 4: nl_claim_new)...")
    parser_file, id_mapping = convert_to_parser_format_task4(str(jsonl_file), output_dir)
    
    if not parser_file:
        print(f"  ERROR: Parser format conversion failed")
        return False, logic_type, None
    
    print(f"  ✓ Parser format conversion complete: {parser_file}")
    
    # Step 2: Run TAPEX evaluation
    print(f"  [2/2] Running TAPEX evaluation...")
    tapex_output_dir = os.path.join(output_dir, 'tapex_out')
    # Use very short affix to avoid path length issues
    # Map logic types to short codes
    logic_type_short = {
        'aggregation': 'agg',
        'comparative': 'comp',
        'count': 'cnt',
        'majority': 'maj',
        'ordinal': 'ord',
        'superlative': 'sup',
        'unique': 'uni'
    }
    short_affix = f"t4_{logic_type_short.get(logic_type, logic_type[:4])}"
    tapex_acc, detailed_results = run_tapex_evaluation(
        parser_file, csv_dir, tapex_model_path, device, batch_size, tapex_output_dir, short_affix=short_affix
    )
    
    if tapex_acc is not None:
        print(f"  ✓ TAPEX evaluation completed successfully (Accuracy: {tapex_acc:.4f})")
        
        # Save per-sample results
        per_sample_file = os.path.join(output_dir, 'tapex_per_sample_results.json')
        if detailed_results:
            # Check if detailed_results is already in the right format
            if isinstance(detailed_results, dict) and 'per_sample_results' in detailed_results:
                # Already in the right format
                per_sample_data = detailed_results
            elif isinstance(detailed_results, list):
                # List of per-sample results
                per_sample_data = {
                    'model': model_name,
                    'logic_type': logic_type,
                    'task': 'task4_nl_claim_new',
                    'accuracy': tapex_acc,
                    'total_samples': len(detailed_results),
                    'per_sample_results': detailed_results
                }
            else:
                # Save as-is
                per_sample_data = {
                    'model': model_name,
                    'logic_type': logic_type,
                    'task': 'task4_nl_claim_new',
                    'accuracy': tapex_acc,
                    'detailed_results': detailed_results
                }
            
            with open(per_sample_file, 'w', encoding='utf-8') as f:
                json.dump(per_sample_data, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Per-sample results saved to: {per_sample_file}")
        
        # Save a marker file to indicate completion
        result_data = {
            'status': 'completed',
            'model': model_name,
            'logic_type': logic_type,
            'task': 'task4_nl_claim_new',
            'accuracy': tapex_acc
        }
        if detailed_results:
            if isinstance(detailed_results, list):
                result_data['total_samples'] = len(detailed_results)
            elif isinstance(detailed_results, dict) and 'total_samples' in detailed_results:
                result_data['total_samples'] = detailed_results.get('total_samples')
        
        with open(tapex_result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2)
        
        return True, logic_type, tapex_acc
    else:
        print(f"  ✗ TAPEX evaluation failed")
        return False, logic_type, None


def process_all_logic_types(split_base_dir, tapex_model_path, csv_dir, output_base_dir, device=-1, batch_size=12, skip_existing=False):
    """Process all models and logic types from round1_split_by_logic_type directory."""
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
    print(f"TAPEX Evaluation by Logic Type - Task 4 (nl_claim_new)")
    print(f"{'='*80}")
    print(f"Split base directory: {split_base_dir}")
    print(f"TAPEX model: {tapex_model_path}")
    print(f"CSV directory: {csv_dir}")
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
                tapex_model_path,
                csv_dir,
                output_base_dir,
                device,
                batch_size,
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
    summary_file = os.path.join(output_base_dir, 'tapex_evaluation_summary_task4.json')
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Calculate overall statistics
    all_accuracies = []
    for model_results in results.values():
        for lt_result in model_results.values():
            if lt_result.get('status') == 'success' and 'accuracy' in lt_result:
                all_accuracies.append(lt_result['accuracy'])
    
    summary_data = {
        'task': 'task4_nl_claim_new',
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
        description="Run TAPEX evaluation for Task 4 (nl_claim_new) separately for each logic type",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all models and logic types
  python run_tapex_evaluation_by_logic_type_task4.py output/res/round1_split_by_logic_type D:\\AT2T\\TT_design\\PLOG\\models\\tapex-large-finetuned-tabfact TT-data/all_csv output/res/tapex_task4
  
  # Skip already processed files
  python run_tapex_evaluation_by_logic_type_task4.py output/res/round1_split_by_logic_type D:\\AT2T\\TT_design\\PLOG\\models\\tapex-large-finetuned-tabfact TT-data/all_csv output/res/tapex_task4 --skip-existing
  
  # Custom device and batch size
  python run_tapex_evaluation_by_logic_type_task4.py output/res/round1_split_by_logic_type D:\\AT2T\\TT_design\\PLOG\\models\\tapex-large-finetuned-tabfact TT-data/all_csv output/res/tapex_task4 --device -1 --batch-size 8
        """
    )
    parser.add_argument('split_base_dir', type=str,
                        help='Base directory containing split JSONL files (e.g., output/res/round1_split_by_logic_type)')
    parser.add_argument('tapex_model_path', type=str,
                        help='Path to TAPEX model directory')
    parser.add_argument('csv_dir', type=str,
                        help='Directory containing CSV files')
    parser.add_argument('output_base_dir', type=str,
                        help='Base output directory for TAPEX results')
    parser.add_argument('--device', type=int, default=-1,
                        help='CUDA device ID (0, 1, etc.) or -1 for CPU (default: -1)')
    parser.add_argument('--batch-size', type=int, default=12,
                        help='Batch size for TAPEX evaluation (default: 12)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip files that have already been processed')
    
    args = parser.parse_args()
    
    # Convert paths to absolute
    split_base_dir = os.path.abspath(args.split_base_dir)
    tapex_model_path = os.path.abspath(args.tapex_model_path)
    csv_dir = os.path.abspath(args.csv_dir)
    output_base_dir = os.path.abspath(args.output_base_dir)
    
    # Check if files exist
    if not os.path.exists(split_base_dir):
        print(f"ERROR: Split base directory not found: {split_base_dir}")
        sys.exit(1)
    
    if not os.path.exists(tapex_model_path):
        print(f"ERROR: TAPEX model not found: {tapex_model_path}")
        sys.exit(1)
    
    if not os.path.exists(csv_dir):
        print(f"ERROR: CSV directory not found: {csv_dir}")
        sys.exit(1)
    
    process_all_logic_types(
        split_base_dir,
        tapex_model_path,
        csv_dir,
        output_base_dir,
        device=args.device,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing
    )


if __name__ == '__main__':
    main()
