"""
Quick script to run evaluation
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Eval.evaluate_tt_framework import TTEvaluator

def main():
    # Default paths
    model_output = "output/test/TT_GPT4_direct_output_two_shot_CoT.json"
    gold_data = "TT-data/all_file2_combined.json"
    test_data = "TT-data/L2T_700_test_5samples.json"
    output = "Eval/evaluation_results.json"
    
    # Allow command line overrides
    if len(sys.argv) > 1:
        model_output = sys.argv[1]
    if len(sys.argv) > 2:
        gold_data = sys.argv[2]
    if len(sys.argv) > 3:
        test_data = sys.argv[3]
    if len(sys.argv) > 4:
        output = sys.argv[4]
    
    print("Starting evaluation...")
    print(f"  Model output: {model_output}")
    print(f"  Gold data: {gold_data}")
    print(f"  Test data: {test_data}")
    print(f"  Output: {output}")
    print()
    
    evaluator = TTEvaluator(gold_data)
    results = evaluator.evaluate(model_output, test_data, output)
    
    print("\nEvaluation complete!")
    return results

if __name__ == '__main__':
    main()

