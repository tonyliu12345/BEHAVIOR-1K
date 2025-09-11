"""
Batch evaluation script for multiple model outputs.
This script processes all model output files in the evaluation folder and generates evaluation results.
"""

import os
import sys
import glob
from pathlib import Path
import json

# Add parent directories to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'OmniGibson'))

try:
    from EmbodiedVLM.heuristics.evaluators import OrderingEvaluator
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def extract_model_name_from_filename(filename):
    """
    Extract model name from filename like 'behavior_eqa_ordering_model_name.jsonl'
    """
    basename = os.path.basename(filename)
    if basename.startswith('behavior_eqa_ordering_') and basename.endswith('.jsonl'):
        model_name = basename[len('behavior_eqa_ordering_'):-len('.jsonl')]
        return model_name
    return None

def reset_evaluator_state(evaluator):
    """
    Reset the evaluator state to avoid cross-model contamination
    """
    evaluator.eval_results.clear()
    evaluator.skipped_items.clear()
    evaluator.wrong_case_signatures.clear()
    # Note: We keep _verifiers_cache as it contains task-specific verifiers that can be reused

def evaluate_single_model(evaluator, jsonl_path, model_name, output_dir):
    """
    Evaluate a single model and save results
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING MODEL: {model_name}")
    print(f"{'='*60}")
    print(f"Input file: {jsonl_path}")
    
    try:
        # Clear previous evaluation results to avoid caching issues
        reset_evaluator_state(evaluator)
        
        # Run the evaluation
        evaluator.evaluate(jsonl_path, analyze_wrong_case=True)
        
        # Generate reports
        print(f"\nGenerating reports for {model_name}...")
        
        # 1. Overall performance report
        overall_report = evaluator.report_overall_score()
        print(f"\n1. OVERALL PERFORMANCE for {model_name}:")
        
        if 'overall' in overall_report:
            overall_stats = overall_report['overall']
            print(f"   Total Evaluated: {overall_stats.get('count', 0)}")
            print(f"   Overall Accuracy: {overall_stats.get('overall_accuracy', 0.0)*100:.2f}%")
            print(f"   Exact Match Accuracy: {overall_stats.get('exact_match_accuracy', 0.0)*100:.2f}%")
            print(f"   Semantic Match Accuracy: {overall_stats.get('semantic_match_accuracy', 0.0)*100:.2f}%")
            print(f"   Average Pair Correct Ratio: {overall_stats.get('avg_pair_correct_ratio', 0.0)*100:.2f}%")
        
        if 'forward_dynamics' in overall_report:
            fwd_stats = overall_report['forward_dynamics']
            print(f"   Forward Dynamics Accuracy: {fwd_stats.get('overall_accuracy', 0.0):.3f} "
                  f"({fwd_stats.get('count', 0)} samples)")
        
        if 'inverse_dynamics' in overall_report:
            inv_stats = overall_report['inverse_dynamics']
            print(f"   Inverse Dynamics Accuracy: {inv_stats.get('overall_accuracy', 0.0):.3f} "
                  f"({inv_stats.get('count', 0)} samples)")
        
        # 2. Performance by task type
        task_type_report = evaluator.report_by_task_type()
        print(f"\n2. PERFORMANCE BY TASK TYPE for {model_name}:")
        for question_type, step_results in task_type_report.items():
            print(f"   {question_type}:")
            for step_num, stats in step_results.items():
                accuracy = stats.get('overall_accuracy', 0.0)
                count = stats.get('count', 0)
                pair_correct = stats.get('avg_pair_correct_ratio', 0.0)
                print(f"     {step_num} steps: accuracy: {accuracy*100:.2f}% "
                      f"pair correct ratio: {pair_correct*100:.2f}% "
                      f"({count} samples)")
        
        # 3. Save results to JSON file
        output_file = os.path.join(output_dir, f"evaluation_results_{model_name}.json")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(task_type_report, f, indent=2)
        
        print(f"\n3. Results saved to: {output_file}")
        
        return {
            'model_name': model_name,
            'status': 'success',
            'output_file': output_file,
            'overall_stats': overall_report.get('overall', {}),
            'task_type_count': len(task_type_report)
        }
        
    except Exception as e:
        print(f"ERROR evaluating {model_name}: {str(e)}")
        return {
            'model_name': model_name,
            'status': 'error',
            'error': str(e)
        }

def main():
    """
    Batch evaluation of all model outputs
    """
    # Configuration
    input_root_dir = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/segmented_replayed_trajecotries"
    raw_data_dir = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/replayed_trajectories"
    eval_folder = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/Fixed_data/evaluation/new_model_outputs"
    output_dir = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/evaluation/eval_results"
    
    print("="*80)
    print("BATCH EVALUATION OF MODEL OUTPUTS")
    print("="*80)
    print(f"Evaluation folder: {eval_folder}")
    print(f"Output directory: {output_dir}")
    
    # Initialize the evaluator
    print("\nInitializing evaluator...")
    evaluator = OrderingEvaluator(
        input_root_dir=input_root_dir,
        raw_data_dir=raw_data_dir
    )
    
    # Find all model output files
    pattern = os.path.join(eval_folder, "behavior_eqa_ordering_*.jsonl")
    model_files = glob.glob(pattern)
    
    if not model_files:
        print(f"No model output files found in {eval_folder}")
        print(f"Looking for pattern: behavior_eqa_ordering_*.jsonl")
        return
    
    print(f"\nFound {len(model_files)} model output files:")
    for file in sorted(model_files):
        model_name = extract_model_name_from_filename(file)
        print(f"  - {os.path.basename(file)} -> {model_name}")
    
    # Process each model
    # Note: The evaluator caches results in eval_results dict. We need to clear this
    # between models to avoid "already_evaluated" skips that cause identical results.
    results_summary = []
    successful_evaluations = 0
    failed_evaluations = 0
    
    for i, jsonl_path in enumerate(sorted(model_files), 1):
        model_name = extract_model_name_from_filename(jsonl_path)
        if model_name is None:
            print(f"Warning: Could not extract model name from {jsonl_path}")
            continue
            
        print(f"\n[{i}/{len(model_files)}] Processing {model_name}...")
        
        result = evaluate_single_model(evaluator, jsonl_path, model_name, output_dir)
        results_summary.append(result)
        
        if result['status'] == 'success':
            successful_evaluations += 1
        else:
            failed_evaluations += 1
    
    # Generate final summary
    print("\n" + "="*80)
    print("BATCH EVALUATION SUMMARY")
    print("="*80)
    print(f"Total models processed: {len(results_summary)}")
    print(f"Successful evaluations: {successful_evaluations}")
    print(f"Failed evaluations: {failed_evaluations}")
    
    if successful_evaluations > 0:
        print(f"\nSuccessful evaluations:")
        for result in results_summary:
            if result['status'] == 'success':
                overall_acc = result['overall_stats'].get('overall_accuracy', 0.0)
                print(f"  ✓ {result['model_name']}: {overall_acc*100:.2f}% accuracy")
    
    if failed_evaluations > 0:
        print(f"\nFailed evaluations:")
        for result in results_summary:
            if result['status'] == 'error':
                print(f"  ✗ {result['model_name']}: {result['error']}")
    
    # Save summary report
    summary_file = os.path.join(output_dir, "batch_evaluation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'total_processed': len(results_summary),
            'successful': successful_evaluations,
            'failed': failed_evaluations,
            'results': results_summary
        }, f, indent=2)
    
    print(f"\nSummary report saved to: {summary_file}")
    print(f"Individual results saved to: {output_dir}")
    print("\nBatch evaluation completed!")

if __name__ == "__main__":
    main()
