"""
Usage example for the OrderingEvaluator class.
This script demonstrates how to use the OrderingEvaluator for batch evaluation of ordering tasks.
"""

import os
import sys

# Add parent directories to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'OmniGibson'))


from pathlib import Path
import json

try:
    from EmbodiedVLM.heuristics.evaluators import OrderingEvaluator
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def main():
    """
    Example usage of OrderingEvaluator
    """
    # Initialize the evaluator
    input_root_dir = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/segmented_replayed_trajecotries"  # Replace with actual path
    raw_data_dir = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/replayed_trajectories"      # Replace with actual path
    
    evaluator = OrderingEvaluator(
        input_root_dir=input_root_dir,
        raw_data_dir=raw_data_dir
    )
    model_name = "gpt-5"
    # Path to your JSONL evaluation file
    jsonl_path = f"/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/Fixed_data/evaluation/model_outputs/behavior_eqa_ordering_{model_name}.jsonl"  # Replace with actual path
    wrong_case_path = f"/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/evaluation/error_analysis/behavior_eqa_ordering_{model_name}_wrong_case.jsonl"
    
    print("Starting batch evaluation...")
    
    # Run the evaluation
    evaluator.evaluate(jsonl_path, analyze_wrong_case=True)
    
    # Generate reports
    print("\n" + "="*50)
    print("EVALUATION REPORTS")
    print("="*50)
    
    # 1. Overall performance report
    overall_report = evaluator.report_overall_score()
    print("\n1. OVERALL PERFORMANCE:")
    
    if 'overall' in overall_report:
        overall_stats = overall_report['overall']
        print(f"   Total Evaluated: {overall_stats.get('count', 0)}")
        print(f"   Overall Accuracy: {overall_stats.get('overall_accuracy', 0.0)*100:.2f}")
        print(f"   Exact Match Accuracy: {overall_stats.get('exact_match_accuracy', 0.0)*100:.2f}")
        print(f"   Semantic Match Accuracy: {overall_stats.get('semantic_match_accuracy', 0.0)*100:.2f}")
        print(f"   Average Pair Correct Ratio: {overall_stats.get('avg_pair_correct_ratio', 0.0)*100:.2f}")
    
    if 'forward_dynamics' in overall_report:
        fwd_stats = overall_report['forward_dynamics']
        print(f"   Forward Dynamics Accuracy: {fwd_stats.get('overall_accuracy', 0.0):.3f} "
              f"({fwd_stats.get('count', 0)} samples)")
    
    if 'inverse_dynamics' in overall_report:
        inv_stats = overall_report['inverse_dynamics']
        print(f"   Inverse Dynamics Accuracy: {inv_stats.get('overall_accuracy', 0.0):.3f} "
              f"({inv_stats.get('count', 0)} samples)")
    
    # 2. Performance by task type
    output_file = f"./eval/evaluation_results_{model_name}.json"
    task_type_report = evaluator.report_by_task_type()
    print("\n2. PERFORMANCE BY TASK TYPE:")
    for question_type, step_results in task_type_report.items():
        print(f"   {question_type}:")
        for step_num, stats in step_results.items():
            accuracy = stats.get('overall_accuracy', 0.0)
            count = stats.get('count', 0)
            print(f"     {step_num} steps. accuracy: {accuracy*100:.2f} pair correct ratio: {stats.get('avg_pair_correct_ratio', 0.0)*100:.2f} ({count} samples)")
    
    # 5. Save performance by task type to JSON file
    # with open(output_file, 'w') as f:
    #     json.dump(task_type_report, f, indent=2)
    # print(f"\n5. Performance by task type saved to: {output_file}")

    # Save wrong case results to JSONL file
    for data_id, wrong_case_entry in evaluator.wrong_case_signatures.items():
        match = wrong_case_entry['eval_metrics']['match']
        # if match:
        #     continue
        with open(wrong_case_path, 'a') as f:
            json.dump(wrong_case_entry, f, ensure_ascii=False)
            f.write('\n')
    print(f"\n6. Wrong case results saved to: {wrong_case_path}")
    
    # Save skipped items report
    # skipped_report = evaluator.report_skipped_items()
    # if skipped_report['total_skipped'] > 0:
    #     skipped_output_file = "skipped_items_report.json"
    #     with open(skipped_output_file, 'w') as f:
    #         json.dump(skipped_report, f, indent=2)
    #     print(f"   Skipped items report saved to: {skipped_output_file}")
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":  
    main()
