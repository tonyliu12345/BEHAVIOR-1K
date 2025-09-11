"""
Ablation evaluation script for the OrderingEvaluator class.
This script demonstrates how to use the OrderingEvaluator for batch evaluation of ablation ordering tasks,
with detailed analysis of performance across different step settings and comprehensive statistical analysis.
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
import numpy as np
from scipy import stats
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

try:
    from EmbodiedVLM.heuristics.evaluators import OrderingEvaluator
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate comprehensive statistics including SD, SEM, and 95% CI.
    
    Args:
        values: List of numerical values (e.g., accuracy scores)
        
    Returns:
        Dict containing mean, std, sem, and 95% confidence interval bounds
    """
    if not values:
        return {'mean': 0.0, 'std': 0.0, 'sem': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0, 'n': 0}
    
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0.0  # Sample standard deviation
    sem = std / np.sqrt(n) if n > 0 else 0.0  # Standard error of mean
    
    # 95% confidence interval using t-distribution
    if n > 1:
        t_critical = stats.t.ppf(0.975, df=n-1)  # 95% CI, two-tailed
        ci_lower = mean - t_critical * sem
        ci_upper = mean + t_critical * sem
    else:
        ci_lower = ci_upper = mean
    
    return {
        'mean': mean,
        'std': std,
        'sem': sem,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n': n
    }

def paired_t_test(group1: List[float], group2: List[float]) -> Dict[str, float]:
    """
    Perform paired t-test between two groups of equal length.
    
    Args:
        group1: First group of values (e.g., baseline performance)
        group2: Second group of values (e.g., ablation performance)
        
    Returns:
        Dict containing t-statistic, p-value, and effect size (Cohen's d)
    """
    if len(group1) != len(group2) or len(group1) < 2:
        return {'t_statistic': 0.0, 'p_value': 1.0, 'cohens_d': 0.0, 'significant': False}
    
    # Convert to numpy arrays
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    # Perform paired t-test
    t_statistic, p_value = stats.ttest_rel(group1, group2)
    
    # Calculate Cohen's d for effect size
    differences = group1 - group2
    cohens_d = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences, ddof=1) > 0 else 0.0
    
    # Check significance at α = 0.05
    significant = p_value < 0.05
    
    return {
        't_statistic': t_statistic,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': significant
    }

def get_ablation_category_mapping():
    """
    Define mapping from ablation settings to their categories and display names.
    Returns a dictionary with category info and ordered settings.
    
    Note: The evaluation data contains a single 'raw_baseline' entry, but based on 
    the reference format, it should be treated as the baseline for different ablations.
    We'll handle this special case in the print function.
    """
    return {
        "Ablation 1: Image Realism": {
            "settings": [
                "realistic_frame",
                "path_tracing", 
                "raw_baseline",  # Will be displayed as raw_baseline(ray_tracing)
                "ray_tracing_raw"
            ],
            "baseline_display_name": "raw_baseline(ray_tracing)"
        },
        "Ablation 2.1: FOV": {
            "settings": [
                "aperture_30",
                "raw_baseline",  # Will be displayed as raw_baseline(40) 
                "aperture_60",
                "aperture_80",
                "fisheyePolynomial"
            ],
            "baseline_display_name": "raw_baseline(40)"
        },
        "Ablation 2.2: Camera Height": {
            "settings": [
                "high",
                "raw_baseline",  # Will be displayed as raw_baseline(mid)
                "low"
            ],
            "baseline_display_name": "raw_baseline(mid)"
        },
        "Ablation 3.1: Robot Appearance": {
            "settings": [
                "white_color",
                "raw_baseline",  # Will be displayed as raw_baseline(fancy)
                "random_color",
                "skin_color"
            ],
            "baseline_display_name": "raw_baseline(fancy)"
        },
        "Ablation 4: Visual Prompting": {
            "settings": [
                "raw_baseline",  # Will be displayed as raw_baseline(no_bbox_no_name)
                "consistent_no_name",
                "consistent_random_name",
                "consistent_with_name",
                "random_no_name",
                "random_random_name",
                "random_with_name",
                "same_no_name",
                "same_with_name",
                "mask_overlay"
            ],
            "baseline_display_name": "raw_baseline(no_bbox_no_name)"
        }
    }

def process_ablation_data_with_stats(evaluator, fine_grained: bool = False) -> Dict[str, Any]:
    """
    Process ablation evaluation results with comprehensive statistical analysis.
    
    Args:
        evaluator: OrderingEvaluator instance with eval_results
        fine_grained: If True, report step-by-step; if False, aggregate across steps
        
    Returns:
        Processed data with statistics and significance testing
    """
    if not evaluator.eval_results:
        print("No evaluation results found. Please run evaluate() first.")
        return {}
        
    # Group raw data by ablation setting, dynamics type, and step
    raw_data_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    for result in evaluator.eval_results.values():
        data_id = result.get('id', '')
        task_type = result.get('type', '')
        eval_metrics = result.get('eval_metrics', {})
        
        if not data_id or not task_type or not eval_metrics:
            continue
            
        # Extract ablation setting from ID (last part after final underscore)
        ablation_setting = 'unknown'
        if '_' in data_id:
            ablation_setting = '_'.join(data_id.split('_')[9:])
        
        # Extract dynamics type
        dynamics_type = 'unknown'
        if 'forward' in task_type.lower():
            dynamics_type = 'forward_dynamics'
        elif 'inverse' in task_type.lower():
            dynamics_type = 'inverse_dynamics'
            
        # Extract step number
        parts = task_type.split('_')
        step_num = 'unknown'
        for part in parts:
            if 'step' in part.lower():
                step_digits = ''.join(filter(str.isdigit, part))
                if step_digits:
                    step_num = step_digits
                    break
        if step_num == 'unknown':
            for part in parts:
                if part.isdigit():
                    step_num = part
                    break
                    
        # Store individual result data
        raw_data_groups[ablation_setting][dynamics_type][step_num]['overall_accuracy'].append(eval_metrics.get('match', False))
        raw_data_groups[ablation_setting][dynamics_type][step_num]['exact_match'].append(eval_metrics.get('exact_match', False))
        raw_data_groups[ablation_setting][dynamics_type][step_num]['semantic_match'].append(eval_metrics.get('semantic_match', False))
        raw_data_groups[ablation_setting][dynamics_type][step_num]['pair_correct_ratio'].append(eval_metrics.get('pair_correct_ratio', 0.0))
    
    # Process data based on fine_grained setting
    processed_data = {}
    
    for ablation_setting, dynamics_data in raw_data_groups.items():
        processed_data[ablation_setting] = {}
        
        for dynamics_type, step_data in dynamics_data.items():
            if fine_grained:
                # Keep step-by-step data
                processed_data[ablation_setting][dynamics_type] = {}
                for step_num, metrics in step_data.items():
                    processed_data[ablation_setting][dynamics_type][step_num] = {
                        'overall_accuracy': calculate_statistics([float(x) for x in metrics['overall_accuracy']]),
                        'exact_match': calculate_statistics([float(x) for x in metrics['exact_match']]),
                        'semantic_match': calculate_statistics([float(x) for x in metrics['semantic_match']]),
                        'pair_correct_ratio': calculate_statistics(metrics['pair_correct_ratio']),
                        'raw_data': metrics  # Keep raw data for t-tests
                    }
            else:
                # Aggregate across all steps (3, 6, 9)
                aggregated_metrics = defaultdict(list)
                for step_num, metrics in step_data.items():
                    for metric_name, metric_values in metrics.items():
                        aggregated_metrics[metric_name].extend(metric_values)
                
                processed_data[ablation_setting][dynamics_type] = {
                    'aggregated': {
                        'overall_accuracy': calculate_statistics([float(x) for x in aggregated_metrics['overall_accuracy']]),
                        'exact_match': calculate_statistics([float(x) for x in aggregated_metrics['exact_match']]),
                        'semantic_match': calculate_statistics([float(x) for x in aggregated_metrics['semantic_match']]),
                        'pair_correct_ratio': calculate_statistics(aggregated_metrics['pair_correct_ratio']),
                        'raw_data': dict(aggregated_metrics)  # Keep raw data for t-tests
                    }
                }
    
    return processed_data

def perform_significance_testing(processed_data: Dict[str, Any], baseline_name: str = 'raw_baseline') -> Dict[str, Any]:
    """
    Perform statistical significance testing between baseline and other ablation settings.
    
    Args:
        processed_data: Output from process_ablation_data_with_stats
        baseline_name: Name of the baseline setting to compare against
        
    Returns:
        Dictionary with significance test results
    """
    significance_results = {}
    
    if baseline_name not in processed_data:
        print(f"Warning: Baseline '{baseline_name}' not found in data")
        return significance_results
    
    baseline_data = processed_data[baseline_name]
    
    for ablation_setting, ablation_data in processed_data.items():
        if ablation_setting == baseline_name:
            continue
            
        significance_results[ablation_setting] = {}
        
        for dynamics_type in ablation_data:
            if dynamics_type not in baseline_data:
                continue
                
            significance_results[ablation_setting][dynamics_type] = {}
            
            # Compare each step or aggregated data
            for step_or_agg in ablation_data[dynamics_type]:
                if step_or_agg not in baseline_data[dynamics_type]:
                    continue
                    
                baseline_metrics = baseline_data[dynamics_type][step_or_agg]['raw_data']
                ablation_metrics = ablation_data[dynamics_type][step_or_agg]['raw_data']
                
                significance_results[ablation_setting][dynamics_type][step_or_agg] = {}
                
                # Perform t-tests for each metric
                for metric_name in ['overall_accuracy', 'exact_match', 'semantic_match', 'pair_correct_ratio']:
                    if metric_name in baseline_metrics and metric_name in ablation_metrics:
                        baseline_values = [float(x) if metric_name != 'pair_correct_ratio' else x 
                                         for x in baseline_metrics[metric_name]]
                        ablation_values = [float(x) if metric_name != 'pair_correct_ratio' else x 
                                         for x in ablation_metrics[metric_name]]
                        
                        if len(baseline_values) == len(ablation_values) and len(baseline_values) > 1:
                            t_test_result = paired_t_test(baseline_values, ablation_values)
                            significance_results[ablation_setting][dynamics_type][step_or_agg][metric_name] = t_test_result
    
    return significance_results

def print_statistical_ablation_report(processed_data: Dict[str, Any], 
                                      significance_results: Dict[str, Any],
                                      fine_grained: bool = False) -> None:
    """
    Print ablation report with comprehensive statistical analysis including error bars and significance tests.
    
    Args:
        processed_data: Output from process_ablation_data_with_stats
        significance_results: Output from perform_significance_testing
        fine_grained: Whether to show step-by-step or aggregated results
    """
    category_mapping = get_ablation_category_mapping()
    
    print(f"\n{'='*80}")
    print(f"📊 STATISTICAL ABLATION REPORT {'(Fine-grained)' if fine_grained else '(Aggregated)'}")
    print(f"{'='*80}")
    
    # Track which settings we've seen in the data
    settings_in_data = set(processed_data.keys())
    raw_baseline_data = processed_data.get("raw_baseline", None)
    
    for category, info in category_mapping.items():
        # Check if any settings from this category exist in the data
        category_settings_in_data = []
        
        for setting in info["settings"]:
            if setting == "raw_baseline":
                if raw_baseline_data is not None:
                    category_settings_in_data.append(setting)
            elif setting in settings_in_data:
                category_settings_in_data.append(setting)
        
        if not category_settings_in_data:
            continue
            
        print(f"\n## {category}")
        
        for setting in category_settings_in_data:
            # Get the data
            if setting == "raw_baseline" and raw_baseline_data is not None:
                setting_data = raw_baseline_data
            elif setting in processed_data:
                setting_data = processed_data[setting]
            else:
                continue
            
            # Handle special display name for raw_baseline
            display_name = setting
            if setting == "raw_baseline" and "baseline_display_name" in info:
                display_name = info["baseline_display_name"]
            
            print(f"\n🧪 ABLATION SETTING: {display_name}")
            print("=" * 60)
            
            # Process Forward Dynamics first, then Inverse Dynamics
            dynamics_order = ["forward_dynamics", "inverse_dynamics"]
            
            for dynamics_type in dynamics_order:
                if dynamics_type not in setting_data:
                    continue
                
                step_data = setting_data[dynamics_type]
                if not step_data:
                    continue
                
                print(f"\n🔄 {dynamics_type.upper().replace('_', ' ')}:")
                print("-" * 80)
                
                if fine_grained:
                    # Fine-grained: show each step
                    print(f"{'Step':<6} {'Count':<6} {'Accuracy':<12} {'95% CI':<20} {'SEM':<8} {'Significance':<12}")
                    print("-" * 80)
                    
                    # Sort steps numerically
                    sorted_steps = sorted(step_data.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
                    
                    for step in sorted_steps:
                        stats = step_data[step]['overall_accuracy']
                        
                        # Format confidence interval
                        ci_str = f"[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]"
                        
                        # Get significance information if available
                        sig_str = "-"
                        if (setting in significance_results and 
                            dynamics_type in significance_results[setting] and 
                            step in significance_results[setting][dynamics_type]):
                            
                            sig_info = significance_results[setting][dynamics_type][step]['overall_accuracy']
                            p_val = sig_info['p_value']
                            if p_val < 0.001:
                                sig_str = "***"
                            elif p_val < 0.01:
                                sig_str = "**"
                            elif p_val < 0.05:
                                sig_str = "*"
                            else:
                                sig_str = "n.s."
                            sig_str += f" (p={p_val:.3f})"
                        
                        print(f"{step:<6} {stats['n']:<6} {stats['mean']:<12.4f} "
                              f"{ci_str:<20} {stats['sem']:<8.4f} {sig_str:<12}")
                else:
                    # Aggregated: show combined performance
                    print(f"{'Metric':<15} {'Count':<6} {'Mean':<10} {'95% CI':<20} {'SEM':<8} {'Significance':<15}")
                    print("-" * 80)
                    
                    metrics_to_show = ['overall_accuracy', 'exact_match', 'semantic_match', 'pair_correct_ratio']
                    metric_display_names = {
                        'overall_accuracy': 'Overall Acc',
                        'exact_match': 'Exact Match',
                        'semantic_match': 'Semantic',
                        'pair_correct_ratio': 'Pair Correct'
                    }
                    
                    for metric_name in metrics_to_show:
                        if 'aggregated' in step_data and metric_name in step_data['aggregated']:
                            stats = step_data['aggregated'][metric_name]
                            
                            # Format confidence interval
                            ci_str = f"[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]"
                            
                            # Get significance information if available
                            sig_str = "-"
                            if (setting in significance_results and 
                                dynamics_type in significance_results[setting] and 
                                'aggregated' in significance_results[setting][dynamics_type] and
                                metric_name in significance_results[setting][dynamics_type]['aggregated']):
                                
                                sig_info = significance_results[setting][dynamics_type]['aggregated'][metric_name]
                                p_val = sig_info['p_value']
                                if p_val < 0.001:
                                    sig_str = "***"
                                elif p_val < 0.01:
                                    sig_str = "**"
                                elif p_val < 0.05:
                                    sig_str = "*"
                                else:
                                    sig_str = "n.s."
                                sig_str += f" (p={p_val:.3f})"
                            
                            display_name = metric_display_names.get(metric_name, metric_name)
                            print(f"{display_name:<15} {stats['n']:<6} {stats['mean']:<10.4f} "
                                  f"{ci_str:<20} {stats['sem']:<8.4f} {sig_str:<15}")
            
            print()  # Empty line after each setting
    
    print(f"\n{'='*80}")
    print("📋 LEGEND:")
    print("*** p < 0.001, ** p < 0.01, * p < 0.05, n.s. = not significant")
    print("95% CI = 95% Confidence Interval")
    print("SEM = Standard Error of the Mean")
    print(f"{'='*80}")

def print_organized_ablation_report(ablation_report):
    """
    Print the ablation report organized by categories, matching the reference format.
    """
    category_mapping = get_ablation_category_mapping()
    
    # Track which settings we've seen in the data
    settings_in_data = set(ablation_report.keys())
    
    # Handle the special case of raw_baseline - it should appear in each category
    raw_baseline_data = ablation_report.get("raw_baseline", None)
    
    for category, info in category_mapping.items():
        # Check if any settings from this category exist in the data
        category_settings_in_data = []
        
        for setting in info["settings"]:
            if setting == "raw_baseline":
                # Include raw_baseline if it exists in the data
                if raw_baseline_data is not None:
                    category_settings_in_data.append(setting)
            elif setting in settings_in_data:
                category_settings_in_data.append(setting)
        
        if not category_settings_in_data:
            continue
            
        print(f"\n## {category}")
        
        for setting in category_settings_in_data:
            # Get the data - use raw_baseline_data for raw_baseline entries
            if setting == "raw_baseline" and raw_baseline_data is not None:
                dynamics_data = raw_baseline_data
            elif setting in ablation_report:
                dynamics_data = ablation_report[setting]
            else:
                continue
            
            # Handle special display name for raw_baseline
            display_name = setting
            if setting == "raw_baseline" and "baseline_display_name" in info:
                display_name = info["baseline_display_name"]
            
            print(f"🧪 ABLATION SETTING: {display_name}")
            print("=" * 60)
            
            # Process Forward Dynamics first, then Inverse Dynamics
            dynamics_order = ["forward_dynamics", "inverse_dynamics"]
            
            for dynamics_type in dynamics_order:
                if dynamics_type not in dynamics_data:
                    continue
                    
                step_data = dynamics_data[dynamics_type]
                if not step_data:
                    continue
                
                print(f"\n🔄 {dynamics_type.upper().replace('_', ' ')}:")
                print("-" * 60)
                print(f"{'Step':<8} {'Count':<8} {'Accuracy':<10} {'Exact':<8} {'Semantic':<10} {'Pair Correct Ratio':<12}")
                print("-" * 70)
                
                # Sort steps numerically
                sorted_steps = sorted(step_data.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
                
                for step in sorted_steps:
                    stats = step_data[step]
                    print(f"{step:<8} {stats['count']:<8} {stats['overall_accuracy']:<10.4f} "
                          f"{stats['exact_match_accuracy']:<8.4f} {stats['semantic_match_accuracy']:<10.4f} "
                          f"{stats['avg_pair_correct_ratio']:<12.4f}")
            
            print()  # Empty line after each setting
    
    # Handle any unmatched settings (excluding raw_baseline which is handled above)
    matched_settings = set()
    for info in category_mapping.values():
        for setting in info["settings"]:
            if setting == "raw_baseline":
                matched_settings.add("raw_baseline")
            else:
                matched_settings.add(setting)
    
    unmatched_settings = settings_in_data - matched_settings
    
    if unmatched_settings:
        print(f"\n## Other Ablation Settings")
        for setting in sorted(unmatched_settings):
            dynamics_data = ablation_report[setting]
            
            print(f"🧪 ABLATION SETTING: {setting}")
            print("=" * 60)
            
            dynamics_order = ["forward_dynamics", "inverse_dynamics"]
            
            for dynamics_type in dynamics_order:
                if dynamics_type not in dynamics_data:
                    continue
                    
                step_data = dynamics_data[dynamics_type]
                if not step_data:
                    continue
                
                print(f"\n🔄 {dynamics_type.upper().replace('_', ' ')}:")
                print("-" * 60)
                print(f"{'Step':<8} {'Count':<8} {'Accuracy':<10} {'Exact':<8} {'Semantic':<10} {'Pair Correct Ratio':<12}")
                print("-" * 70)
                
                sorted_steps = sorted(step_data.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
                
                for step in sorted_steps:
                    stats = step_data[step]
                    print(f"{step:<8} {stats['count']:<8} {stats['overall_accuracy']:<10.4f} "
                          f"{stats['exact_match_accuracy']:<8.4f} {stats['semantic_match_accuracy']:<10.4f} "
                          f"{stats['avg_pair_correct_ratio']:<12.4f}")
            
            print()

def main():
    """
    Ablation evaluation using OrderingEvaluator with detailed step performance analysis and statistical significance testing
    """
    # ===== CONFIGURATION =====
    # Set this to True for step-by-step analysis, False for aggregated analysis
    REPORT_FINE_GRAINED = False  # Default: show aggregated results across 3, 6, 9 steps
    
    # Initialize the evaluator
    input_root_dir = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/segmented_replayed_trajecotries"
    raw_data_dir = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/replayed_trajectories"
    
    evaluator = OrderingEvaluator(
        input_root_dir=input_root_dir,
        raw_data_dir=raw_data_dir
    )
    
    # Model and ablation settings
    # model_name = "gpt-5-mini-2025-08-07"
    model_name = "internvl3.5-241b-a28b"
    ablation_type = "ablation"
    
    # Path to your ablation JSONL evaluation file
    jsonl_path = f"/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/Fixed_data/evaluation/ablation_model_outputs/behavior_eqa_ordering_ablation_{model_name}.jsonl"
    
    print("="*80)
    print("🧪 ABLATION EVALUATION - STATISTICAL ANALYSIS")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Evaluation file: {jsonl_path}")
    print(f"Input root: {input_root_dir}")
    print(f"Raw data: {raw_data_dir}")
    print(f"Report mode: {'Fine-grained (step-by-step)' if REPORT_FINE_GRAINED else 'Aggregated (3,6,9 steps combined)'}")
    
    # Check if file exists
    if not os.path.exists(jsonl_path):
        print(f"❌ ERROR: Evaluation file not found: {jsonl_path}")
        return
    
    print("\n🚀 Starting batch evaluation...")
    
    # Run the evaluation
    evaluator.evaluate(jsonl_path)
    
    # Generate comprehensive ablation report
    print("\n" + "="*80)
    print("📊 STATISTICAL ABLATION ANALYSIS")
    print("="*80)
    
    # ===== NEW STATISTICAL ANALYSIS =====
    print("\n🧮 PROCESSING EVALUATION DATA WITH STATISTICAL ANALYSIS...")
    
    # Process data with statistical analysis
    processed_data = process_ablation_data_with_stats(evaluator, fine_grained=REPORT_FINE_GRAINED)
    
    # Perform significance testing against baseline
    significance_results = perform_significance_testing(processed_data, baseline_name='raw_baseline')
    
    # Print statistical ablation report
    print_statistical_ablation_report(processed_data, significance_results, fine_grained=REPORT_FINE_GRAINED)
    
    # ===== LEGACY REPORT (for comparison) =====
    if REPORT_FINE_GRAINED:  # Only show legacy report when fine-grained for detailed comparison
        print("\n" + "="*80)
        print("📋 LEGACY DETAILED ABLATION TASK REPORT (for comparison)")
        print("="*80)
        ablation_report = evaluator.report_by_ablation_task()
        print_organized_ablation_report(ablation_report)
    
    # 2. Overall performance summary
    print("\n" + "="*50)
    print("📈 OVERALL PERFORMANCE SUMMARY")
    print("="*50)
    
    overall_report = evaluator.report_overall_score()
    
    if 'overall' in overall_report:
        overall_stats = overall_report['overall']
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total Evaluated: {overall_stats.get('count', 0)}")
        print(f"   Overall Accuracy: {overall_stats.get('overall_accuracy', 0.0)*100:.2f}%")
        print(f"   Exact Match Accuracy: {overall_stats.get('exact_match_accuracy', 0.0)*100:.2f}%")
        print(f"   Semantic Match Accuracy: {overall_stats.get('semantic_match_accuracy', 0.0)*100:.2f}%")
    
    if 'forward_dynamics' in overall_report:
        fwd_stats = overall_report['forward_dynamics']
        print(f"\n🔄 FORWARD DYNAMICS:")
        print(f"   Accuracy: {fwd_stats.get('overall_accuracy', 0.0)*100:.2f}% ({fwd_stats.get('count', 0)} samples)")
    
    if 'inverse_dynamics' in overall_report:
        inv_stats = overall_report['inverse_dynamics']
        print(f"\n🔄 INVERSE DYNAMICS:")
        print(f"   Accuracy: {inv_stats.get('overall_accuracy', 0.0)*100:.2f}% ({inv_stats.get('count', 0)} samples)")
    
    print(f"\n" + "="*80)
    print("🎉 STATISTICAL ABLATION EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"📊 Results organized by ablation category with comprehensive statistical analysis")
    print(f"🔬 Includes 95% confidence intervals, standard errors, and paired t-test significance testing")
    print(f"⚙️  REPORT_FINE_GRAINED = {REPORT_FINE_GRAINED} ({'Step-by-step' if REPORT_FINE_GRAINED else 'Aggregated across 3,6,9 steps'})")
    print(f"🎯 Statistical significance marked: *** p<0.001, ** p<0.01, * p<0.05")


if __name__ == "__main__":
    main()
