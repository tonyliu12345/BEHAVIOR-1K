#!/usr/bin/env python3
"""
GT Answer Validator

This script validates gt_answer sequences in a JSONL file and reports any issues.
It can be used to quickly check for the duplicate gt_answer bug without needing
to access scene graph files.
"""

import json
import sys
from collections import defaultdict
from typing import List, Dict, Any, Tuple


def detect_gt_answer_issues(gt_answer: List[int]) -> Tuple[bool, str]:
    """
    Detect if a gt_answer sequence is invalid.
    
    Args:
        gt_answer: The ground truth answer sequence
        
    Returns:
        Tuple[bool, str]: (is_invalid, issue_description)
    """
    if not gt_answer or not isinstance(gt_answer, list):
        return True, "empty_or_invalid_format"
    
    # Check if all elements are integers
    if not all(isinstance(x, int) for x in gt_answer):
        return True, "non_integer_elements"
    
    # Check for expected sequential numbers from 1 to N
    expected_sequence = list(range(1, len(gt_answer) + 1))
    sorted_gt = sorted(gt_answer)
    
    if sorted_gt != expected_sequence:
        # Analyze the specific pattern
        if len(set(gt_answer)) != len(gt_answer):
            # Has duplicates
            duplicate_counts = defaultdict(int)
            for num in gt_answer:
                duplicate_counts[num] += 1
            duplicates = [num for num, count in duplicate_counts.items() if count > 1]
            return True, f"duplicates_{len(duplicates)}_numbers"
        elif min(gt_answer) != 1 or max(gt_answer) != len(gt_answer):
            # Wrong range
            return True, f"wrong_range_{min(gt_answer)}_to_{max(gt_answer)}"
        else:
            # Missing numbers
            missing = set(expected_sequence) - set(gt_answer)
            return True, f"missing_{len(missing)}_numbers"
    
    return False, "valid"


def validate_jsonl_file(jsonl_path: str, max_lines: int = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Validate all gt_answer sequences in a JSONL file.
    
    Args:
        jsonl_path: Path to the JSONL file
        max_lines: Maximum number of lines to process (for testing)
        verbose: Whether to print detailed information for each invalid case
        
    Returns:
        Dictionary with validation statistics
    """
    stats = {
        'total_processed': 0,
        'total_invalid': 0,
        'inverse_dynamics_processed': 0,
        'inverse_dynamics_invalid': 0,
        'forward_dynamics_processed': 0,
        'forward_dynamics_invalid': 0,
        'patterns': defaultdict(int),
        'invalid_cases': []
    }
    
    print(f"Validating gt_answer sequences in: {jsonl_path}")
    if max_lines:
        print(f"Processing first {max_lines} lines only")
    
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if max_lines and line_num > max_lines:
                break
                
            try:
                data_point = json.loads(line.strip())
                
                # Extract required fields
                data_id = data_point.get('id', f'unknown_{line_num}')
                task_type = data_point.get('type', '')
                gt_answer = data_point.get('gt_answer', [])
                
                if not gt_answer:
                    continue
                    
                stats['total_processed'] += 1
                
                # Track by dynamics type
                if 'inverse' in task_type.lower():
                    stats['inverse_dynamics_processed'] += 1
                elif 'forward' in task_type.lower():
                    stats['forward_dynamics_processed'] += 1
                
                # Check for issues
                is_invalid, issue_type = detect_gt_answer_issues(gt_answer)
                
                if is_invalid:
                    stats['total_invalid'] += 1
                    stats['patterns'][issue_type] += 1
                    
                    if 'inverse' in task_type.lower():
                        stats['inverse_dynamics_invalid'] += 1
                    elif 'forward' in task_type.lower():
                        stats['forward_dynamics_invalid'] += 1
                    
                    invalid_case = {
                        'line_num': line_num,
                        'id': data_id,
                        'type': task_type,
                        'gt_answer': gt_answer,
                        'issue_type': issue_type
                    }
                    
                    # Add detailed analysis for duplicates
                    if 'duplicates' in issue_type:
                        duplicate_counts = defaultdict(int)
                        for num in gt_answer:
                            duplicate_counts[num] += 1
                        duplicates = {num: count for num, count in duplicate_counts.items() if count > 1}
                        missing = set(range(1, len(gt_answer) + 1)) - set(gt_answer)
                        invalid_case['duplicates'] = duplicates
                        invalid_case['missing'] = sorted(missing) if missing else []
                    
                    stats['invalid_cases'].append(invalid_case)
                    
                    if verbose:
                        print(f"Line {line_num}: INVALID - {data_id[:60]}...")
                        print(f"  Type: {task_type}")
                        print(f"  gt_answer: {gt_answer}")
                        print(f"  Issue: {issue_type}")
                        if 'duplicates' in invalid_case:
                            print(f"  Duplicates: {invalid_case['duplicates']}")
                            print(f"  Missing: {invalid_case['missing']}")
                        print()
                        
            except json.JSONDecodeError:
                print(f"Line {line_num}: JSON decode error, skipping")
                continue
            except Exception as e:
                print(f"Line {line_num}: Error processing line: {str(e)}")
                continue
    
    return stats


def print_validation_summary(stats: Dict[str, Any]):
    """Print a summary of validation results."""
    print("\n" + "="*80)
    print("GT ANSWER VALIDATION SUMMARY")
    print("="*80)
    print(f"Total processed:           {stats['total_processed']}")
    print(f"Total invalid:             {stats['total_invalid']}")
    print(f"Invalid percentage:        {stats['total_invalid']/stats['total_processed']*100:.2f}%" if stats['total_processed'] > 0 else "N/A")
    
    print(f"\nBy dynamics type:")
    print(f"  Forward dynamics:")
    print(f"    Processed:             {stats['forward_dynamics_processed']}")
    print(f"    Invalid:               {stats['forward_dynamics_invalid']}")
    print(f"    Invalid percentage:    {stats['forward_dynamics_invalid']/stats['forward_dynamics_processed']*100:.2f}%" if stats['forward_dynamics_processed'] > 0 else "N/A")
    
    print(f"  Inverse dynamics:")
    print(f"    Processed:             {stats['inverse_dynamics_processed']}")
    print(f"    Invalid:               {stats['inverse_dynamics_invalid']}")
    print(f"    Invalid percentage:    {stats['inverse_dynamics_invalid']/stats['inverse_dynamics_processed']*100:.2f}%" if stats['inverse_dynamics_processed'] > 0 else "N/A")
    
    if stats['patterns']:
        print(f"\nIssue patterns:")
        for pattern, count in sorted(stats['patterns'].items()):
            percentage = count/stats['total_invalid']*100 if stats['total_invalid'] > 0 else 0
            print(f"  {pattern}: {count} ({percentage:.1f}%)")
    
    if stats['invalid_cases']:
        print(f"\nFirst 10 invalid cases:")
        for i, case in enumerate(stats['invalid_cases'][:10], 1):
            print(f"  {i}. Line {case['line_num']}: {case['id'][:50]}...")
            print(f"     Type: {case['type']}")
            print(f"     gt_answer: {case['gt_answer']}")
            print(f"     Issue: {case['issue_type']}")
            if 'duplicates' in case:
                print(f"     Duplicates: {case['duplicates']}, Missing: {case['missing']}")
    
    print("="*80)


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python validate_gt_answers.py <jsonl_file> [max_lines] [--verbose]")
        print("Example: python validate_gt_answers.py data.jsonl 1000 --verbose")
        return 1
    
    jsonl_path = sys.argv[1]
    max_lines = None
    verbose = False
    
    if len(sys.argv) > 2:
        try:
            max_lines = int(sys.argv[2])
        except ValueError:
            if sys.argv[2] == '--verbose':
                verbose = True
    
    if len(sys.argv) > 3 and sys.argv[3] == '--verbose':
        verbose = True
    
    try:
        stats = validate_jsonl_file(jsonl_path, max_lines, verbose)
        print_validation_summary(stats)
        
        if stats['total_invalid'] > 0:
            print(f"\n⚠️  Found {stats['total_invalid']} invalid gt_answer sequences!")
            print("   These need to be fixed using the fix_gt_answer.py script.")
        else:
            print("\n✅ All gt_answer sequences are valid!")
            
    except FileNotFoundError:
        print(f"Error: File not found: {jsonl_path}")
        return 1
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
