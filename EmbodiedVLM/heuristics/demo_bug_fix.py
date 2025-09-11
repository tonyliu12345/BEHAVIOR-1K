#!/usr/bin/env python3
"""
Demonstration of the GT Answer Bug Fix

This script demonstrates the bug detection and fixing logic without requiring
actual scene graph files. It shows how the invalid gt_answer sequences are
detected and fixed using the corrected algorithm.
"""

import json
import random
from typing import List, Dict, Any
from collections import defaultdict


def detect_gt_answer_issues(gt_answer: List[int]) -> tuple[bool, str]:
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


def calculate_correct_order_buggy(correct_action_sequence: List[str], shuffled_action_sequence: List[str]) -> List[int]:
    """
    The OLD BUGGY algorithm that causes duplicates.
    """
    correct_order = []
    
    # OLD BUGGY CODE - using index() which always returns first occurrence
    for original_action in correct_action_sequence:
        shuffled_position = shuffled_action_sequence.index(original_action) + 1
        correct_order.append(shuffled_position)
    
    return correct_order


def calculate_correct_order_fixed(correct_action_sequence: List[str], shuffled_action_sequence: List[str]) -> List[int]:
    """
    The FIXED algorithm that properly handles duplicates.
    """
    correct_order = []
    
    # Track which positions have been used to handle duplicate actions correctly
    used_positions = set()
    for original_action in correct_action_sequence:
        # Find the next unused position of this action in the shuffled sequence
        for i, shuffled_action in enumerate(shuffled_action_sequence):
            if shuffled_action == original_action and i not in used_positions:
                shuffled_position = i + 1
                correct_order.append(shuffled_position)
                used_positions.add(i)
                break
        else:
            # This should never happen if the sequences are properly constructed
            raise ValueError(f"Could not find unused position for action: {original_action}")
    
    # Validate that correct_order contains sequential numbers from 1 to N
    expected_sequence = list(range(1, len(correct_action_sequence) + 1))
    if sorted(correct_order) != expected_sequence:
        raise ValueError(f"Invalid correct_order sequence: {correct_order}. "
                       f"Expected sorted: {expected_sequence}, got sorted: {sorted(correct_order)}")
    
    return correct_order


def demonstrate_bug_fix():
    """Demonstrate the bug and its fix with concrete examples."""
    print("=" * 80)
    print("DEMONSTRATION: GT Answer Bug Fix for Inverse Dynamics Ordering")
    print("=" * 80)
    
    # Test cases with duplicate actions (the problematic scenario)
    test_cases = [
        {
            "name": "Simple duplicate case",
            "actions": ["action_A", "action_B", "action_A", "action_C"]
        },
        {
            "name": "Multiple duplicates",
            "actions": ["action_A", "action_B", "action_A", "action_B", "action_C"]
        },
        {
            "name": "Three of same action",
            "actions": ["action_A", "action_B", "action_A", "action_C", "action_A"]
        },
        {
            "name": "Complex pattern (like Halloween decorations)",
            "actions": ["open_cabinet", "grasp_item", "move_item", "close_cabinet", 
                       "open_cabinet", "place_item", "close_cabinet", "grasp_item", "close_cabinet"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}: {test_case['name']}")
        print(f"{'='*60}")
        
        correct_action_sequence = test_case['actions']
        print(f"Original action sequence: {correct_action_sequence}")
        
        # Set a fixed seed for reproducible results
        random.seed(42 + i)
        shuffled_action_sequence = correct_action_sequence[:]
        random.shuffle(shuffled_action_sequence)
        print(f"Shuffled action sequence: {shuffled_action_sequence}")
        
        # Show the buggy behavior
        try:
            buggy_result = calculate_correct_order_buggy(correct_action_sequence, shuffled_action_sequence)
            is_invalid, issue_type = detect_gt_answer_issues(buggy_result)
            
            print(f"\n🐛 BUGGY ALGORITHM RESULT:")
            print(f"   gt_answer: {buggy_result}")
            print(f"   Is invalid: {is_invalid}")
            print(f"   Issue type: {issue_type}")
            
            if is_invalid:
                # Count duplicates and show details
                duplicate_counts = defaultdict(int)
                for num in buggy_result:
                    duplicate_counts[num] += 1
                duplicates = {num: count for num, count in duplicate_counts.items() if count > 1}
                if duplicates:
                    print(f"   Duplicates found: {duplicates}")
                    
        except Exception as e:
            print(f"🐛 BUGGY ALGORITHM FAILED: {str(e)}")
        
        # Show the fixed behavior
        try:
            fixed_result = calculate_correct_order_fixed(correct_action_sequence, shuffled_action_sequence)
            is_invalid, issue_type = detect_gt_answer_issues(fixed_result)
            
            print(f"\n✅ FIXED ALGORITHM RESULT:")
            print(f"   gt_answer: {fixed_result}")
            print(f"   Is invalid: {is_invalid}")
            print(f"   Issue type: {issue_type}")
            print(f"   Sorted: {sorted(fixed_result)}")
            print(f"   Expected: {list(range(1, len(correct_action_sequence) + 1))}")
            
        except Exception as e:
            print(f"✅ FIXED ALGORITHM FAILED: {str(e)}")
    
    print(f"\n{'='*80}")
    print("REAL EXAMPLES FROM THE DATASET")
    print(f"{'='*80}")
    
    # Show real examples from the dataset
    real_examples = [
        {
            "id": "putting_away_Halloween_decorations_1747389873610361_inverse_dynamics_ordering_10_steps_b939fd75",
            "gt_answer": [4, 6, 1, 5, 8, 7, 2, 3, 7]
        },
        {
            "id": "putting_up_Christmas_decorations_inside_1752574067758371_inverse_dynamics_ordering_10_steps_a8585a86", 
            "gt_answer": [3, 5, 1, 8, 7, 4, 2, 6, 8]
        },
        {
            "id": "slicing_vegetables_1754479015822159_inverse_dynamics_ordering_10_steps_06845903",
            "gt_answer": [7, 4, 4, 6, 2, 3, 1, 8, 5]
        }
    ]
    
    for example in real_examples:
        print(f"\n📋 REAL EXAMPLE: {example['id'][:50]}...")
        gt_answer = example['gt_answer']
        is_invalid, issue_type = detect_gt_answer_issues(gt_answer)
        
        print(f"   Original gt_answer: {gt_answer}")
        print(f"   Length: {len(gt_answer)}")
        print(f"   Sorted: {sorted(gt_answer)}")
        print(f"   Expected: {list(range(1, len(gt_answer) + 1))}")
        print(f"   Is invalid: {is_invalid}")
        print(f"   Issue type: {issue_type}")
        
        if is_invalid:
            # Count duplicates
            duplicate_counts = defaultdict(int)
            for num in gt_answer:
                duplicate_counts[num] += 1
            duplicates = {num: count for num, count in duplicate_counts.items() if count > 1}
            missing = set(range(1, len(gt_answer) + 1)) - set(gt_answer)
            
            if duplicates:
                print(f"   Duplicates: {duplicates}")
            if missing:
                print(f"   Missing numbers: {sorted(missing)}")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("✅ The bug has been identified and fixed!")
    print("✅ The issue was in the inverse dynamics ordering gt_answer generation")
    print("✅ The problem: list.index() always returns the first occurrence")
    print("✅ The solution: Track used positions to handle duplicates correctly")
    print("✅ The fix prevents invalid gt_answer sequences with duplicates")
    print(f"{'='*80}")


if __name__ == "__main__":
    demonstrate_bug_fix()
