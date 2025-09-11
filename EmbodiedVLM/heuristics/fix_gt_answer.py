#!/usr/bin/env python3
"""
GT Answer Fixer - Simple script to fix invalid gt_answer sequences in JSONL files.

Fixes duplicate numbers in gt_answer sequences by properly handling duplicate actions.

Usage:
    # Single file mode (default)
    python fix_gt_answer.py --input_file input.jsonl --output_file output.jsonl
    
    # Batch processing mode
    python fix_gt_answer.py --batch --input_dir /path/to/input/dir --output_root /path/to/output/root

Examples:
    # Process single file
    python fix_gt_answer.py --input_file data/broken.jsonl --output_file data/fixed.jsonl
    
    # Process all JSONL files in a directory (maintains directory structure)
    python fix_gt_answer.py --batch --input_dir evaluation/model_outputs --output_root Fixed_data
"""

import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm


class GTAnswerFixer:
    """Simple class to fix invalid gt_answer sequences in ordering tasks."""
    
    def __init__(self):
        """Initialize the GTAnswerFixer."""
        self.stats = {
            'total_processed': 0,
            'invalid_gt_answers': 0,
            'fixed_gt_answers': 0,
            'errors': 0,
            'skipped': 0
        }
        
    
    def _is_gt_answer_invalid(self, gt_answer: List[int]) -> bool:
        """Check if a gt_answer sequence is invalid (has duplicates or wrong range)."""
        if not gt_answer or not isinstance(gt_answer, list):
            return True
        
        if not all(isinstance(x, int) for x in gt_answer):
            return True
        
        # Check if it's a valid sequence from 1 to N
        expected_sequence = list(range(1, len(gt_answer) + 1))
        return sorted(gt_answer) != expected_sequence

    def _fix_gt_answer(self, question: str, gt_answer: List[int]) -> List[int]:
        """Fix gt_answer by properly handling duplicate actions."""
        # Extract actions from question
        actions_dict = self._extract_actions_from_question(question)
        
        # Group same actions together
        same_actions_idx_dict = {}
        for i, action in actions_dict.items():
            if action in same_actions_idx_dict:
                same_actions_idx_dict[action].append(i)
            else:
                same_actions_idx_dict[action] = [i]
        
        # Sort indices for each action group
        for action, idx_list in same_actions_idx_dict.items():
            idx_list.sort()

        # Build mapping for duplicate actions
        min_num_same_action_dict = {}
        for _, idx_list in same_actions_idx_dict.items():
            if len(idx_list) > 1:
                min_idx = min(idx_list)
                min_num_same_action_dict[min_idx] = idx_list
        
        # Fix the gt_answer sequence
        fixed_gt_answer = []
        for i in gt_answer:
            if i in min_num_same_action_dict and i in min_num_same_action_dict[i]:
                fixed_gt_answer.append(i)
                min_num_same_action_dict[i].remove(i)
            elif i in min_num_same_action_dict and i not in min_num_same_action_dict[i]:
                popped_idx = min_num_same_action_dict[i].pop(0)
                fixed_gt_answer.append(popped_idx)
            else:
                fixed_gt_answer.append(i)
        
        return fixed_gt_answer

    def _extract_actions_from_question(self, question: str) -> Dict[int, str]:
        """Extract action strings from the question text."""
        actions_dict = {}
        lines = question.split('\n')
        actions_section_started = False
        
        for line in lines:
            line = line.strip()
            
            # Check if we've reached the actions section
            if "## Actions in Order" in line or "## Shuffled Actions" in line:
                actions_section_started = True
                continue
            
            # Skip lines before the actions section
            if not actions_section_started:
                continue
            
            # Stop processing if we reach another section
            if line.startswith("##") or line.startswith("Now please provide"):
                break
            
            # Look for action lines in format: [Action N] description
            action_match = re.match(r'\[Action\s+(\d+)\]\s*(.+)', line)
            if action_match:
                action_num = int(action_match.group(1))
                action_text = action_match.group(2).strip()
                actions_dict[action_num] = action_text
        
        return actions_dict
    
    def fix_jsonl_file(self, input_jsonl_path: str, output_jsonl_path: str) -> None:
        """Fix all invalid gt_answer sequences in a JSONL file."""
        print(f"Processing: {input_jsonl_path}")
        print(f"Output: {output_jsonl_path}")
        
        # Ensure output directory exists
        Path(output_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
        
        fixed_items = []
        
        with open(input_jsonl_path, 'r') as infile:
            lines = infile.readlines()
        
        for line_num, line in enumerate(tqdm(lines, desc="Processing"), 1):
            try:
                self.stats['total_processed'] += 1
                data_point = json.loads(line.strip())
                
                # Only process inverse dynamics ordering tasks
                task_type = data_point.get('type', '')
                if 'inverse' not in task_type.lower() or 'ordering' not in task_type.lower():
                    fixed_items.append(data_point)
                    continue
                
                gt_answer = data_point.get('gt_answer', [])
                question = data_point.get('question', '')
                
                # Skip if missing required fields
                if not gt_answer or not question:
                    self.stats['skipped'] += 1
                    fixed_items.append(data_point)
                    continue
                
                # Check if gt_answer needs fixing
                if self._is_gt_answer_invalid(gt_answer):
                    self.stats['invalid_gt_answers'] += 1
                    
                    try:
                        # Fix the gt_answer
                        fixed_gt_answer = self._fix_gt_answer(question, gt_answer)
                        data_point['gt_answer'] = fixed_gt_answer
                        # data_point['_original_gt_answer'] = gt_answer
                        self.stats['fixed_gt_answers'] += 1
                        
                    except Exception as e:
                        print(f"Line {line_num}: Error fixing - {str(e)}")
                        self.stats['errors'] += 1
                
                fixed_items.append(data_point)
                
            except Exception as e:
                print(f"Line {line_num}: Error processing - {str(e)}")
                self.stats['errors'] += 1
                continue
        
        # Write fixed data to output file
        with open(output_jsonl_path, 'w') as outfile:
            for item in fixed_items:
                json.dump(item, outfile)
                outfile.write('\n')
        
        self._print_summary()
    
    def fix_batch(self, input_dir: str, output_root: str) -> None:
        """Fix all JSONL files in a directory, maintaining directory structure."""
        input_path = Path(input_dir)
        output_root_path = Path(output_root)
        
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Find all JSONL files recursively
        jsonl_files = list(input_path.rglob("*.jsonl"))
        
        if not jsonl_files:
            print(f"No JSONL files found in {input_dir}")
            return
        
        print(f"Found {len(jsonl_files)} JSONL files to process")
        
        # Reset stats for batch processing
        total_stats = {
            'total_processed': 0,
            'invalid_gt_answers': 0,
            'fixed_gt_answers': 0,
            'errors': 0,
            'skipped': 0
        }
        
        for jsonl_file in jsonl_files:
            # Calculate relative path from input directory
            rel_path = jsonl_file.relative_to(input_path)
            
            # Create output path maintaining directory structure
            output_file = output_root_path / rel_path
            
            print(f"\n--- Processing: {rel_path} ---")
            
            # Reset individual file stats
            self.stats = {
                'total_processed': 0,
                'invalid_gt_answers': 0,
                'fixed_gt_answers': 0,
                'errors': 0,
                'skipped': 0
            }
            
            try:
                self.fix_jsonl_file(str(jsonl_file), str(output_file))
                
                # Add to total stats
                for key in total_stats:
                    total_stats[key] += self.stats[key]
                    
            except Exception as e:
                print(f"Error processing {rel_path}: {str(e)}")
                total_stats['errors'] += 1
        
        # Print overall batch summary
        print(f"\n{'='*60}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Files processed: {len(jsonl_files)}")
        print(f"Total items processed: {total_stats['total_processed']}")
        print(f"Invalid answers: {total_stats['invalid_gt_answers']}")
        print(f"Successfully fixed: {total_stats['fixed_gt_answers']}")
        print(f"Errors: {total_stats['errors']}")
        print(f"Skipped: {total_stats['skipped']}")
        print(f"{'='*60}")
    
    def _print_summary(self):
        """Print summary statistics."""
        print(f"\nSummary:")
        print(f"  Total processed: {self.stats['total_processed']}")
        print(f"  Invalid answers: {self.stats['invalid_gt_answers']}")
        print(f"  Successfully fixed: {self.stats['fixed_gt_answers']}")
        print(f"  Errors: {self.stats['errors']}")
        print(f"  Skipped: {self.stats['skipped']}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Fix invalid gt_answer sequences in JSONL files"
    )
    
    # Batch mode
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable batch processing mode"
    )
    
    # Single file mode arguments
    parser.add_argument(
        "--input_file", 
        help="Input JSONL file to fix",
        default="/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/QA/behavior_eqa_ordering.jsonl"
    )
    parser.add_argument(
        "--output_file", 
        help="Output JSONL file with fixed answers",
        default="/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/QA/fix/behavior_eqa_ordering_fixed.jsonl"
    )
    
    # Batch mode arguments
    parser.add_argument(
        "--input_dir",
        help="Input directory containing JSONL files (for batch mode)",
        default="/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/evaluation/model_outputs"
    )
    parser.add_argument(
        "--output_root",
        help="Output root directory (for batch mode)",
        default="/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/Fixed_data/evaluation/model_outputs"
    )
    
    args = parser.parse_args()
    
    # Create the fixer
    fixer = GTAnswerFixer()
    
    try:
        if args.batch:
            # Batch processing mode
            print("🔄 Batch processing mode enabled")
            fixer.fix_batch(args.input_dir, args.output_root)
            print(f"✓ Batch processing completed! Results written to: {args.output_root}")
        else:
            # Single file mode
            # Validate input file exists
            if not Path(args.input_file).exists():
                print(f"Error: Input file '{args.input_file}' does not exist")
                return 1
            
            print("📄 Single file processing mode")
            fixer.fix_jsonl_file(args.input_file, args.output_file)
            print(f"✓ Fixing completed! Results written to: {args.output_file}")
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
