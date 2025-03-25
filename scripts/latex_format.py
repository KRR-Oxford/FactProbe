#!/usr/bin/env python3

import click
from pathlib import Path
import json
from typing import Dict, Any, List

RELATIONS = ["P190", "P26", "P3373", "P47"]
RELATION_NAME = { "P3373": "sibling", "P47": "bordersWith", "P26": "spouse", "P190": "twinnedTown", "P460": "saidSameAs" }
DIRECTIONS = ["high2low", "low2high", "high2high"]


def format_latex_rows(
    relation: str,
    analysis_question: Dict[str, Any],
    analysis_statement: Dict[str, Any],
    direction: str
) -> List[str]:
    """Format analysis results for a single relation into LaTeX rows.

    Args:
        relation: The relation name (e.g., 'P26')
        analysis_question: Analysis results for questions
        analysis_statement: Analysis results for statements
        direction: Direction of analysis ('high2low', 'low2high', or 'high2high')

    Returns:
        List of LaTeX formatted rows
    """
    latex_rows = []
    
    # Add the multirow header
    num_rows = "1" if direction == "high2high" else "3"
    latex_rows.append(f"\\multirow{{{num_rows}}}{{*}}{{\\textsf{{{RELATION_NAME[relation]}}}}}")
    
    # Add data rows
    for k in analysis_question[direction]:
        data_q = analysis_question[direction][k]
        data_s = analysis_statement[direction][k]
        
        row = (
            f"& {k} & "
            f"{data_q['total']} & "
            f"{data_q['forward_acc']:.3f} & "
            f"{data_q['backward_acc']:.3f} & "
            f"{data_q['diff_arrow']} & "
            f"{data_q['stat_sig']} & "
            f"{data_s['forward_acc']:.3f} & "
            f"{data_s['backward_acc']:.3f} & "
            f"{data_s['diff_arrow']} & "
            f"{data_s['stat_sig']} \\\\"
        )
        latex_rows.append(row)
    
    latex_rows.append("\\midrule")
    return latex_rows


@click.command()
@click.argument('model', type=str)
@click.argument('temp', type=click.Choice(['0', '1']))
def main(model: str, temp: str):
    """Generate LaTeX table rows for all relations in a specific model and temperature setting.
    
    MODEL: Name of the model (e.g., 'OLMo-2-0325-32B-Instruct')
    TEMP: Temperature setting (0 or 1)
    """
    analysis_dir = Path("analysis")
    
    # Print header
    print("% LaTeX table rows for each direction")
    
    for direction in DIRECTIONS:
        print(f"\n% {direction} direction")
        print("% Copy these rows into your LaTeX table")
        print("% Format: relation & range & samples & Q-F & Q-B & Q-Diff & Q-Sig & S-F & S-B & S-Diff & S-Sig \\\\")
        
        for relation in RELATIONS:
            # Load analysis results
            question_file = analysis_dir / relation / model / f"{relation}_all_question_temp{temp}.analysis.json"
            statement_file = analysis_dir / relation / model / f"{relation}_all_statement_temp{temp}.analysis.json"
            
            try:
                with open(question_file) as f:
                    analysis_question = json.load(f)
                with open(statement_file) as f:
                    analysis_statement = json.load(f)
                    
                latex_rows = format_latex_rows(
                    relation,
                    analysis_question,
                    analysis_statement,
                    direction
                )
                
                # Print the rows
                if direction != 'high2high':
                    for row in latex_rows:
                        print(row)
                else:
                    print(latex_rows[0])
                    print(latex_rows[1].replace("0-1K", "$\geq$100K"))
                    print(latex_rows[-1])
                    
            except FileNotFoundError:
                print(f"% Warning: Could not find analysis files for {relation}")
                continue
        
        print("\n% End of rows for", direction)


if __name__ == "__main__":
    main()
