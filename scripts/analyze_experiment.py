from pathlib import Path
from typing import Dict, Any

import click
import pandas as pd
from deeponto.utils import load_file, save_file
from factprobe.utils.analysis import analyse_results_all_freqs, freq_dict_from_triple_df


def analyze_experiment(
    results_path: str | Path,
    triple_df_path: str | Path,
    output_path: str | Path | None = None
) -> Dict[str, Any]:
    """Analyze experiment results using a separate triple DataFrame.

    Args:
        results_path: Path to the .pkl file containing experiment results
        triple_df_path: Path to the .pkl file containing triple DataFrame
        output_path: Optional path to save the analysis results. If None,
                    will save in the same directory as the input file.

    Returns:
        Dictionary containing the analysis results
    """
    # Load the experiment results and triple DataFrame
    results = load_file(results_path)
    triple_df = pd.read_csv(triple_df_path)
    
    # If output_path is not specified, save in the same directory
    if output_path is None:
        output_path = Path(results_path).with_suffix('.analysis.json')
    
    # Extract frequency dictionary from triple_df
    freq_dict = freq_dict_from_triple_df(triple_df)
    
    # Analyze results for different directions
    analysis_results = {}
    for direction in ['high2low', 'low2high', 'high2high']:
        analysis_results[direction] = analyse_results_all_freqs(
            results,
            freq_dict,
            direction
        )
    
    # Save the analysis results
    save_file(analysis_results, output_path)
    click.echo(f"Analysis results saved to: {output_path}")
    
    return analysis_results


@click.command()
@click.argument('results_path', type=click.Path(exists=True))
@click.argument('triple_df_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Path to save the analysis results')
def main(results_path: str, triple_df_path: str, output: str | None):
    """Analyze experiment results from .pkl files.
    
    RESULTS_PATH: Path to the .pkl file containing experiment results
    TRIPLE_DF_PATH: Path to the .pkl file containing triple DataFrame
    """
    analyze_experiment(results_path, triple_df_path, output)


if __name__ == "__main__":
    main()
