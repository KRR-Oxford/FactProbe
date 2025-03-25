


def analysis_to_latex(
    analysis_question: dict,
    analysis_statement: dict,
    direction: str,
    relation_name: str,
) -> list[str]:
    """Generate LaTeX table rows for the analysis results."""

    latex_rows = []

    # Add the multirow header
    num_rows = "1" if direction == "high2high" else "3"
    latex_rows.append(f"\\multirow{{{num_rows}}}{{*}}{{\\textsf{{{relation_name}}}}}")

    # Add data rows
    for k in analysis_question:
        forward_acc_question = analysis_question[k]["forward_acc"]
        backward_acc_question = analysis_question[k]["backward_acc"]
        forward_acc_statement = analysis_statement[k]["forward_acc"]
        backward_acc_statement = analysis_statement[k]["backward_acc"]
        total_question = analysis_question[k]["total"]
        total_statement = analysis_statement[k]["total"]
        assert total_question == total_statement

        row = (
            f"& {k} & "
            f"{analysis_question[k]['total']} & "
            f"{forward_acc_question:.3f} & "
            f"{backward_acc_question:.3f} & "
            f"{analysis_question[k]['diff_arrow']} & "
            f"{analysis_question[k]['stat_sig']} & "
            f"{forward_acc_statement:.3f} & "
            f"{backward_acc_statement:.3f} & "
            f"{analysis_statement[k]['diff_arrow']} & "
            f"{analysis_statement[k]['stat_sig']} \\\\"
        )
        latex_rows.append(row)

    return latex_rows
