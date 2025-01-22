"""Output column."""

OUTPUT_COLUMN = "output"
OUTPUT_PROB_COLUMN_PREFIX = "output_prob_"


def output_prob_column(class_idx: int) -> str:
    """Produce a column name for the output probability for a class."""
    return f"{OUTPUT_PROB_COLUMN_PREFIX}{class_idx}"
