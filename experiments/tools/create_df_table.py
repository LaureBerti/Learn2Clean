import pandas as pd
from rich import box
from rich.table import Table


def create_df_table(
    df: pd.DataFrame, title: str = "Dataset Preview", rows: int = 5
) -> Table:
    table = Table(title=title, box=box.ROUNDED, show_lines=True)

    table.add_column("Index", justify="right", style="cyan", no_wrap=True)
    for col in df.columns:
        style = "magenta" if pd.api.types.is_numeric_dtype(df[col]) else "green"
        table.add_column(str(col), style=style, overflow="fold")

    for idx, row in df.head(rows).iterrows():
        row_values = [str(idx)] + [str(x) for x in row.values]
        table.add_row(*row_values)

    return table
