import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.tree import Tree

from experiments.tools.create_df_table import create_df_table

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from learn2clean.configs import register_all_configs
from learn2clean.loaders.base import DatasetLoader

# Initialize Rich console
console = Console(width=160)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="dataset/titanic_csv",
)
def main(cfg: DictConfig) -> None:
    console.clear()
    console.rule("[bold blue]🔍 DATASET INSPECTION[/]")

    # -----------------------------------------------------------
    # A. Config Extraction
    # -----------------------------------------------------------
    # Handles two cases:
    # 1. Running a full global config (cfg.dataset exists)
    # 2. Running a specific dataset config file (cfg IS the dataset config)
    dataset_cfg = cfg.dataset if "dataset" in cfg else cfg

    # Display config as colored YAML
    yaml_str = OmegaConf.to_yaml(dataset_cfg)
    syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True)

    console.print(
        Panel(
            syntax,
            title=f"[bold]Resolved Configuration[/] (target: [yellow]{dataset_cfg.get('_target_', 'N/A')}[/])",
            expand=False,
            border_style="blue",
        )
    )

    # -----------------------------------------------------------
    # B. Instantiation & Loading (With Spinner)
    # -----------------------------------------------------------
    loader: DatasetLoader | None = None
    df = None

    # Try/Except block lets Rich handle the error display beautifully
    try:
        with console.status("[bold green]Loading data...[/]", spinner="dots"):
            # Simulate a small delay if loading is too fast to see the spinner
            # time.sleep(0.5)
            loader = instantiate(dataset_cfg)
            df = loader.load()

    except Exception:
        console.print_exception(show_locals=True)
        return

    console.print(f"\n[bold green]✅ Loading successful![/]")

    # -----------------------------------------------------------
    # C. Analysis
    # -----------------------------------------------------------
    rows, cols = df.shape

    # Create a tree for key metadata
    tree = Tree(f"📊 [bold]DataFrame Metadata[/]")
    tree.add(f"Dimensions: [bold cyan]{rows}[/] rows x [bold cyan]{cols}[/] columns")

    # --- Target Analysis ---
    target = getattr(loader, "target_col", None)
    target_node = tree.add("🎯 [bold]Target Analysis[/]")

    if target:
        if target in df.columns:
            target_node.label = Text(
                f"🎯 Target: '{target}' (Found)", style="bold green"
            )

            # Details
            n_unique = df[target].nunique()
            dtype = df[target].dtype

            target_node.add(f"Type: [cyan]{dtype}[/]")
            target_node.add(f"Unique Values: [cyan]{n_unique}[/]")

            if n_unique < 20:
                dist_node = target_node.add("Distribution")
                counts = df[target].value_counts()
                for val, count in counts.items():
                    dist_node.add(f"{val}: [magenta]{count}[/]")
        else:
            target_node.label = Text(
                f"⚠️ Target '{target}' configured but MISSING!", style="bold red"
            )
            # Suggestion
            target_node.add(
                f"[dim]Available columns: {', '.join(list(df.columns)[:5])}...[/]"
            )
    else:
        target_node.label = Text("⚪ No target defined (Unsupervised)", style="dim")

    console.print(tree)
    console.print("")

    # -----------------------------------------------------------
    # D. Visual Preview (Table)
    # -----------------------------------------------------------
    table = create_df_table(df)
    console.print(table)

    console.rule("[bold blue]End of inspection[/]")


if __name__ == "__main__":
    os.environ["PROJECT_ROOT"] = str(Path(__file__).resolve().parents[2])
    register_all_configs()
    main()
