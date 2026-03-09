import logging
import math
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from experiments.tools.instantiate_list import instantiate_list
from learn2clean.actions.data_frame_action import DataFrameAction
from learn2clean.configs import register_all_configs
from learn2clean.spaces.permutation_space import PermutationSpace

# Initialize Rich Console for pretty output
console = Console(width=140)
log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="tutorials/07_permutation_space",
)
def main(cfg: DictConfig) -> None:
    """
    Tutorial 07: Visualizing the Permutation Space.

    Demonstrates the math behind the 'One-Shot' environment.
    We visualize how a set of atomic actions creates a massive combinatorial space,
    and how we can efficiently sample from it using the Factoradic Number System.
    """

    # --- 1. INTRODUCTION ---
    console.print(
        Panel.fit(
            "[bold cyan]Tutorial 07: The Permutation Space[/bold cyan]\n"
            "[italic]Visualizing the combinatorial explosion of cleaning pipelines[/italic]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    console.print(
        "\n[bold]Goal:[/bold] Demonstrate how a single integer index represents a complex "
        "data cleaning pipeline without generating all possibilities in memory.\n"
    )

    # --- 2. SETUP & INSTANTIATION ---
    with console.status(
        "[bold green]Instantiating actions and space...", spinner="dots"
    ):
        # Instantiate atomic actions from config
        actions: list[DataFrameAction] = instantiate_list(cfg.actions)

        # Create the PermutationSpace
        permutation_space = PermutationSpace(actions)

    # Display Atomic Actions
    action_table = Table(title="Atomic Actions Available", box=box.SIMPLE)
    action_table.add_column("ID", justify="center", style="cyan", no_wrap=True)
    action_table.add_column("Action Name", style="magenta")

    for i, action in enumerate(actions):
        action_table.add_row(str(i), action.name)

    console.print(action_table)

    # --- 3. SPACE ANALYSIS (THE REVEAL) ---
    console.print("\n[bold underline]Space Cardinality Analysis[/bold underline]")

    # Create a table for the 'Buckets' (permutations by length)
    stats_table = Table(box=box.ROUNDED, show_footer=True)
    stats_table.add_column("Pipeline Length", justify="center")
    stats_table.add_column("Combinations (Count)", justify="right")
    stats_table.add_column("Share of Space", justify="right")

    for length, count in enumerate(permutation_space.bucket_counts, start=1):
        percentage = (count / permutation_space.n) * 100
        stats_table.add_row(f"Length {length}", f"{count:,}", f"{percentage:.2f}%")

    # Add total footprint
    stats_table.columns[1].footer = Text(
        f"{permutation_space.n:,}", style="bold yellow"
    )
    stats_table.columns[0].footer = Text("TOTAL SPACE SIZE", style="bold")

    console.print(stats_table)

    # Mathematical Verification
    theoretical_total = sum(
        math.perm(len(actions), k) for k in range(1, len(actions) + 1)
    )
    is_correct = permutation_space.n == theoretical_total

    msg = (
        (
            f"[bold green]✓ Verified:[/bold green] Internal Calculation ({permutation_space.n:,}) "
            f"== Sum P(n,k) ({theoretical_total:,})"
        )
        if is_correct
        else "[bold red]Error in calculation![/bold red]"
    )

    console.print(Panel(msg, expand=False))

    # --- 4. SAMPLING DEMO (UNRANKING) ---
    console.print(
        "\n[bold underline]Sampling Demonstration (Unranking)[/bold underline]"
    )
    console.print(
        "We simulate an agent picking random integers and decoding them into executable pipelines.\n"
    )

    if "seed" in cfg.experiment:
        permutation_space.seed(cfg.experiment.seed)

    sample_table = Table(title="Randomly Sampled Pipelines", show_lines=True)
    sample_table.add_column("#", justify="center", style="dim")
    sample_table.add_column("Sampled Index (Action ID)", justify="right", style="cyan")
    sample_table.add_column("Decoded Pipeline (Executable)", style="green")

    for i in range(1, 6):
        # 1. Sample a random integer
        sample_idx = permutation_space.sample()

        # 2. Decode it (Unrank)
        action_set: tuple[DataFrameAction, ...] = permutation_space.idx_to_permutation(
            sample_idx
        )

        # Format the pipeline string
        pipeline_str = " -> ".join([f"[bold]{a.name}[/bold]" for a in action_set])

        sample_table.add_row(str(i), f"{sample_idx:,}", pipeline_str)

    console.print(sample_table)

    # --- 5. BEST PRACTICES & LIMITATIONS ---
    console.print("\n[bold underline]Expert Recommendations[/bold underline]")

    # Définition du contenu textuel
    warning_text = (
        "[bold yellow]1. Technical Hard Limit (20 items):[/bold yellow]\n"
        "   Do not exceed 20 atomic actions. 21! (factorial) exceeds the capacity of a "
        "64-bit integer, causing memory overflow errors.\n\n"
        "[bold orange1]2. Practical RL Limit (~10 items):[/bold orange1]\n"
        "   For standard algorithms like PPO, a space with >10 items results in "
        "millions of combinations. Without masking or curriculum learning, "
        "convergence will be extremely slow.\n\n"
        "[bold green]3. Next Steps:[/bold green]\n"
        "   To see how to plug this space into a Gymnasium environment and train an agent, "
        "please check the next tutorial: [italic]0x_gymnasium_integration.py[/italic]"
    )

    console.print(
        Panel(
            warning_text,
            title="Usage Guidelines",
            border_style="yellow",
            expand=False,
        )
    )

    console.print("\n[bold green]Tutorial completed successfully.[/bold green]")


if __name__ == "__main__":
    # Set environment for Hydra relative paths
    os.environ["PROJECT_ROOT"] = str(Path(__file__).resolve().parents[2])
    register_all_configs()
    main()
