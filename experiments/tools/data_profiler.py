import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig
from ydata_profiling import ProfileReport

log = logging.getLogger(__name__)


class DataProfiler:
    """
    A unified wrapper around 'ydata-profiling' to generate standardized data reports
    within the Hydra experiment workflow.

    This class handles:
    - Configuration management via Hydra (toggle on/off, minimal mode, etc.).
    - Automatic output directory creation relative to the experiment run.
    - Exporting reports in multiple formats (HTML, JSON) and console summaries.
    """

    def __init__(self, cfg: DictConfig, root_dir: Path | str | None = None):
        """
        Initialize the DataProfiler with Hydra configuration settings.

        Args:
            cfg (DictConfig): The 'profiler' section of the Hydra config.
                              Must contain keys like 'enabled', 'minimal', 'dir', etc.
            root_dir (Path | str | None): The base directory where the profiling folder
                                          will be created. Ideally set to `cfg.paths.run_output_dir`
                                          to keep reports inside the specific experiment folder.
                                          If None, defaults to the current working directory.
        """
        self.enabled = cfg.get("enabled", False)

        # Determine the base path: use provided root_dir or fallback to CWD
        base_path = Path(root_dir) if root_dir else Path.cwd()

        # Construct the full output path
        # Example: .../outputs/2023-10-27/10-00-00/profiling_reports
        self.output_dir = base_path / cfg.get("dir", "profiling_reports")

        # Load configuration flags
        self.minimal = cfg.get("minimal", True)
        self.title_base = cfg.get("title", "Pandas Profiling Report")
        self.save_html = cfg.get("html", True)
        self.save_json = cfg.get("json", False)
        self.console_summary = cfg.get("console_summary", False)

        # Create the output directory immediately if profiling is enabled
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"Profiler output directory ready at: {self.output_dir}")

    def run(self, df: pd.DataFrame, step_name: str = "dataset") -> None:
        """
        Generates and saves a profile report for the provided DataFrame.

        Args:
            df (pd.DataFrame): The pandas DataFrame to analyze.
            step_name (str): A distinct name for this profiling step (e.g., 'raw', 'cleaned').
                             This name is appended to the output filename (e.g., 'profile_raw.html').
        """
        # 1. Early Exit: Check if disabled via config
        if not self.enabled:
            log.info(f"Profiling disabled. Skipping report for '{step_name}'.")
            return

        # 2. Safety Check: Ensure DataFrame is valid
        if df is None or df.empty:
            log.warning(
                f"DataFrame is empty or None. Skipping profile for '{step_name}'."
            )
            return

        log.info(
            f"Generating ProfileReport for '{step_name}' (minimal={self.minimal})..."
        )

        try:
            # 3. Configure Report
            # Customize title to distinguish between raw and cleaned data reports
            title = f"{self.title_base} - {step_name.capitalize()}"

            profile = ProfileReport(
                df,
                title=title,
                minimal=self.minimal,
                explorative=not self.minimal,
                lazy=False,  # Force immediate calculation to avoid threading/pickling issues in some envs
            )

            # 4. Export to HTML
            if self.save_html:
                output_file = self.output_dir / f"profile_{step_name}.html"
                profile.to_file(output_file)
                log.info(f"HTML Report saved to: {output_file}")

            # 5. Export to JSON (Optional, useful for programmatic parsing)
            if self.save_json:
                output_file = self.output_dir / f"profile_{step_name}.json"
                profile.to_file(output_file)
                log.info(f"JSON Report saved to: {output_file}")

            # 6. Console Summary (Optional quick view)
            if self.console_summary:
                description = profile.get_description()
                table_stats = description.table

                log.info(f"--- Summary for {step_name} ---")
                log.info(f"Variables: {table_stats['n_var']}")
                log.info(f"Observations: {table_stats['n']}")
                log.info(
                    f"Missing cells: {table_stats['n_cells_missing']} "
                    f"({table_stats['p_cells_missing']:.1%})"
                )
                log.info("-----------------------------")

        except Exception as e:
            log.error(f"Failed to generate profile report for {step_name}: {e}")
