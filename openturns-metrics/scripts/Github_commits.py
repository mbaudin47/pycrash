"""
Plots the commits per week of several software.
This is based on GitHub history:
https://github.com/SGpp/SGpp/graphs/contributors?all=1
https://github.com/cossan-working-group/OpenCossan/graphs/contributors?all=1
https://github.com/UCL-CCS/EasyVVUQ/graphs/contributors?all=1
https://github.com/snl-dakota/dakota/graphs/contributors?all=1
https://github.com/lanl/GPMSA/graphs/contributors?all=1
https://github.com/openturns/openturns/graphs/contributors?all=1
https://github.com/llnl/psuade/graphs/contributors?all=1
https://github.com/libqueso/queso/graphs/contributors?all=1
https://github.com/idaholab/raven/graphs/contributors?all=1
https://github.com/SURGroup/UQpy/graphs/contributors?all=1
https://github.com/jonathf/chaospy/graphs/contributors?all=1
https://github.com/sandialabs/UQTk/graphs/contributors?all=1

"""

# %%
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import itertools


# %%
def plot_commit_evolution(
    data_directory,
    frequency="W",
    smoothing_window=None,
    figsize=(8, 4),
    glob_pattern="*Commits_over_time.csv",
    cleanup_pattern="_Commits_over_time.csv",
):
    """
    Reads commit files from the specified directory, sorts software
    by descending total commit count, and plots their evolution over time.

    Parameters:
    -----------
    data_directory : str or Path
        Path to the directory containing the *_Commits_over_time.csv files.
    frequency : str, default 'W'
        Resampling frequency for the data ('W', 'M', 'MS', 'Y', 'YS', etc.).
    smoothing_window : int, optional
        Window size for the moving average. If None, no smoothing is applied.
    figsize : tuple, default (8, 4)
        Dimensions of the Matplotlib figure.
    """
    # 1. File search
    pattern = os.path.join(data_directory, glob_pattern)
    files = glob.glob(pattern)

    print(f"Found {len(files)} files:")
    print(files)

    df_list = []

    # 2. Reading and data extraction
    for file in files:
        if os.path.exists(file):
            file_name = Path(file).name
            software_name = file_name.replace(cleanup_pattern, "")

            # Reading the CSV file
            df = pd.read_csv(file, sep=";")

            # Converting the date column to datetime objects
            df["Week of"] = pd.to_datetime(df["Week of"])

            # Adding the software name to identify the data
            df["Software"] = software_name

            df_list.append(df)

    # 3. Consolidation, resampling, and data sorting
    if df_list:
        df_global = pd.concat(df_list, ignore_index=True)

        # Pivoting data
        df_pivot = df_global.pivot(
            index="Week of", columns="Software", values="Commits"
        )

        # Replacing missing values before resampling
        df_pivot = df_pivot.fillna(0)

        # Resampling according to frequency and summing commits
        df_resampled = df_pivot.resample(frequency).sum()

        # Calculating total commits per software to establish ranking
        total_commits = df_resampled.sum().sort_values(ascending=False)

        # Reordering DataFrame columns based on descending ranking
        df_resampled = df_resampled[total_commits.index]

        # Applying smoothing if specified (after sorting to preserve order)
        smoothing_title = ""
        if smoothing_window is not None and smoothing_window > 1:
            df_resampled = df_resampled.rolling(
                window=smoothing_window, min_periods=1
            ).mean()
            smoothing_title = f" ({smoothing_window}-period moving average)"

        # Determining the time label for axes and title
        freq_dict = {
            "W": "per week",
            "M": "per month",
            "MS": "per month",
            "Y": "per year",
            "YS": "per year",
        }
        time_label = freq_dict.get(frequency, "")

        # 4. Graphical representation with Matplotlib
        plt.figure(figsize=figsize)

        # Defining available line styles and creating a cycler
        line_styles = ["-", "--", "-.", ":"]
        style_cycle = itertools.cycle(line_styles)

        # Plotting curves in ranking order
        for software in df_resampled.columns:
            current_style = next(style_cycle)
            software_total = int(total_commits[software])

            plt.plot(
                df_resampled.index,
                df_resampled[software],
                label=f"{software} ({software_total} commits)",
                linestyle=current_style,
                linewidth=1.7,
            )

        plt.title(f"Commit Evolution over Time {time_label}{smoothing_title}")
        plt.xlabel("Date")
        plt.ylabel(f"Number of Commits {time_label}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(
            title="Software (Total)", loc="upper left", bbox_to_anchor=(1.0, 1.0)
        )
        plt.tight_layout()

        # Displaying the plot
        plt.show()
    else:
        print("No matching files were found.")


# %%
plot_commit_evolution("../data")

# %%
plot_commit_evolution("../data", frequency="M")

# %%
plot_commit_evolution("../data", frequency="Y")

# %%
plot_commit_evolution("../data", frequency="M", smoothing_window=8)

# %%
plot_commit_evolution("../data", frequency="Y", smoothing_window=2)
plt.savefig("../figures/Github_commits.png", bbox_inches="tight")
plt.savefig("../figures/Github_commits.pdf", bbox_inches="tight")
