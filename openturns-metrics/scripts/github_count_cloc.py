# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import squarify

# %%
# 1. Data loading
# Replace 'github_count_cloc.csv' with the actual path to your file
df = pd.read_csv("github_count_cloc.csv")

# %%
# Cleaning potential spaces in column names or strings
df.columns = df.columns.str.strip()
df["software"] = df["software"].str.strip()
df["language"] = df["language"].str.strip()

# Filtering to ignore CSV, SVG, HTML, XML, and JSON files
df = df[~df["language"].isin(["CSV", "SVG", "HTML", "XML", "JSON"])]

# %%
# ==============================================================================
# CHART 1: Total lines of code per software
# ==============================================================================
code_totals = df.groupby("software")["code"].sum().sort_values(ascending=False)

plt.figure(figsize=(40, 4))
bars = plt.bar(
    code_totals.index, code_totals.values, color="skyblue", edgecolor="grey"
)

plt.title(
    "Total Lines of Code per Software", 
)
plt.xlabel("Software")
plt.ylabel("Lines of Code")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Adding values above the bars
max_cloc = 0
for bar in bars:
    y_val = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        y_val + (y_val * 0.01),
        f"{y_val:,}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    max_cloc = max(max_cloc, y_val)
plt.ylim(0, max_cloc * 1.1)
plt.savefig("../figures/github_count_cloc_total.png", bbox_inches="tight")
plt.savefig("../figures/github_count_cloc_total.pdf", bbox_inches="tight")
plt.show()

# %%
# ==============================================================================
# CHART 2: Treemaps per project
# ==============================================================================
unique_software = df["software"].unique()

for software in unique_software:
    df_software = df[df["software"] == software].copy()

    # Sorting by descending code volume
    df_software = df_software.sort_values(by="code", ascending=False)

    # Calculating percentage for proper filtering or labeling
    total_software = df_software["code"].sum()
    df_software["pourcentage"] = (df_software["code"] / total_software) * 100

    # To avoid visual clutter, group languages < 1.5% under "Others"
    threshold = 1.5
    main_languages = df_software[df_software["pourcentage"] >= threshold].copy()
    others = df_software[df_software["pourcentage"] < threshold]

    if not others.empty:
        new_row = pd.DataFrame(
            {
                "software": [software],
                "language": ["Others"],
                "files": [others["files"].sum()],
                "empty": [others["empty"].sum()],
                "comments": [others["comments"].sum()],
                "code": [others["code"].sum()],
                "pourcentage": [others["pourcentage"].sum()],
            }
        )
        df_visualization = pd.concat([main_languages, new_row], ignore_index=True)
    else:
        df_visualization = main_languages

    # Preparing labels (Name + Percentage)
    labels = [
        f"{row['language']}\n{row['code']:,} lines\n({row['pourcentage']:.1f}%)"
        for _, row in df_visualization.iterrows()
    ]

    # Generating a distinct color palette
    colors = plt.cm.tab20(np.linspace(0, 1, len(df_visualization)))

    # Creating the figure for the current software treemap
    plt.figure(figsize=(5, 5))
    squarify.plot(
        sizes=df_visualization["code"],
        label=labels,
        color=colors,
        alpha=0.8,
        edgecolor="white",
        linewidth=2,
        text_kwargs={"fontsize": 10, "weight": "bold"},
    )

    plt.title(
        f"Language Distribution in Project: {software}\n(Total: {total_software:,} lines of code)",
        fontsize=14,
        fontweight="bold",
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"../figures/github_count_cloc_{software}_fractions.png", bbox_inches="tight")
    plt.savefig(f"../figures/github_count_cloc_{software}_fractions.pdf", bbox_inches="tight")
    plt.show()
# %%