import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def correlation_analysis(
    df: pd.DataFrame,
    output_dir: str = ".",
    correlation_method: str = "pearson",
    plot_heatmap: bool = True,
    heatmap_filename: str = "correlation_heatmap.png",
    matrix_filename: str = "correlation_matrix.csv",
) -> pd.DataFrame:
    """
    Computes and saves a correlation matrix for the given DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Compute correlation matrix - numeric only
    corr_matrix = df.corr(method=correlation_method)

    # Save matrix to CSV
    matrix_path = os.path.join(output_dir, matrix_filename)
    corr_matrix.to_csv(matrix_path, index=True)
    print(f"[INFO] Correlation matrix saved to {matrix_path}")

    # Plot heatmap
    if plot_heatmap:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, annot=True, cmap="RdBu", center=0, fmt=".2f", square=True
        )
        plt.title(f"{correlation_method.capitalize()} Correlation Heatmap")
        heatmap_path = os.path.join(output_dir, heatmap_filename)
        plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Correlation heatmap saved to {heatmap_path}")

    return corr_matrix
