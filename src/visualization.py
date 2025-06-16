import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_plots(df, results, output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)

    heatmap_path = os.path.join(output_dir, "heatmap.png")
    scatter_path = os.path.join(output_dir, "scatter.png")

    # Heatmap
    fig1, ax1 = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax1)
    fig1.tight_layout()
    fig1.savefig(heatmap_path)
    plt.close(fig1)

    # Scatter
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x='concentration', y='behavior_metric', hue='group', ax=ax2)
    fig2.tight_layout()
    fig2.savefig(scatter_path)
    plt.close(fig2)

    return {
        "heatmap": heatmap_path,
        "scatter": scatter_path
    }
