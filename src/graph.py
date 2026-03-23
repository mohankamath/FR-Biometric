import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULTS_FILE = "data/benchmark_results.csv"
OUT_GRAPH = "data/benchmark_graphs.png"

def create_graphs():
    if not os.path.exists(RESULTS_FILE):
        print(f"Cannot find {RESULTS_FILE}. Run the experiment script first.")
        return

    df = pd.read_csv(RESULTS_FILE)
    sns.set_theme(style="whitegrid")
    
    metrics = {
        "Accuracy": "System Accuracy (Higher is better)",
        "FAR": "False Acceptance Rate (Lower is better)",
        "FRR": "False Rejection Rate (Lower is better)"
    }

    plt.figure(figsize=(18, 5))

    for i, (metric, title) in enumerate(metrics.items(), 1):
        plt.subplot(1, 3, i)
        
        sns.lineplot(
            data=df, 
            x="Subjects", 
            y=metric, 
            hue="Model", 
            marker="o",
            linewidth=2
        )
        
        plt.title(title, fontsize=12, pad=10)
        plt.xlabel("Number of Subjects in Database")
        plt.ylabel(metric)
        plt.xticks(df["Subjects"].unique())
        
        if metric == "Accuracy":
            plt.ylim(0.8, 1.05) 
        
    plt.tight_layout()
    plt.savefig(OUT_GRAPH, dpi=300)
    print(f"Graph saved successfully to {OUT_GRAPH}")
    plt.show()

if __name__ == "__main__":
    create_graphs()