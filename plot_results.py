import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "results"
GRAPH_DIR = os.path.join(RESULTS_DIR, "graphs")

os.makedirs(GRAPH_DIR, exist_ok=True)


def read_csv_safely(path):
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return None
    return pd.read_csv(path)


def get_value(df, column, default=np.nan):
    if df is None or column not in df.columns:
        return default

    value = df[column].iloc[0]

    if str(value).strip().upper() in ["NA", "NONE", "NAN", ""]:
        return default

    try:
        return float(value)
    except Exception:
        return default


def save_bar_chart(title, labels, values, ylabel, filename):
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=15)
    plt.tight_layout()

    output_path = os.path.join(GRAPH_DIR, filename)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def save_grouped_bar_chart(title, categories, pytorch_values, libtorch_values, ylabel, filename):
    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, pytorch_values, width, label="PyTorch")
    plt.bar(x + width / 2, libtorch_values, width, label="LibTorch")

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(x, categories, rotation=15)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(GRAPH_DIR, filename)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def calculate_temperature_cost(temperature_c, baseline_temp=40.0, temp_factor=0.01):
    if pd.isna(temperature_c):
        return 0.0

    extra_temp = max(0.0, float(temperature_c) - baseline_temp)
    return extra_temp * temp_factor


def calculate_gcf(
    power_watt,
    time_seconds,
    memory_percent,
    temperature_c=np.nan,
    carbon_factor=0.000417,
    memory_factor=0.001
):
    if pd.isna(power_watt):
        power_watt = 5.0

    if pd.isna(time_seconds):
        time_seconds = 0.0

    if pd.isna(memory_percent):
        memory_percent = 0.0

    time_hours = float(time_seconds) / 3600.0

    energy_carbon_cost = float(power_watt) * time_hours * carbon_factor
    memory_cost = float(memory_percent) * memory_factor
    temperature_cost = calculate_temperature_cost(temperature_c)

    return energy_carbon_cost + memory_cost + temperature_cost


def main():
    pytorch_deploy = read_csv_safely(
        os.path.join(RESULTS_DIR, "pytorch_deployment_metrics.csv")
    )

    libtorch_deploy = read_csv_safely(
        os.path.join(RESULTS_DIR, "libtorch_deployment_metrics.csv")
    )

    pytorch_prediction = read_csv_safely(
        os.path.join(RESULTS_DIR, "pytorch_prediction_summary.csv")
    )

    libtorch_prediction = read_csv_safely(
        os.path.join(RESULTS_DIR, "libtorch_prediction_summary.csv")
    )

    # ==================================================
    # Deployment Phase Comparison
    # ==================================================
    if pytorch_deploy is not None and libtorch_deploy is not None:

        pytorch_loading_time = get_value(
            pytorch_deploy,
            "model_loading_time_seconds"
        )

        libtorch_loading_time = get_value(
            libtorch_deploy,
            "model_loading_time_seconds"
        )

        save_bar_chart(
            title="Model Loading Time Comparison",
            labels=["PyTorch", "LibTorch"],
            values=[pytorch_loading_time, libtorch_loading_time],
            ylabel="Time (seconds)",
            filename="01_deployment_loading_time_comparison.png"
        )

        pytorch_model_size = get_value(
            pytorch_deploy,
            "model_file_size_mb"
        )

        libtorch_model_size = get_value(
            libtorch_deploy,
            "model_file_size_mb"
        )

        save_bar_chart(
            title="Model File Size Comparison",
            labels=["PyTorch", "LibTorch"],
            values=[pytorch_model_size, libtorch_model_size],
            ylabel="Size (MB)",
            filename="02_model_file_size_comparison.png"
        )

        pytorch_memory_increase = get_value(
            pytorch_deploy,
            "memory_increase_mb"
        )

        libtorch_memory_increase = get_value(
            libtorch_deploy,
            "memory_increase_mb"
        )

        save_bar_chart(
            title="Deployment Memory Increase Comparison",
            labels=["PyTorch", "LibTorch"],
            values=[pytorch_memory_increase, libtorch_memory_increase],
            ylabel="Memory Increase (MB)",
            filename="03_deployment_memory_increase_comparison.png"
        )

        pytorch_deploy_cpu = get_value(
            pytorch_deploy,
            "avg_cpu_percent"
        )

        libtorch_deploy_cpu = get_value(
            libtorch_deploy,
            "estimated_cpu_percent"
        )

        pytorch_deploy_mem = get_value(
            pytorch_deploy,
            "avg_memory_percent"
        )

        libtorch_deploy_mem = get_value(
            libtorch_deploy,
            "avg_memory_percent"
        )

        save_grouped_bar_chart(
            title="Deployment CPU and Memory Usage",
            categories=["CPU Usage (%)", "Memory Usage (%)"],
            pytorch_values=[pytorch_deploy_cpu, pytorch_deploy_mem],
            libtorch_values=[libtorch_deploy_cpu, libtorch_deploy_mem],
            ylabel="Percentage",
            filename="04_deployment_cpu_memory_comparison.png"
        )

        pytorch_deployment_gcf = calculate_gcf(
            power_watt=5.0,
            time_seconds=pytorch_loading_time,
            memory_percent=pytorch_deploy_mem,
            temperature_c=get_value(pytorch_deploy, "avg_temperature_c")
        )

        libtorch_deployment_gcf = calculate_gcf(
            power_watt=5.0,
            time_seconds=libtorch_loading_time,
            memory_percent=libtorch_deploy_mem,
            temperature_c=get_value(libtorch_deploy, "temperature_c")
        )

        save_bar_chart(
            title="Deployment GCF Score Comparison",
            labels=["PyTorch", "LibTorch"],
            values=[pytorch_deployment_gcf, libtorch_deployment_gcf],
            ylabel="GCF Score",
            filename="05_deployment_gcf_comparison.png"
        )

    # ==================================================
    # Post-Deployment Prediction Phase Comparison
    # ==================================================
    if pytorch_prediction is not None and libtorch_prediction is not None:

        pytorch_accuracy = get_value(
            pytorch_prediction,
            "prediction_accuracy"
        )

        libtorch_accuracy = get_value(
            libtorch_prediction,
            "prediction_accuracy"
        )

        save_bar_chart(
            title="Post-Deployment Prediction Accuracy",
            labels=["PyTorch", "LibTorch"],
            values=[pytorch_accuracy, libtorch_accuracy],
            ylabel="Accuracy",
            filename="06_prediction_accuracy_comparison.png"
        )

        pytorch_total_time = get_value(
            pytorch_prediction,
            "total_prediction_time_seconds"
        )

        libtorch_total_time = get_value(
            libtorch_prediction,
            "total_prediction_time_seconds"
        )

        save_bar_chart(
            title="Total Prediction Time Comparison",
            labels=["PyTorch", "LibTorch"],
            values=[pytorch_total_time, libtorch_total_time],
            ylabel="Time (seconds)",
            filename="07_total_prediction_time_comparison.png"
        )

        pytorch_latency = get_value(
            pytorch_prediction,
            "average_latency_seconds_per_image"
        )

        libtorch_latency = get_value(
            libtorch_prediction,
            "average_latency_seconds_per_image"
        )

        save_bar_chart(
            title="Average Prediction Latency per Image",
            labels=["PyTorch", "LibTorch"],
            values=[pytorch_latency, libtorch_latency],
            ylabel="Latency (seconds/image)",
            filename="08_average_latency_comparison.png"
        )

        pytorch_throughput = get_value(
            pytorch_prediction,
            "throughput_images_per_second"
        )

        libtorch_throughput = get_value(
            libtorch_prediction,
            "throughput_images_per_second"
        )

        save_bar_chart(
            title="Prediction Throughput Comparison",
            labels=["PyTorch", "LibTorch"],
            values=[pytorch_throughput, libtorch_throughput],
            ylabel="Images per second",
            filename="09_prediction_throughput_comparison.png"
        )

        pytorch_pred_cpu = get_value(
            pytorch_prediction,
            "avg_cpu_percent"
        )

        libtorch_pred_cpu = get_value(
            libtorch_prediction,
            "estimated_cpu_percent"
        )

        pytorch_pred_mem = get_value(
            pytorch_prediction,
            "avg_memory_percent"
        )

        libtorch_pred_mem = get_value(
            libtorch_prediction,
            "avg_memory_percent"
        )

        save_grouped_bar_chart(
            title="Prediction CPU and Memory Usage",
            categories=["CPU Usage (%)", "Memory Usage (%)"],
            pytorch_values=[pytorch_pred_cpu, pytorch_pred_mem],
            libtorch_values=[libtorch_pred_cpu, libtorch_pred_mem],
            ylabel="Percentage",
            filename="10_prediction_cpu_memory_comparison.png"
        )

        pytorch_prediction_gcf = get_value(
            pytorch_prediction,
            "prediction_gcf_score"
        )

        libtorch_prediction_gcf = get_value(
            libtorch_prediction,
            "prediction_gcf_score"
        )

        save_bar_chart(
            title="Prediction GCF Score Comparison",
            labels=["PyTorch", "LibTorch"],
            values=[pytorch_prediction_gcf, libtorch_prediction_gcf],
            ylabel="GCF Score",
            filename="11_prediction_gcf_comparison.png"
        )

    # ==================================================
    # Final Comparison CSV
    # ==================================================
    comparison_rows = []

    if pytorch_deploy is not None and libtorch_deploy is not None:
        comparison_rows.extend([
            {
                "phase": "Deployment",
                "metric": "Model Loading Time (s)",
                "pytorch": get_value(pytorch_deploy, "model_loading_time_seconds"),
                "libtorch": get_value(libtorch_deploy, "model_loading_time_seconds")
            },
            {
                "phase": "Deployment",
                "metric": "Model File Size (MB)",
                "pytorch": get_value(pytorch_deploy, "model_file_size_mb"),
                "libtorch": get_value(libtorch_deploy, "model_file_size_mb")
            },
            {
                "phase": "Deployment",
                "metric": "Memory Increase (MB)",
                "pytorch": get_value(pytorch_deploy, "memory_increase_mb"),
                "libtorch": get_value(libtorch_deploy, "memory_increase_mb")
            }
        ])

    if pytorch_prediction is not None and libtorch_prediction is not None:
        comparison_rows.extend([
            {
                "phase": "Prediction",
                "metric": "Prediction Accuracy",
                "pytorch": get_value(pytorch_prediction, "prediction_accuracy"),
                "libtorch": get_value(libtorch_prediction, "prediction_accuracy")
            },
            {
                "phase": "Prediction",
                "metric": "Total Prediction Time (s)",
                "pytorch": get_value(pytorch_prediction, "total_prediction_time_seconds"),
                "libtorch": get_value(libtorch_prediction, "total_prediction_time_seconds")
            },
            {
                "phase": "Prediction",
                "metric": "Average Latency (s/image)",
                "pytorch": get_value(pytorch_prediction, "average_latency_seconds_per_image"),
                "libtorch": get_value(libtorch_prediction, "average_latency_seconds_per_image")
            },
            {
                "phase": "Prediction",
                "metric": "Throughput (images/s)",
                "pytorch": get_value(pytorch_prediction, "throughput_images_per_second"),
                "libtorch": get_value(libtorch_prediction, "throughput_images_per_second")
            },
            {
                "phase": "Prediction",
                "metric": "Prediction GCF Score",
                "pytorch": get_value(pytorch_prediction, "prediction_gcf_score"),
                "libtorch": get_value(libtorch_prediction, "prediction_gcf_score")
            }
        ])

    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_path = os.path.join(RESULTS_DIR, "final_comparison_table.csv")
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Saved: {comparison_path}")

    print("\nAll comparison graphs generated successfully.")
    print(f"Graph folder: {GRAPH_DIR}")


if __name__ == "__main__":
    main()