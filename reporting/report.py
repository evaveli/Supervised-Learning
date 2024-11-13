import pandas as pd
import os
import time
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


def create_class_distribution_report(df, filename="class_distribution_report.pdf"):
    """
    Create a PDF report showing the class distribution statistics.
    """
    doc = SimpleDocTemplate(filename, pagesize=landscape(letter))
    styles = getSampleStyleSheet()
    elements = []

    # Add title
    elements.append(Paragraph("Class Distribution Analysis", styles["Title"]))
    elements.append(Paragraph("<br/><br/>", styles["Normal"]))

    # Calculate distribution
    class_column = "class" if "class" in df.columns else "Class"
    class_counts = df[class_column].value_counts().sort_index()
    total_samples = len(df)
    class_percentages = (class_counts / total_samples * 100).round(2)

    # Prepare table data
    headers = ["Class", "Count", "Percentage (%)", "Cumulative (%)"]
    table_data = [headers]

    cumulative_percentage = 0
    for class_label in class_counts.index:
        count = class_counts[class_label]
        percentage = class_percentages[class_label]
        cumulative_percentage += percentage
        row = [
            str(class_label),
            str(count),
            f"{percentage:.2f}",
            f"{cumulative_percentage:.2f}",
        ]
        table_data.append(row)

    # Add total row
    table_data.append(["Total", str(total_samples), "100.00", "100.00"])

    # Create table with specific column widths
    col_widths = [147, 147, 147, 147]
    table = Table(table_data, colWidths=col_widths)

    # Apply table style
    table.setStyle(get_default_table_style())

    elements.append(table)

    # Add footnotes
    elements.append(Spacer(1, 30))
    footnote_style = ParagraphStyle(
        "FootnoteStyle", parent=styles["Normal"], fontSize=8, textColor=colors.gray
    )

    footnote_text = "Generated on: " + time.strftime("%Y-%m-%d %H:%M:%S")
    footnote_text1 = "Count shows the number of instances in each class"
    footnote_text2 = "Percentage shows the relative frequency of each class"
    footnote_text3 = "Cumulative percentage shows the running total of percentages"
    footnote_text4 = "Created by Eva Veli, Niklas Long Schiefelbein, and Andras Kasa"

    elements.append(Paragraph(footnote_text, footnote_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(footnote_text1, footnote_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(footnote_text2, footnote_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(footnote_text3, footnote_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(footnote_text4, footnote_style))
    # Build the document
    doc.build(elements)


def get_default_table_style():
    """
    Returns the default table style for reports.

    The style includes center alignment, bold header with gray background, grid lines,
    and alternating row backgrounds for better readability.

    Returns:
        TableStyle: A ReportLab TableStyle object with predefined styling.
    """
    return TableStyle(
        [
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 11),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]
    )


def format_configuration(config):
    """
    Formats a configuration dictionary into a multi-line string for PDF readability.

    If a 'reduction_method' is specified, includes both SVM and reduction-specific parameters.
    Otherwise, returns the string representation of the entire configuration.

    Parameters:
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        str: Formatted configuration string with line breaks.
    """
    if "reduction_method" in config:
        # For reduced configurations (Wilson/GCNN/DROP3 + SVM)
        reduction_method = config["reduction_method"]
        if reduction_method == "Wilson":
            return (
                f"SVM: C={config['C']}, gamma={config['gamma']}\n"
                f"kernel={config['kernel']}\n"
                f"Wilson: k={config['wil_k']}, th={config['wil_th']}\n"
                f"p={config['wil_p']}, v={config['wil_v']}\n"
                f"d={config['wil_d']}, w={config['wil_w']}"
            )
        elif reduction_method == "GCNN":
            return (
                f"SVM: C={config['C']}, gamma={config['gamma']}\n"
                f"kernel={config['kernel']}\n"
                f"GCNN: k={config['gcnn_k']}, a={config['gcnn_a']}\n"
                f"p={config['gcnn_p']}, v={config['gcnn_v']}\n"
                f"d={config['gcnn_d']}, w={config['gcnn_w']}"
            )
        elif reduction_method == "DROP3":
            return (
                f"SVM: C={config['C']}, gamma={config['gamma']}\n"
                f"kernel={config['kernel']}\n"
                f"DROP3: k={config['drop3_k']}\n"
                f"p={config['drop3_p']}, v={config['drop3_v']}\n"
                f"d={config['drop3_d']}, w={config['drop3_w']}"
            )
    return str(config)  # For original configurations


def create_performance_report(results, filename="model_performance_report.pdf"):
    """
    Generates a PDF report of model performance metrics.

    The report includes a sorted table of models with their configurations and evaluation metrics,
    along with a timestamp and footnotes for clarity.

    Parameters:
        results (list of dict): List containing performance metrics and configurations for each model.
        filename (str, optional): Name of the output PDF file. Defaults to "model_performance_report.pdf".
    """
    doc = SimpleDocTemplate(filename, pagesize=landscape(letter))
    styles = getSampleStyleSheet()
    elements = []

    # Add title
    elements.append(Paragraph("Model Performance Report", styles["Title"]))
    elements.append(Paragraph("<br/><br/>", styles["Normal"]))

    # Create paragraph style for configuration cell
    config_style = ParagraphStyle(
        "ConfigStyle",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,  # Line spacing
        spaceAfter=6,
    )

    # Prepare table data
    headers = [
        "Model",
        "Configuration",
        "Prec.",
        "Recall",
        "Acc.",
        "F1 Scr.",
        "Std Dv",
        "Time (s)",
    ]
    table_data = [headers]

    results = sorted(
        results,
        key=lambda x: (
            x["precision"],
            x["recall"],
            x["accuracy"],
            -x["f1"],
            x["std_accuracy"],
            -x["mean_time"],
        ),
        reverse=True,
    )

    for result in results:
        # Format configuration with line breaks and wrap in Paragraph
        config_text = format_configuration(result["configuration"])
        config_para = Paragraph(config_text.replace("\n", "<br/>"), config_style)

        row = [
            result["method"],
            config_para,  # Use Paragraph object instead of string
            f"{result['precision']:.3f}",
            f"{result['recall']:.3f}",
            f"{result['accuracy']:.3f}",
            f"{result['f1']:.3f}",
            f"{result['std_accuracy']:.3f}",
            f"{result['mean_time']:.4f}",
        ]
        table_data.append(row)

    # Create table with adjusted column widths
    col_widths = [165, 165, 45, 45, 45, 45, 45, 45]  # Adjust these values as needed
    table = Table(table_data, repeatRows=1, colWidths=col_widths)

    # Apply the default table style
    table.setStyle(get_default_table_style())

    elements.append(table)

    # Add spacer before footnote
    elements.append(Spacer(1, 30))

    # Add footnote
    footnote_style = ParagraphStyle(
        "FootnoteStyle", parent=styles["Normal"], fontSize=8, textColor=colors.gray
    )
    footnote_text = "Generated on: " + time.strftime("%Y-%m-%d %H:%M:%S")
    footnote_text1 = "Prec.=Precision, Acc.=Accuracy, F1 Scr.=F1 Score, Std Dev.=Standard Deviation, V=Voting Method, D=Distance Metric, W=Weighting Method"
    footnote_text2 = "Created by Eva Veli, Niklas Long Schiefelbein, and Andras Kasa"
    elements.append(Spacer(1, 25))
    elements.append(Paragraph(footnote_text, footnote_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(footnote_text1, footnote_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(footnote_text2, footnote_style))

    # Build PDF
    doc.build(elements)
    print(f"Report generated: {filename}")


def create_nemenyi_report(friedman_results, filename="nemenyi_comparison_report.pdf"):
    """
    Generates a PDF report of pairwise model comparisons based on Nemenyi test results.

    The report includes a table comparing each pair of models, indicating whether their
    performance difference is statistically significant based on the critical difference.

    Parameters:
        friedman_results (dict): A dictionary containing:
            - 'sorted_results' (list of dict): Sorted list of models by performance.
            - 'ranks' (dict): Mapping of model names to their average ranks.
            - 'critical_difference' (float): The critical difference threshold.

        filename (str): The name of the output PDF file. Defaults to "nemenyi_comparison_report.pdf".

    """
    doc = SimpleDocTemplate(filename, pagesize=landscape(letter))
    styles = getSampleStyleSheet()
    elements = []

    elements.append(
        Paragraph("Nemenyi Test - Pairwise Model Comparisons", styles["Title"])
    )
    elements.append(Paragraph("<br/><br/>", styles["Normal"]))

    sorted_results = friedman_results[
        "sorted_results"
    ]  # This is already sorted by performance
    ranks = friedman_results["ranks"]
    cd = friedman_results["critical_difference"]

    # Create comparison table
    headers = ["Model A", "Model B", "Rank Diff", "Significant?"]
    table_data = [headers]

    # Use the order from sorted_results
    models = [
        result["method"] for result in sorted_results
    ]  # This preserves the correct order

    # Perform pairwise comparisons maintaining order
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model_a = models[i]  # Better performing model
            model_b = models[j]  # Worse performing model
            rank_diff = abs(ranks[model_a] - ranks[model_b])
            is_significant = rank_diff > cd

            row = [
                model_a,
                model_b,
                f"{rank_diff:.4f}",
                "Yes" if is_significant else "No",
            ]
            table_data.append(row)

    # Rest of the function remains the same...

    # Create table
    table = Table(table_data, repeatRows=1)
    # Apply the default table style
    table.setStyle(get_default_table_style())

    elements.append(table)

    # Add footnotes
    elements.append(Spacer(1, 30))
    footnote_style = ParagraphStyle(
        "FootnoteStyle", parent=styles["Normal"], fontSize=8, textColor=colors.gray
    )

    footnote_text = "Generated on: " + time.strftime("%Y-%m-%d %H:%M:%S")
    footnote_text1 = f"Critical Difference (CD) = {cd:.4f}"
    footnote_text2 = (
        "Significant difference exists when Rank Difference > Critical Difference"
    )
    footnote_text3 = "Created by Eva Veli, Niklas Long Schiefelbein, and Andras Kasa"

    elements.append(Paragraph(footnote_text, footnote_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(footnote_text1, footnote_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(footnote_text2, footnote_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(footnote_text3, footnote_style))
    # Build PDF
    doc.build(elements)
    print(f"Nemenyi test report generated: {filename}")


def knn_create_reduction_comparison_report(
    best_knn,
    best_wilson,
    best_gcnn,
    best_drop3,
    train_folds,
    wilson_reduced_folds,
    gcnn_reduced_folds,
    drop3_reduced_folds,
    wilson_time,
    gcnn_time,
    drop3_time,
    filename="reduction_comparison_report.pdf",
):
    """
    Generates a PDF report comparing different instance reduction methods.

    The report includes metrics such as Accuracy, Storage ratio, Processing Time, and Best Configurations
    for each reduction technique alongside the original KNN model.

    Parameters:
        best_knn (list of dict): Best KNN model results with metrics and configuration.
        best_wilson (list of dict): Best Wilson reduction method results.
        best_gcnn (list of dict): Best GCNN reduction method results.
        best_drop3 (list of dict): Best DROP3 reduction method results.
        train_folds (list): List of training data folds.
        wilson_reduced_folds (dict): Reduced training folds using Wilson method.
        gcnn_reduced_folds (dict): Reduced training folds using GCNN method.
        drop3_reduced_folds (dict): Reduced training folds using DROP3 method.
        wilson_time (float): Average processing time for Wilson reduction.
        gcnn_time (float): Average processing time for GCNN reduction.
        drop3_time (float): Average processing time for DROP3 reduction.
        filename (str): Output PDF filename. Defaults to "reduction_comparison_report.pdf".
    """
    doc = SimpleDocTemplate(filename, pagesize=landscape(letter))
    styles = getSampleStyleSheet()
    elements = []

    # Add title
    elements.append(
        Paragraph("Instance Reduction Methods Comparison for KNNs", styles["Title"])
    )
    elements.append(Paragraph("<br/><br/>", styles["Normal"]))

    # Get original dataset size (from first fold)
    original_size = len(train_folds[0])

    # Get best results from each method
    original_results = best_knn[0]
    wilson_results = best_wilson[0]
    gcnn_results = best_gcnn[0]
    drop3_results = best_drop3[0]

    wilson_key = (
        wilson_results["configuration"]["k"],
        wilson_results["configuration"]["threshold"],
    )
    wilson_size = len(wilson_reduced_folds[wilson_key][0])
    wilson_ratio = (wilson_size / original_size) * 100

    gcnn_key = gcnn_results["configuration"]["alpha"]
    gcnn_size = len(gcnn_reduced_folds[gcnn_key][0])
    gcnn_ratio = (gcnn_size / original_size) * 100

    drop3_key = drop3_results["configuration"]["k"]
    drop3_size = len(drop3_reduced_folds[drop3_key][0])
    drop3_ratio = (drop3_size / original_size) * 100

    # Prepare table data
    headers = ["Metric", "Original (KNN)", "Wilson", "GCNN", "DROP3"]
    table_data = [headers]

    # Add accuracy row
    accuracy_row = [
        "Accuracy (%)",
        f"{original_results['accuracy']*100:.2f}",
        f"{wilson_results['accuracy']*100:.2f}",
        f"{gcnn_results['accuracy']*100:.2f}",
        f"{drop3_results['accuracy']*100:.2f}",
    ]

    # Modified storage row to use the calculated ratios directly
    storage_row = [
        "Storage (%)",
        "100.00",
        f"{wilson_ratio:.2f}",  # Use the ratio we calculated
        f"{gcnn_ratio:.2f}",  # Use the ratio we calculated
        f"{drop3_ratio:.2f}",
    ]  # Use the ratio we calculated

    # Add time row
    time_row = [
        "Time (s)",
        f"{original_results['mean_time']:.4f}",
        f"{wilson_results['mean_time']:.4f}",
        f"{gcnn_results['mean_time']:.4f}",
        f"{drop3_results['mean_time']:.4f}",
    ]

    process_time_row = [
        "Total Process (s)",
        "N/A",  # Original KNN doesn't have reduction time
        f"{wilson_time:.2f}",
        f"{gcnn_time:.2f}",
        f"{drop3_time:.2f}",
    ]

    config_style = ParagraphStyle(
        "ConfigStyle",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,  # Line spacing
        alignment=1,  # Center alignment
    )
    # Add configuration row
    config_row = [
        "Best Config",
        Paragraph(
            format_configuration(original_results["configuration"]).replace(
                "\n", "<br/>"
            ),
            config_style,
        ),
        Paragraph(
            format_configuration(wilson_results["configuration"]).replace(
                "\n", "<br/>"
            ),
            config_style,
        ),
        Paragraph(
            format_configuration(gcnn_results["configuration"]).replace("\n", "<br/>"),
            config_style,
        ),
        Paragraph(
            format_configuration(drop3_results["configuration"]).replace("\n", "<br/>"),
            config_style,
        ),
    ]

    table_data.extend(
        [accuracy_row, storage_row, process_time_row, time_row, config_row]
    )

    # Create table with specific column widths
    col_widths = [95, 147, 147, 147, 147]
    table = Table(table_data, colWidths=col_widths)

    # Apply the default table style
    table.setStyle(get_default_table_style())

    elements.append(table)

    # Add footnotes
    elements.append(Spacer(1, 30))
    footnote_style = ParagraphStyle(
        "FootnoteStyle", parent=styles["Normal"], fontSize=8, textColor=colors.gray
    )

    footnote_text = "Generated on: " + time.strftime("%Y-%m-%d %H:%M:%S")
    footnote_text1 = (
        "Storage ratio shows the percentage of instances retained after reduction"
    )
    footnote_text2 = "V=Voting Method, D=Distance Metric, W=Weighting Method"
    footnote_text3 = "Time represents the average processing time per fold"
    footnote_text4 = "Created by Eva Veli, Niklas Long Schiefelbein, and Andras Kasa"

    elements.append(Paragraph(footnote_text, footnote_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(footnote_text1, footnote_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(footnote_text2, footnote_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(footnote_text3, footnote_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(footnote_text4, footnote_style))
    # Build PDF
    doc.build(elements)
    print(f"Reduction comparison report generated: {filename}")


def svm_create_reduction_comparison_report(
    best_svm,
    best_wilson_svm,
    best_gcnn_svm,
    best_drop3_svm,
    train_folds,
    wilson_reduced_folds,
    gcnn_reduced_folds,
    drop3_reduced_folds,
    wilson_time,
    gcnn_time,
    drop3_time,
    filename="reduction_comparison_report_svm.pdf",
):
    """
    Generates a PDF report comparing different instance reduction methods applied to SVM.

    The report includes metrics such as Accuracy, Storage Ratio, Processing Time, and Best Configurations
    for each reduction technique (Wilson, GCNN, DROP3) alongside the original SVM model.

    Parameters:
        best_svm (list of dict): Best SVM model results containing metrics and configuration.
        best_wilson_svm (list of dict): Best Wilson reduction method results for SVM.
        best_gcnn_svm (list of dict): Best GCNN reduction method results for SVM.
        best_drop3_svm (list of dict): Best DROP3 reduction method results for SVM.
        train_folds (list of pd.DataFrame): List of training data folds.
        wilson_reduced_folds (dict): Reduced training folds using Wilson method, keyed by (k, threshold).
        gcnn_reduced_folds (dict): Reduced training folds using GCNN method, keyed by alpha.
        drop3_reduced_folds (dict): Reduced training folds using DROP3 method, keyed by k.
        wilson_time (float): Average processing time for Wilson reduction across folds.
        gcnn_time (float): Average processing time for GCNN reduction across folds.
        drop3_time (float): Average processing time for DROP3 reduction across folds.
        filename (str): Output PDF filename. Defaults to "reduction_comparison_report_svm.pdf".

    """
    doc = SimpleDocTemplate(filename, pagesize=landscape(letter))
    styles = getSampleStyleSheet()
    elements = []

    # Add title
    elements.append(
        Paragraph("Instance Reduction Methods Comparison for SVMs", styles["Title"])
    )
    elements.append(Paragraph("<br/><br/>", styles["Normal"]))

    # Get original dataset size (from first fold)
    original_size = len(train_folds[0])

    # Get best results from each method
    original_results = best_svm[0]
    wilson_results = best_wilson_svm[0]
    gcnn_results = best_gcnn_svm[0]
    drop3_results = best_drop3_svm[0]

    wilson_key = (
        wilson_results["configuration"]["wil_k"],
        wilson_results["configuration"]["wil_th"],
    )
    wilson_size = len(wilson_reduced_folds[wilson_key][0])
    wilson_ratio = (wilson_size / original_size) * 100

    gcnn_key = gcnn_results["configuration"]["gcnn_a"]
    gcnn_size = len(gcnn_reduced_folds[gcnn_key][0])
    gcnn_ratio = (gcnn_size / original_size) * 100

    drop3_key = drop3_results["configuration"]["drop3_k"]
    drop3_size = len(drop3_reduced_folds[drop3_key][0])
    drop3_ratio = (drop3_size / original_size) * 100

    # Prepare table data
    headers = ["Metric", "Original (SVM)", "Wilson", "GCNN", "DROP3"]
    table_data = [headers]

    # Add accuracy row
    accuracy_row = [
        "Accuracy (%)",
        f"{original_results['accuracy']*100:.2f}",
        f"{wilson_results['accuracy']*100:.2f}",
        f"{gcnn_results['accuracy']*100:.2f}",
        f"{drop3_results['accuracy']*100:.2f}",
    ]

    # Modified storage row to use the calculated ratios directly
    storage_row = [
        "Storage (%)",
        "100.00",
        f"{wilson_ratio:.2f}",  # Use the ratio we calculated
        f"{gcnn_ratio:.2f}",  # Use the ratio we calculated
        f"{drop3_ratio:.2f}",
    ]  # Use the ratio we calculated

    # Add time row
    time_row = [
        "Time (s)",
        f"{original_results['mean_time']:.4f}",
        f"{wilson_results['mean_time']:.4f}",
        f"{gcnn_results['mean_time']:.4f}",
        f"{drop3_results['mean_time']:.4f}",
    ]

    process_time_row = [
        "Total Process (s)",
        "N/A",  # Original KNN doesn't have reduction time
        f"{wilson_time:.2f}",
        f"{gcnn_time:.2f}",
        f"{drop3_time:.2f}",
    ]

    config_style = ParagraphStyle(
        "ConfigStyle",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,  # Line spacing
        alignment=1,  # Center alignment
    )
    # Add configuration row
    config_row = [
        "Best Config",
        Paragraph(
            format_configuration(original_results["configuration"]).replace(
                "\n", "<br/>"
            ),
            config_style,
        ),
        Paragraph(
            format_configuration(wilson_results["configuration"]).replace(
                "\n", "<br/>"
            ),
            config_style,
        ),
        Paragraph(
            format_configuration(gcnn_results["configuration"]).replace("\n", "<br/>"),
            config_style,
        ),
        Paragraph(
            format_configuration(drop3_results["configuration"]).replace("\n", "<br/>"),
            config_style,
        ),
    ]

    table_data.extend(
        [accuracy_row, storage_row, process_time_row, time_row, config_row]
    )

    # Create table with specific column widths
    col_widths = [95, 147, 147, 147, 147]
    table = Table(table_data, colWidths=col_widths)

    # Apply the default table style
    table.setStyle(get_default_table_style())

    elements.append(table)

    # Add footnotes
    elements.append(Spacer(1, 30))
    footnote_style = ParagraphStyle(
        "FootnoteStyle", parent=styles["Normal"], fontSize=8, textColor=colors.gray
    )

    footnote_text = "Generated on: " + time.strftime("%Y-%m-%d %H:%M:%S")
    footnote_text1 = (
        "Storage ratio shows the percentage of instances retained after reduction"
    )
    footnote_text2 = "V=Voting Method, D=Distance Metric, W=Weighting Method"
    footnote_text3 = "Time represents the average processing time per fold"
    footnote_text4 = "Created by Eva Veli, Niklas Long Schiefelbein, and Andras Kasa"

    elements.append(Paragraph(footnote_text, footnote_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(footnote_text1, footnote_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(footnote_text2, footnote_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(footnote_text3, footnote_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(footnote_text4, footnote_style))
    # Build PDF
    doc.build(elements)
    print(f"Reduction comparison report generated: {filename}")


def aggregate_results_by_x(results, x, output_dir):
    """
    Aggregates results over all folds, computing mean metrics grouped by voting scheme.

    Parameters:
        results: list of dictionaries, each containing 'voting_method' and 'metrics'
        x: str, the key to group results by
        output_dir: str, the directory to save the aggregated results to

    Returns:
        aggregated_results: dictionary mapping voting methods to mean metrics
    """
    from collections import defaultdict
    import numpy as np

    # Initialize a dictionary to collect metrics per voting method
    x_metrics = defaultdict(list)

    # Collect metrics per voting method
    for result in results:
        voting_method = result[x]
        mean_accuracy = result["mean_accuracy"]
        std_accuracy = result["std_accuracy"]
        mean_time = result["mean_time"]
        mean_precision = result["mean_precision"]
        mean_recall = result["mean_recall"]
        metrics = {
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "mean_time": mean_time,
            "mean_precision": mean_precision,
            "mean_recall": mean_recall,
        }
        x_metrics[voting_method].append(metrics)

    # Compute mean metrics per voting method
    aggregated_results = {}
    for voting_method, metrics_list in x_metrics.items():
        # Compute mean of each metric
        mean_metrics = {}
        for key in metrics_list[0]:
            mean_metrics[key] = np.mean([metrics[key] for metrics in metrics_list])
        aggregated_results[voting_method] = mean_metrics

    pd.DataFrame(aggregated_results).to_csv(
        os.path.join(output_dir, f"mean_for_parameter_{x}.csv")
    )

    return aggregated_results


def create_summary_statistics_report(
    summary_table,
    title,
    p_value,
    critical_difference,
    filename="summary_statistics.pdf",
):
    """
    Create a PDF report with summary statistics

    Parameters:
    -----------
    summary_table : pandas DataFrame
        Table containing the summary statistics
    title : str
        Title for the report
    filename : str
        Name of the output PDF file
    """
    p_value = float(p_value)
    critical_difference = float(critical_difference)

    # Create the document
    doc = SimpleDocTemplate(filename, pagesize=letter)
    elements = []

    # Add title
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Heading1"], fontSize=14, spaceAfter=30
    )
    test_style = ParagraphStyle(
        "TestResults", parent=styles["Normal"], fontSize=11, spaceAfter=20
    )

    significance_text = f"Friedman Test p-value: {p_value:.4f}"
    if p_value < 0.05:
        significance_text += " (Significant differences found!)"
        significance_text += (
            f"\nNemenyi Test Critical difference: {critical_difference:.4f}"
        )
    else:
        significance_text += " (No significant differences found)"

    elements.append(Paragraph(title, title_style))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(significance_text, test_style))
    elements.append(Spacer(1, 12))

    # Convert DataFrame to list for table
    data = [summary_table.columns.tolist()] + summary_table.values.tolist()

    # Format numeric columns to 4 decimal places
    for i in range(1, len(data)):  # Skip header row
        for j in range(1, 5):  # Columns 1-4 are numeric
            data[i][j] = f"{data[i][j]:.4f}"

    # Create table
    table = Table(data)

    # Apply the default table style
    table.setStyle(get_default_table_style())

    elements.append(table)

    # Add summary text
    summary_style = ParagraphStyle(
        "Summary", parent=styles["Normal"], fontSize=10, spaceAfter=30
    )
    elements.append(Spacer(1, 20))

    # Add method comparisons
    methods = summary_table["Method"].tolist()
    accuracies = summary_table["Mean Accuracy"].astype(float)
    best_method = methods[accuracies.argmax()]
    worst_method = methods[accuracies.argmin()]

    # Create footer with summary statistics
    best_text = (
        f"Best performing method: {best_method} (accuracy: {accuracies.max():.4f})"
    )
    worst_text = (
        f"Worst performing method: {worst_method} (accuracy: {accuracies.min():.4f})"
    )
    diff_text = f"Accuracy difference: {accuracies.max() - accuracies.min():.4f}"

    footer_text = f"{best_text}<br/>" f"{worst_text}<br/>" f"{diff_text}"
    footnote_style = ParagraphStyle(
        "FootnoteStyle", parent=styles["Normal"], fontSize=8, textColor=colors.gray
    )

    footnote_text = "Generated on: " + time.strftime("%Y-%m-%d %H:%M:%S")
    footnote_text1 = "Created by Eva Veli, Niklas Long Schiefelbein, and Andras Kasa"
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(footer_text, summary_style))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(footnote_text, footnote_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(footnote_text1, footnote_style))

    # Build PDF
    doc.build(elements)


def create_cd_diagram(method_names, rankings, cd_value, filename="cd_diagram.pdf"):
    """
    Create a Critical Difference diagram with standard format and handling overlapping labels
    """
    plt.figure(figsize=(10, 3))
    ax = plt.gca()

    # Set up the plot
    max_rank = 3.0
    min_rank = 1.0
    ax.set_xlim(max_rank, min_rank)  # Reversed x-axis
    ax.set_ylim(0, 2.5)  # Increased y-limit to accommodate staggered labels

    # Remove all spines except bottom
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Remove y-axis
    ax.yaxis.set_visible(False)

    # Add CD scale at the top
    cd_start = 2.0  # Starting position for CD
    plt.plot([cd_start, cd_start + cd_value], [2.2, 2.2], "k-", linewidth=1)
    plt.plot([cd_start, cd_start], [2.15, 2.25], "k-", linewidth=1)
    plt.plot(
        [cd_start + cd_value, cd_start + cd_value], [2.15, 2.25], "k-", linewidth=1
    )
    plt.text(cd_start + cd_value / 2, 2.3, f"CD = {cd_value:.2f}", ha="center")

    # Plot method names with staggered heights for overlapping positions
    base_y_pos = 1.4
    used_positions = {}

    # Sort methods by rank to handle overlaps from left to right
    method_rank_pairs = list(zip(method_names, rankings))
    method_rank_pairs.sort(key=lambda x: x[1], reverse=True)

    label_positions = {}  # Store final y-positions for connection lines

    # Increased threshold for detecting overlaps and offset
    overlap_threshold = 0.4  # Increased from 0.2
    y_step = 0.3  # Increased from 0.2

    for name, rank in method_rank_pairs:
        y_offset = 0
        while True:
            current_pos = round(rank, 2)
            if (
                current_pos in used_positions
                and abs(used_positions[current_pos] - (base_y_pos + y_offset))
                < overlap_threshold
            ):
                y_offset += y_step
            else:
                break

        plt.text(rank, base_y_pos + y_offset, name, ha="center", va="center")
        used_positions[current_pos] = base_y_pos + y_offset
        label_positions[name] = (rank, base_y_pos + y_offset)

    # Draw lines connecting methods that are not significantly different
    for i, (name1, rank1) in enumerate(zip(method_names, rankings)):
        for name2, rank2 in zip(method_names[i + 1 :], rankings[i + 1 :]):
            if abs(rank1 - rank2) < cd_value:
                # Use minimum y-position of connected labels for the line
                y = min(label_positions[name1][1], label_positions[name2][1]) - 0.1
                plt.plot([rank1, rank2], [y, y], "k-", linewidth=1)

    # Add x-axis with rank values
    plt.xticks(np.arange(min_rank, max_rank + 0.25, 0.25))

    # Add bottom line
    plt.axhline(y=0.5, color="k", linewidth=1)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()


def create_hyperparameter_cd_diagram(
    knn_results, method_key, methods_list, method_name, report_dir, filename=None
):
    """
    Create a CD diagram for comparing hyperparameter performance
    """
    # Group results by method
    grouped_results = {method: [] for method in methods_list}

    for result in knn_results:
        method = str(result[method_key])
        if method in methods_list:
            grouped_results[method].append(result["mean_accuracy"])

    # Calculate average ranks
    all_methods_accuracies = [
        np.array(grouped_results[method]) for method in methods_list
    ]
    n_configs = len(all_methods_accuracies[0])

    # Calculate ranks for each configuration
    ranks = []
    for i in range(n_configs):
        config_accuracies = [accs[i] for accs in all_methods_accuracies]
        config_ranks = stats.rankdata([-acc for acc in config_accuracies])
        ranks.append(config_ranks)

    # Calculate average ranks
    avg_ranks = np.mean(ranks, axis=0)

    # Calculate Critical Difference
    k = len(methods_list)
    n = n_configs
    q_alpha = 2.89  # for alpha=0.05
    cd_value = q_alpha * np.sqrt((k * (k + 1)) / (6 * n))

    if filename is None:
        filename = f"{method_name.lower().replace(' ', '_')}_cd_diagram.pdf"

    create_cd_diagram(
        methods_list, avg_ranks, cd_value, os.path.join(report_dir, filename)
    )

    # Print numerical results
    print(f"\nAverage ranks for {method_name}:")
    for method, rank in zip(methods_list, avg_ranks):
        print(f"{method}: {rank:.3f}")
    print(f"Critical Difference: {cd_value:.3f}")
