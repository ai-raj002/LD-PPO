import pandas as pd
import numpy as np
from datetime import datetime
import io
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def export_results_to_csv(results_dict, algorithm_name):
    """Export algorithm results to CSV"""
    if not results_dict or len(results_dict) == 0:
        return None
    
    df = pd.DataFrame(results_dict)
    csv_string = df.to_csv(index=False)
    return csv_string

def export_comparison_to_csv(comparison_data):
    """Export comparison results to CSV"""
    # Create comparison DataFrame
    metrics_df = pd.DataFrame(comparison_data.get('metrics', {})).T
    metrics_df.index.name = 'Algorithm'

    # Build primary CSV for algorithm metrics
    main_csv = metrics_df.to_csv()

    # If dataset-level metrics exist, write them as a separate two-column table (Metric,Value)
    dataset_metrics = comparison_data.get('dataset_metrics', {}) if comparison_data else {}
    if dataset_metrics:
        dm_df = pd.DataFrame(list(dataset_metrics.items()), columns=['Metric', 'Value'])
        # Append a blank line and then the dataset metrics block to keep CSV parsable
        buffer = io.StringIO()
        buffer.write(main_csv)
        buffer.write('\n')
        dm_df.to_csv(buffer, index=False)
        return buffer.getvalue()

    return main_csv

def generate_comparison_report(comparison_data, algorithms_metrics, output_format='pdf'):
    """Generate a comprehensive comparison report"""
    
    if output_format == 'pdf':
        return generate_pdf_report(comparison_data, algorithms_metrics)
    elif output_format == 'html':
        return generate_html_report(comparison_data, algorithms_metrics)
    else:
        return None

def generate_pdf_report(comparison_data, algorithms_metrics):
    """Generate PDF report using ReportLab"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    
    # Title
    story.append(Paragraph("Load Balancing in Cloud Computing Using Deep Reinforcement Learning (PPO)", title_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("Algorithm Comparison Report", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    # Date
    date_style = ParagraphStyle(
        'DateStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.grey,
        alignment=1
    )
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", date_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    if 'improvement' in comparison_data:
        improvement = comparison_data['improvement']
        summary_text = f"""
        This report compares multiple load balancing algorithms. Key findings:
        <br/><br/>
        <b>PPO Performance:</b> {improvement.get('avg_reward', 0):+.2f}% reward improvement
        <br/>
        <b>CPU Usage Reduction:</b> {improvement.get('avg_cpu', 0):+.2f}%
        <br/>
        <b>Memory Usage Reduction:</b> {improvement.get('avg_mem', 0):+.2f}%
        <br/>
        <b>Bandwidth Usage Reduction:</b> {improvement.get('avg_bw', 0):+.2f}%
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
    
    # Performance Metrics Table
    story.append(Paragraph("Performance Metrics Comparison", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    if 'metrics' in comparison_data:
        metrics_df = pd.DataFrame(comparison_data['metrics']).T
        metrics_data = [['Metric'] + list(metrics_df.columns)]
        
        for metric in metrics_df.index:
            row = [metric] + [f"{val:.4f}" if isinstance(val, (int, float)) else str(val) for val in metrics_df.loc[metric].values]
            metrics_data.append(row)
        
        metrics_table = Table(metrics_data)
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Dataset-level metrics (if present)
    if comparison_data and 'dataset_metrics' in comparison_data and comparison_data['dataset_metrics']:
        story.append(Paragraph("Dataset-level Metrics (selected window)", styles['Heading2']))
        dm = comparison_data['dataset_metrics']
        dm_data = [["Metric", "Value"]]
        for k, v in dm.items():
            dm_data.append([k, f"{v}" if not isinstance(v, (dict, list)) else str(v)])
        dm_table = Table(dm_data)
        dm_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        story.append(dm_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Algorithm Details
    story.append(Paragraph("Algorithm Details", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    for algo_name, metrics in algorithms_metrics.items():
        story.append(Paragraph(f"{algo_name} Algorithm", styles['Heading3']))
        story.append(Spacer(1, 0.05*inch))
        # Safe formatting helper to avoid exceptions when values are None or non-numeric
        def _fmt_num(x, fmt="{:.4f}"):
            try:
                if x is None:
                    return "N/A"
                if isinstance(x, (int, float)):
                    return fmt.format(x)
                # if it's a string already
                return str(x)
            except Exception:
                return str(x)

        def _fmt_pct(x, fmt="{:.2%}"):
            try:
                if x is None:
                    return "N/A"
                if isinstance(x, (int, float)):
                    return fmt.format(x)
                return str(x)
            except Exception:
                return str(x)

        details = (
            f"Total Requests: {metrics.get('total_requests', 0):,}<br/>"
            f"Average Reward: {_fmt_num(metrics.get('avg_reward', None), '{:.4f}')}<br/>"
            f"Average CPU Usage: {_fmt_pct(metrics.get('avg_cpu_usage', None), '{:.2%}')}<br/>"
            f"Average Memory Usage: {_fmt_pct(metrics.get('avg_mem_usage', None), '{:.2%}')}<br/>"
            f"Average Bandwidth Usage: {_fmt_pct(metrics.get('avg_bw_usage', None), '{:.2%}')}<br/>"
            f"Load Balance Index: {_fmt_num(metrics.get('load_balance_index', None), '{:.4f}')}")
        story.append(Paragraph(details, styles['Normal']))
        story.append(Spacer(1, 0.15*inch))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_html_report(comparison_data, algorithms_metrics):
    """Generate HTML report"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Load Balancing in Cloud Computing Using Deep Reinforcement Learning (PPO) - Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #1f77b4; text-align: center; }}
            h2 {{ color: #333; border-bottom: 2px solid #1f77b4; padding-bottom: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #1f77b4; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metric {{ margin: 10px 0; }}
            .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Load Balancing in Cloud Computing Using Deep Reinforcement Learning (PPO)</h1>
        <h2>Algorithm Comparison Report</h2>
        <p style="text-align: center; color: #666;">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>Executive Summary</h2>
    """
    
    if 'improvement' in comparison_data:
        improvement = comparison_data['improvement']
        html_content += f"""
            <p><strong>PPO Performance:</strong> {improvement.get('avg_reward', 0):+.2f}% reward improvement</p>
            <p><strong>CPU Usage Reduction:</strong> {improvement.get('avg_cpu', 0):+.2f}%</p>
            <p><strong>Memory Usage Reduction:</strong> {improvement.get('avg_mem', 0):+.2f}%</p>
            <p><strong>Bandwidth Usage Reduction:</strong> {improvement.get('avg_bw', 0):+.2f}%</p>
        """
    
    html_content += """
        </div>
        
        <h2>Performance Metrics Comparison</h2>
        <table>
            <tr>
                <th>Metric</th>
    """
    
    if 'metrics' in comparison_data:
        metrics_df = pd.DataFrame(comparison_data['metrics']).T
        for col in metrics_df.columns:
            html_content += f"<th>{col}</th>"
        html_content += "</tr>"
        
        for metric in metrics_df.index:
            html_content += f"<tr><td><strong>{metric}</strong></td>"
            for val in metrics_df.loc[metric].values:
                if isinstance(val, (int, float)):
                    html_content += f"<td>{val:.4f}</td>"
                else:
                    html_content += f"<td>{val}</td>"
            html_content += "</tr>"
    
    html_content += """
        </table>
        
        <h2>Dataset-level Metrics (selected window)</h2>
        <table>
"""

    # Insert dataset metrics if present
    if comparison_data and 'dataset_metrics' in comparison_data and comparison_data['dataset_metrics']:
        dm = comparison_data['dataset_metrics']
        for k, v in dm.items():
            html_content += f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>"
        html_content += "</table>"
    else:
        html_content += "</table>"
    html_content += """
        <h2>Algorithm Details</h2>
    """
    
    for algo_name, metrics in algorithms_metrics.items():
        html_content += f"""
        <div class="metric">
            <h3>{algo_name} Algorithm</h3>
            <ul>
                <li>Total Requests: {metrics.get('total_requests', 0):,}</li>
                <li>Average Reward: {metrics.get('avg_reward', 'N/A') if metrics.get('avg_reward', None) is not None else 'N/A'}</li>
                <li>Average CPU Usage: {metrics.get('avg_cpu_usage', 'N/A') if metrics.get('avg_cpu_usage', None) is not None else 'N/A'}</li>
                <li>Average Memory Usage: {metrics.get('avg_mem_usage', 'N/A') if metrics.get('avg_mem_usage', None) is not None else 'N/A'}</li>
                <li>Average Bandwidth Usage: {metrics.get('avg_bw_usage', 'N/A') if metrics.get('avg_bw_usage', None) is not None else 'N/A'}</li>
                <li>Load Balance Index: {metrics.get('load_balance_index', 'N/A') if metrics.get('load_balance_index', None) is not None else 'N/A'}</li>
            </ul>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    return html_content

def create_download_link(data, filename, mime_type):
    """Create a download link for Streamlit"""
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">Download {filename}</a>'
    return href




