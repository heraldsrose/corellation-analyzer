from jinja2 import Template
import pdfkit
import os

HTML_TEMPLATE = '''
<h1>Analysis Report</h1>
<p><b>T-test (Behavior):</b> {{ t_behavior }}</p>
<p><b>T-test (Physiological):</b> {{ t_phys }}</p>
<p><b>Pearson Correlation:</b> {{ pearson }}</p>
<p><b>Spearman Correlation:</b> {{ spearman }}</p>

<h2>Heatmap</h2>
<img src="{{ heatmap }}" width="600"/>

<h2>Scatter Plot</h2>
<img src="{{ scatter }}" width="600"/>
'''

def generate_report(df, results, figs, output_path):
    html = Template(HTML_TEMPLATE).render(
        t_behavior=results['t_test_behavior'],
        t_phys=results['t_test_phys'],
        pearson=results['pearson_corr'],
        spearman=results['spearman_corr'],
        heatmap=figs['heatmap'],
        scatter=figs['scatter']
    )

    html_path = output_path.replace('.pdf', '.html')
    with open(html_path, 'w') as f:
        f.write(html)

    pdfkit.from_file(html_path, output_path)
    os.remove(html_path)
