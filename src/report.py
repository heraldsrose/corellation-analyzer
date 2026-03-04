from jinja2 import Template
import pdfkit
import os
import base64

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body>
<h1>Analysis Report</h1>
<p><b>T-test (Behavior):</b> {{ t_behavior }}</p>
<p><b>T-test (Physiological):</b> {{ t_phys }}</p>
<p><b>Pearson Correlation:</b> {{ pearson }}</p>
<p><b>Spearman Correlation:</b> {{ spearman }}</p>
<h2>Heatmap</h2>
<img src="data:image/png;base64,{{ heatmap }}" width="600"/>
<h2>Scatter Plot</h2>
<img src="data:image/png;base64,{{ scatter }}" width="600"/>
</body>
</html>
'''

def _img_to_base64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def generate_report(df, results, figs, output_path):
    html = Template(HTML_TEMPLATE).render(
        t_behavior=results['t_test_behavior'],
        t_phys=results['t_test_phys'],
        pearson=results['pearson_corr'],
        spearman=results['spearman_corr'],
        heatmap=_img_to_base64(figs['heatmap']),
        scatter=_img_to_base64(figs['scatter'])
    )
    html_path = output_path.replace('.pdf', '.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    options = {'enable-local-file-access': ''}
    pdfkit.from_file(html_path, output_path, options=options)
    os.remove(html_path)
