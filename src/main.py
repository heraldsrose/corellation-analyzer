import argparse
from data_loader import load_data
from analysis import analyze
from visualization import generate_plots
from report import generate_report
from bootstrap_ci import (
    bootstrap_pearson,
    bootstrap_spearman,
    compare_sample_sizes,
    plot_bootstrap_distribution,
    plot_ci_vs_sample_size,
    plot_comparison_methods,
    format_results_table as bootstrap_format_table,
)
from multiple_comparison import (
    compare_groups_to_control,
    apply_corrections,
    summarize_corrections,
    plot_pvalue_comparison,
    plot_significance_heatmap,
    format_results_table as mc_format_table,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',    required=True,  help='Path to CSV input data')
    parser.add_argument('--output',   required=True,  help='Path to save report PDF')

    # Bootstrap
    parser.add_argument('--bootstrap',   action='store_true',
                        help='Run bootstrap CI analysis for correlations')
    parser.add_argument('--bootstrap-n', type=int, default=1000,
                        help='Number of bootstrap iterations (default: 1000)')
    parser.add_argument('--x-col', default=None,
                        help='Column name for X variable in bootstrap analysis')
    parser.add_argument('--y-col', default=None,
                        help='Column name for Y variable in bootstrap analysis')

    # Multiple comparison
    parser.add_argument('--multiple-comparison', action='store_true',
                        help='Run multiple comparison correction (Bonferroni / BH)')
    parser.add_argument('--response-col',     default='behavior_metric',
                        help='Column with biological response (default: behavior_metric)')
    parser.add_argument('--conc-col',         default='concentration',
                        help='Column with concentrations (default: concentration)')
    parser.add_argument('--group-col',        default='group',
                        help='Column with group labels (default: group)')
    parser.add_argument('--control-label',    default='control',
                        help='Label for control group (default: control)')
    parser.add_argument('--test',             default='auto',
                        choices=['auto', 't-test', 'mannwhitney'],
                        help='Statistical test to use (default: auto)')

    args = parser.parse_args()

    # ── Стандартный анализ (без изменений) ───────────────────────────────────
    df = load_data(args.input)
    results = analyze(df)
    figs = generate_plots(df, results)
    generate_report(df, results, figs, args.output)
    print(f'[main] Стандартный отчёт сохранён: {args.output}')

    base = args.output.rsplit('.', 1)[0]

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    if args.bootstrap:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) < 2:
            print('[bootstrap] Ошибка: нужно минимум два числовых столбца.')
        else:
            x_col = args.x_col or numeric_cols[0]
            y_col = args.y_col or numeric_cols[1]

            if x_col not in df.columns or y_col not in df.columns:
                print(f'[bootstrap] Ошибка: столбцы "{x_col}" и/или "{y_col}" не найдены.')
            else:
                x = df[x_col].dropna().values
                y = df[y_col].dropna().values
                if len(x) != len(y):
                    x, y = x[:min(len(x), len(y))], y[:min(len(x), len(y))]

                print(f'\n[bootstrap] Анализ: "{x_col}" vs "{y_col}", '
                      f'n={len(x)}, итераций={args.bootstrap_n}')

                res_p = bootstrap_pearson(x, y, n_iterations=args.bootstrap_n)
                res_s = bootstrap_spearman(x, y, n_iterations=args.bootstrap_n)

                print('\n' + bootstrap_format_table(res_p, res_s).to_string(index=False))

                plot_bootstrap_distribution(
                    res_p, f'{base}_bootstrap_dist.png',
                    title=f'Bootstrap-распределение: {x_col} vs {y_col}'
                )
                plot_comparison_methods(res_p, f'{base}_ci_comparison.png')

                if len(x) >= 10:
                    df_comp = compare_sample_sizes(x, y, n_simulations=200)
                    plot_ci_vs_sample_size(df_comp, f'{base}_ci_vs_n.png')

                print(f'[bootstrap] Графики: {base}_bootstrap_*.png')

    # ── Поправка на множественные сравнения ───────────────────────────────────
    if args.multiple_comparison:
        print(f'\n[multiple_comparison] Метод: {args.test}, '
              f'отклик: {args.response_col}, концентрации: {args.conc_col}')

        try:
            df_comp = compare_groups_to_control(
                df,
                concentration_col=args.conc_col,
                response_col=args.response_col,
                group_col=args.group_col,
                control_label=args.control_label,
                test=args.test
            )
            df_corr = apply_corrections(df_comp)

            print('\n' + summarize_corrections(df_corr))
            print('\n' + mc_format_table(df_corr).to_string(index=False))

            plot_pvalue_comparison(df_corr, f'{base}_pvalue_comparison.png')
            plot_significance_heatmap(df_corr, f'{base}_significance_heatmap.png')

            print(f'\n[multiple_comparison] Графики: {base}_pvalue_*.png, '
                  f'{base}_significance_*.png')

        except ValueError as e:
            print(f'[multiple_comparison] Ошибка: {e}')


if __name__ == '__main__':
    main()

