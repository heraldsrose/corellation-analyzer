"""
multiple_comparison.py — Поправка на множественные сравнения
Модуль для системы статистического анализа данных биотестирования.

Реализует:
- Попарные сравнения каждой концентрации с контролем (t-тест / Манн-Уитни)
- Поправку Бонферрони (консервативная, для небольшого числа сравнений)
- Поправку Бенджамини-Хохберга (FDR, рекомендуется при 5+ сравнениях)
- Визуализацию скорректированных p-значений
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ─── Константы ────────────────────────────────────────────────────────────────

ALPHA = 0.05


# ─── Основные функции ─────────────────────────────────────────────────────────

def compare_groups_to_control(
    df: pd.DataFrame,
    concentration_col: str = 'concentration',
    response_col: str = 'behavior_metric',
    group_col: str = 'group',
    control_label: str = 'control',
    test: str = 'auto'
) -> pd.DataFrame:
    """
    Попарные сравнения каждой концентрации с контрольной группой.

    Parameters
    ----------
    df                : DataFrame с данными биотестирования
    concentration_col : столбец с концентрациями
    response_col      : столбец с биологическим откликом
    group_col         : столбец с метками группы (если есть)
    control_label     : метка контрольной группы
    test              : 't-test', 'mannwhitney' или 'auto'
                        при 'auto' — проверяет нормальность (Шапиро-Уилк)
                        и выбирает метод автоматически

    Returns
    -------
    DataFrame: concentration | n | mean | std | statistic | p_raw | test_used
    """
    # Контрольная группа
    if group_col and group_col in df.columns:
        control_data = df[df[group_col] == control_label][response_col].dropna().values
    else:
        control_data = df[df[concentration_col] == 0.0][response_col].dropna().values

    if len(control_data) == 0:
        raise ValueError(
            f'Контрольная группа не найдена. '
            f'Проверь значения в столбце "{group_col}" или нулевую концентрацию.'
        )

    # Все концентрации кроме 0
    concentrations = sorted(
        df[df[concentration_col] > 0][concentration_col].unique()
    )

    rows = []
    for conc in concentrations:
        exp_data = df[df[concentration_col] == conc][response_col].dropna().values

        if len(exp_data) < 2:
            continue

        # Выбор критерия
        if test == 'auto':
            # Шапиро-Уилк работает при n >= 3
            if len(control_data) >= 3 and len(exp_data) >= 3:
                _, p_norm_ctrl = stats.shapiro(control_data)
                _, p_norm_exp = stats.shapiro(exp_data)
                use_parametric = (p_norm_ctrl > 0.05) and (p_norm_exp > 0.05)
            else:
                use_parametric = True
            test_used = 't-test' if use_parametric else 'Mann-Whitney'
        else:
            test_used = test
            use_parametric = (test == 't-test')

        if use_parametric:
            stat, p_raw = stats.ttest_ind(control_data, exp_data)
        else:
            stat, p_raw = stats.mannwhitneyu(
                control_data, exp_data, alternative='two-sided'
            )

        rows.append({
            'concentration': conc,
            'n_control': len(control_data),
            'n_experiment': len(exp_data),
            'mean_control': float(np.mean(control_data)),
            'mean_experiment': float(np.mean(exp_data)),
            'std_experiment': float(np.std(exp_data, ddof=1)),
            'statistic': float(stat),
            'p_raw': float(p_raw),
            'test_used': test_used
        })

    return pd.DataFrame(rows)


def apply_corrections(
    df_comparisons: pd.DataFrame,
    alpha: float = ALPHA
) -> pd.DataFrame:
    """
    Применяет поправки Бонферрони и Бенджамини-Хохберга к сырым p-значениям.

    Parameters
    ----------
    df_comparisons : результат compare_groups_to_control()
    alpha          : уровень значимости

    Returns
    -------
    DataFrame с добавленными столбцами:
        p_bonferroni | significant_bonferroni |
        p_bh         | significant_bh
    """
    p_values = df_comparisons['p_raw'].values
    result = df_comparisons.copy()

    # Поправка Бонферрони
    _, p_bonf, _, _ = multipletests(p_values, alpha=alpha, method='bonferroni')
    result['p_bonferroni'] = p_bonf
    result['significant_bonferroni'] = p_bonf < alpha

    # Поправка Бенджамини-Хохберга (FDR)
    _, p_bh, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    result['p_bh'] = p_bh
    result['significant_bh'] = p_bh < alpha

    # Исходная значимость без поправки
    result['significant_raw'] = p_values < alpha

    return result


def summarize_corrections(df_corrected: pd.DataFrame) -> str:
    """
    Текстовая сводка: сколько сравнений значимы при каждом методе.
    Удобно для раздела «Результаты» в статье.
    """
    n = len(df_corrected)
    n_raw  = df_corrected['significant_raw'].sum()
    n_bonf = df_corrected['significant_bonferroni'].sum()
    n_bh   = df_corrected['significant_bh'].sum()

    lines = [
        f"Всего попарных сравнений с контролем: {n}",
        f"Значимых без поправки (α={ALPHA}):       {n_raw} из {n}",
        f"Значимых после поправки Бонферрони:    {n_bonf} из {n}",
        f"Значимых после поправки Бенджамини-Хохберга (FDR): {n_bh} из {n}",
    ]

    if n_raw > n_bonf:
        lines.append(
            f"\nПоправка Бонферрони исключила {n_raw - n_bonf} потенциально "
            f"ложноположительных результат(а/ов)."
        )
    if n_bh >= n_bonf:
        lines.append(
            f"Поправка BH (менее консервативная) подтверждает {n_bh} значимых "
            f"сравнений — рекомендуется при числе тестов ≥ 5."
        )

    return '\n'.join(lines)


# ─── Визуализация ─────────────────────────────────────────────────────────────

def plot_pvalue_comparison(
    df_corrected: pd.DataFrame,
    output_path: str,
    alpha: float = ALPHA
) -> None:
    """
    График сравнения сырых и скорректированных p-значений по концентрациям.
    Ключевая иллюстрация для статьи.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    conc_labels = [str(c) for c in df_corrected['concentration']]
    x = np.arange(len(conc_labels))
    width = 0.28

    bars_raw  = ax.bar(x - width, df_corrected['p_raw'],
                       width, label='Без поправки', color='#4C72B0', alpha=0.85)
    bars_bonf = ax.bar(x,         df_corrected['p_bonferroni'],
                       width, label='Бонферрони',   color='#DD8452', alpha=0.85)
    bars_bh   = ax.bar(x + width, df_corrected['p_bh'],
                       width, label='Бенджамини-Хохберг (FDR)',
                       color='#55A868', alpha=0.85)

    # Линия уровня значимости
    ax.axhline(alpha, color='#CC3333', linewidth=1.5, linestyle='--',
               label=f'α = {alpha}')

    # Отметки значимости над столбцами
    for bar, sig in zip(bars_raw, df_corrected['significant_raw']):
        if sig:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005, '*',
                    ha='center', va='bottom', color='#4C72B0', fontsize=13)

    for bar, sig in zip(bars_bh, df_corrected['significant_bh']):
        if sig:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005, '*',
                    ha='center', va='bottom', color='#55A868', fontsize=13)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{c} мг/л' for c in df_corrected['concentration']],
                       fontsize=10)
    ax.set_xlabel('Концентрация токсиканта', fontsize=12)
    ax.set_ylabel('p-значение', fontsize=12)
    ax.set_title('Сравнение p-значений до и после поправки\nна множественные сравнения',
                 fontsize=13, pad=12)
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, min(df_corrected[['p_raw','p_bonferroni','p_bh']].max().max() * 1.25, 1.05))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_significance_heatmap(
    df_corrected: pd.DataFrame,
    output_path: str
) -> None:
    """
    Тепловая карта значимости: концентрации × методы поправки.
    Наглядно показывает, какие сравнения «выживают» после коррекции.
    """
    matrix = df_corrected[
        ['significant_raw', 'significant_bonferroni', 'significant_bh']
    ].astype(int).T

    matrix.index = ['Без поправки', 'Бонферрони', 'Бенджамини-Хохберг']
    matrix.columns = [f'{c} мг/л' for c in df_corrected['concentration']]

    fig, ax = plt.subplots(figsize=(max(6, len(df_corrected) * 1.2), 3.5))

    import seaborn as sns
    sns.heatmap(
        matrix, annot=True, fmt='d', cmap=['#F0F0F0', '#2CA02C'],
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': '1 = значимо, 0 = незначимо'},
        ax=ax, vmin=0, vmax=1
    )
    ax.set_title('Значимость различий при разных методах поправки',
                 fontsize=12, pad=10)
    ax.set_xlabel('Концентрация токсиканта', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─── Форматирование таблицы для статьи ───────────────────────────────────────

def format_results_table(df_corrected: pd.DataFrame) -> pd.DataFrame:
    """
    Сводная таблица в формате, готовом для вставки в статью / диплом.
    """
    table = pd.DataFrame({
        'Концентрация, мг/л': df_corrected['concentration'],
        'n (опыт)':           df_corrected['n_experiment'],
        'Среднее (опыт)':     df_corrected['mean_experiment'].round(2),
        'Критерий':           df_corrected['test_used'],
        'p (без поправки)':   df_corrected['p_raw'].round(4),
        'p (Бонферрони)':     df_corrected['p_bonferroni'].round(4),
        'p (BH, FDR)':        df_corrected['p_bh'].round(4),
        'Значимо (BH)':       df_corrected['significant_bh'].map({True: 'Да', False: 'Нет'})
    })
    return table


# ─── Демо-запуск ─────────────────────────────────────────────────────────────

def _demo():
    import numpy as np
    rng = np.random.default_rng(42)

    # Модельные данные: контроль + 6 концентраций
    concentrations = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    rows = []
    for conc in concentrations:
        n = 10
        baseline = 95 - 18.5 * np.log1p(conc)
        values = rng.normal(baseline, 4, n)
        for v in values:
            rows.append({
                'concentration': conc,
                'group': 'control' if conc == 0 else 'experiment',
                'behavior_metric': round(float(v), 2)
            })

    df = pd.DataFrame(rows)

    print('=' * 60)
    print('Поправка на множественные сравнения (демо)')
    print('=' * 60)

    df_comp = compare_groups_to_control(df)
    df_corr = apply_corrections(df_comp)

    print('\n' + summarize_corrections(df_corr))
    print('\nПолная таблица:')
    print(format_results_table(df_corr).to_string(index=False))

    plot_pvalue_comparison(df_corr, '/home/claude/demo_pvalue_comparison.png')
    plot_significance_heatmap(df_corr, '/home/claude/demo_significance_heatmap.png')
    print('\nГрафики сохранены.')


if __name__ == '__main__':
    _demo()
