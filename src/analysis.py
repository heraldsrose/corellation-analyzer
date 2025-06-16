from scipy.stats import ttest_ind, pearsonr, spearmanr

def analyze(df):
    control = df[df['group'] == 'control']
    experiment = df[df['group'] == 'experiment']

    t_test_behavior = ttest_ind(control['behavior_metric'], experiment['behavior_metric'])
    t_test_phys = ttest_ind(control['physiological_metric'], experiment['physiological_metric'])

    pearson_corr = pearsonr(df['concentration'], df['behavior_metric'])
    spearman_corr = spearmanr(df['concentration'], df['behavior_metric'])

    return {
        't_test_behavior': t_test_behavior,
        't_test_phys': t_test_phys,
        'pearson_corr': pearson_corr,
        'spearman_corr': spearman_corr,
    }
