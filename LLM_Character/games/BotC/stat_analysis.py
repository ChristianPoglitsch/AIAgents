from statsmodels.stats.proportion import proportions_ztest

count = [49, 35]
nobs = [100, 100]

stat, pval = proportions_ztest(count, nobs)
print(f"z-Wert: {stat:.3f}, p-Wert: {pval:.3f}")




# error analysis
from scipy.stats import mannwhitneyu

# Simulated error distributions
group1_errors = [0] * 39 + [1] * 61  # 100 values: 61 errors, 39 non-errors
group2_errors = [0] * (100 - 160) + [1] * 160  # This would imply more than 100 games!

# Correction: Use actual game count. Assume 100 games each group
# If you had 61 errors in 100 games:
group1 = [1]*61 + [0]*39  # 1 = error, 0 = no error
# If you had 160 errors in 100 games, this implies multiple errors per game
# Simulate this by assuming e.g. per-game error counts
import numpy as np
np.random.seed(0)
group2 = np.random.poisson(lam=1.6, size=100)  # Avg 1.6 errors per game

# Run the Mann-Whitney U test
stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
print(f"Mann-Whitney U statistic = {stat}")
print(f"p-value = {p}")
