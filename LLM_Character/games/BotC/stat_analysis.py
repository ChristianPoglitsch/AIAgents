from statsmodels.stats.proportion import proportions_ztest

count = [26, 36]
nobs = [74, 64]

stat, pval = proportions_ztest(count, nobs)
print(f"z-Wert: {stat:.3f}, p-Wert: {pval:.3f}")
