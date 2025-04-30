from statsmodels.stats.proportion import proportions_ztest

count = [35, 42]
nobs = [65, 58]

stat, pval = proportions_ztest(count, nobs)
print(f"z-Wert: {stat:.3f}, p-Wert: {pval:.3f}")
