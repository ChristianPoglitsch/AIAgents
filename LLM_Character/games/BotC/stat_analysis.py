from statsmodels.stats.proportion import proportions_ztest

count = [27, 35]
nobs = [73, 65]

stat, pval = proportions_ztest(count, nobs)
print(f"z-Wert: {stat:.3f}, p-Wert: {pval:.3f}")
