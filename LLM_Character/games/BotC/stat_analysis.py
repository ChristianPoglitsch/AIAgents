from statsmodels.stats.proportion import proportions_ztest

count = [37, 44]
nobs = [63, 56]

stat, pval = proportions_ztest(count, nobs)
print(f"z-Wert: {stat:.3f}, p-Wert: {pval:.3f}")
