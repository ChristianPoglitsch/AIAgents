from statsmodels.stats.proportion import proportions_ztest

count = [31, 22]
nobs = [69, 78]

stat, pval = proportions_ztest(count, nobs)
print(f"z-Wert: {stat:.3f}, p-Wert: {pval:.3f}")
