from scipy.stats import mannwhitneyu

# Normierte Scores zweier Personen
person_a = [0.90, 0.85, 0.88, 0.91, 0.87, 0.89]
person_b = [0.72, 0.68, 0.75, 0.70, 0.74, 0.69]

# Mann-Whitney-U-Test (zwei unabhängige Stichproben)
stat, p_value = mannwhitneyu(person_a, person_b, alternative='two-sided')

print(f"Mann-Whitney-U-Statistik: {stat}")
print(f"P-Wert: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Signifikanter Unterschied zwischen den beiden Personen.")
else:
    print("Kein signifikanter Unterschied zwischen den beiden Personen.")
