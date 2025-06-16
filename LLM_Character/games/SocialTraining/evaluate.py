import random
import pandas as pd
import numpy as np

def extract_scores_with_categories(file_path):
    """
    Extracts LLM evaluation scores and their categories from each sheet in an Excel file.
    Scores are taken from every second column starting at column 1.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        list: List of (score, category) tuples.
    """
    scores_with_categories = []

    # Define category mapping for each column pair (0-15, step=2)
    category_map = {
        0: 'Faux pas', 1: 'Faux pas',
        2: 'Irony', 3: 'Irony',
        4: 'Hinting Task', 5: 'Hinting Task',
        6: 'Strange stories', 7: 'Strange stories'
    }

    xls = pd.ExcelFile(file_path)
    sheets = xls.sheet_names

    for sheet in sheets:
        try:
            int(sheet)  # Ensure sheet name is numeric
        except ValueError:
            continue

        df = pd.read_excel(file_path, sheet_name=sheet)

        for story_index_column in range(0, 16, 2):
            if story_index_column + 1 >= df.shape[1]:
                continue  # skip if column index out of bounds

            count = df.iloc[:, story_index_column].count()
            scores = df.iloc[:count, story_index_column + 1]
            scores = scores.fillna(0)
            score_list = scores.astype(float).tolist()

            category = category_map.get(story_index_column // 2, 'Unknown')
            scores_with_categories.extend([(score, category) for score in score_list])

    return scores_with_categories




# Anzeigeoption: alle Zeilen anzeigen
pd.set_option('display.max_rows', None)

human_list = []
llm_list = []


# human experts
# part 1
file_path = 'LLM_Character/games/SocialTraining/data/experts/Kopie von ToM_may_part1_teilnehmende1.xlsx'
human_list = extract_scores_with_categories(file_path)
print(human_list)

# part 2
file_path = 'LLM_Character/games/SocialTraining/data/experts/ToM_may_part2_ds.xlsx'
part2 = extract_scores_with_categories(file_path)

file_path = 'LLM_Character/games/SocialTraining/data/experts/Kopie von ToM_may_part2_teilnehmende2ks.xlsx'
l = extract_scores_with_categories(file_path)
part2 = [(a_score + b_score, a_cat) for (a_score, a_cat), (b_score, b_cat) in zip(part2, l)]

file_path = 'LLM_Character/games/SocialTraining/data/experts/Kopie von Neu.xlsx'
l = extract_scores_with_categories(file_path)
part2 = [(a_score + b_score, a_cat) for (a_score, a_cat), (b_score, b_cat) in zip(part2, l)]

part2 = [(score / 3, category) for score, category in part2]
human_list = human_list + part2

# LLM
file_path = 'LLM_Character/games/SocialTraining/data/ToM_may_results_all.xlsx'
llm_list = extract_scores_with_categories(file_path)

print(human_list)
print(llm_list)


from scipy.stats import wilcoxon

# Extract scores only
llm_scores = [score for score, category in llm_list]
human_scores = [score for score, category in human_list]

# For paired scores
stat, p_value = wilcoxon(llm_scores, human_scores)

print(f"Wilcoxon test statistic: {stat}, p-value: {p_value}")



# Combine data into a DataFrame for easier filtering
data = []

# Add 'Whole Dataset' category
for score, cat in llm_list:
    data.append({"Score": score, "Evaluator": "LLM", "Category": "Whole Dataset"})
for score, cat in human_list:
    data.append({"Score": score, "Evaluator": "Human", "Category": "Whole Dataset"})

# Add individual categories
for score, cat in llm_list:
    data.append({"Score": score, "Evaluator": "LLM", "Category": cat})
for score, cat in human_list:
    data.append({"Score": score, "Evaluator": "Human", "Category": cat})

df = pd.DataFrame(data)

# Get unique categories
categories = df["Category"].unique()

print("Wilcoxon signed-rank test results by category:\n")

for category in categories:
    llm_scores = df[(df["Category"] == category) & (df["Evaluator"] == "LLM")]["Score"].values
    human_scores = df[(df["Category"] == category) & (df["Evaluator"] == "Human")]["Score"].values

    # Ensure paired lengths
    if len(llm_scores) == len(human_scores) and len(llm_scores) > 0:
        try:
            stat, p = wilcoxon(llm_scores, human_scores)
            print(f"{category}: statistic = {stat:.4f}, p-value = {p:.4f}")
        except ValueError as e:
            print(f"{category}: Test failed ({e})")
    else:
        print(f"{category}: Skipped (unequal or empty number of scores)")


# Plots


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
import pandas as pd

# llm_list = [(score, category), ...]
# human_list = [(score, category), ...]

data = []

for score, cat in llm_list:
    data.append({"Score": score, "Evaluator": "LLM", "Category": "Whole Dataset"})
for score, cat in human_list:
    data.append({"Score": score, "Evaluator": "Human", "Category": "Whole Dataset"})

for score, cat in llm_list:
    data.append({"Score": score, "Evaluator": "LLM", "Category": cat})
for score, cat in human_list:
    data.append({"Score": score, "Evaluator": "Human", "Category": cat})

df = pd.DataFrame(data)

plt.figure(figsize=(14, 7))
sns.set(style="whitegrid")

sns.violinplot(
    x="Category", y="Score", hue="Evaluator", data=df,
    inner="quartile", palette=["#4c72b0", "#55a868"], split=False
)

plt.title("Scores by Category and Evaluator (LLM vs Human)", fontsize=20)
plt.ylabel("Score", fontsize=16)
plt.xlabel("Category", fontsize=16)
plt.xticks(rotation=30, ha="right", fontsize=14)
plt.yticks(fontsize=14)

plt.legend(title="Evaluator", fontsize=14, title_fontsize=16,
           loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space on right for legend
plt.show()


