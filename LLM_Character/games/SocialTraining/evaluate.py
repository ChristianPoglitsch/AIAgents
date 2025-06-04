import pandas as pd

def extract_scores(file_path, scores_list_all=None):
    """
    Extracts LLM evaluation scores from each sheet in an Excel file.
    Scores are taken from every second column starting at column 1.

    Args:
        file_path (str): Path to the Excel file.
        scores_list_all (list, optional): Existing list of scores to append to.
                                          If None, a new list is created.

    Returns:
        list: Updated list of score lists per user and story.
    """
    if scores_list_all is None:
        scores_list_all = []

    xls = pd.ExcelFile(file_path)
    sheets = xls.sheet_names
    #print("Sheets found:", sheets)

    for sheet in sheets:
        try:
            int(sheet)  # Try converting the sheet name to an integer
        except ValueError:
            continue  # Skip this sheet if conversion fails        

        df = pd.read_excel(file_path, sheet_name=sheet)

        for story_index_column in range(0, 16, 2):
            count = df.iloc[:, story_index_column + 1].count()
            scores = df.iloc[:count, story_index_column + 1]

            try:
                score_list = scores.astype(int).tolist()
            except ValueError:
                score_list = pd.to_numeric(scores, errors='coerce').dropna().astype(int).tolist()

            scores_list_all.append(score_list)

    return scores_list_all



# Anzeigeoption: alle Zeilen anzeigen
pd.set_option('display.max_rows', None)

humal_list = []
llm_list = []



file_path = 'LLM_Character/games/SocialTraining/data/ToM_may_part1.xlsx'
humal_list = extract_scores(file_path, humal_list)
print(humal_list)



file_path = 'LLM_Character/games/SocialTraining/data/ToM_may_part1_result.xlsx'
llm_list = extract_scores(file_path, llm_list)
print(llm_list)



from statsmodels.stats.contingency_tables import mcnemar
import numpy as np

# Sample binary labels
human =    humal_list
llm   =    llm_list

# Build contingency table
#a = sum((h == 1 and l == 1) for h, l in zip(human, llm))
#b = sum((h == 1 and l == 0) for h, l in zip(human, llm))
#c = sum((h == 0 and l == 1) for h, l in zip(human, llm))
#d = sum((h == 0 and l == 0) for h, l in zip(human, llm))
#
#table = np.array([[a, b], [c, d]])
#
#result = mcnemar(table, exact=False, correction=True)
#
#print(f"McNemar’s test statistic: {result.statistic:.3f}")
#print(f"p-value: {result.pvalue:.4f}")
        