import pandas as pd
import os
import time
import logging
from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.messages_dataclass import AIMessage, AIMessages
from LLM_Character.util import LOGGER_NAME, setup_logging

logger = logging.getLogger(LOGGER_NAME)
import torch

from LLM_Character.llm_comms.llm_openai import OpenAIComms
from LLM_Character.llm_comms.llm_local import LocalComms


model = OpenAIComms()
model_id = "gpt-4o"

#model = LocalComms()
#model_id = "mistralai/Mistral-7B-Instruct-v0.3"

model.init(model_id)
wrapped_model = LLM_API(model)    
model.max_tokens = 4096
model.set_temperature(0.2)

# Replace 'your_file.xlsx' with the path to your Excel file
file_path = 'LLM_Character/games/SocialTraining/data/ToM_may_part2.xlsx'

# Anzeigeoption: alle Zeilen anzeigen
pd.set_option('display.max_rows', None)


instruction = """
The study contains tasks from four different Theory of Mind categories: Faux Pas, Irony,
Hinting and Strange Stories. Each category includes both a real and a control task, meaning
that all participants complete eight tasks in total.
Evaluate the responses based on the given story and corresponding questions.
Each answer can be rated as either correct or incorrect. One point is awarded for each correct answers, otherwise zero points. Half points are not possible. The symbol - is equal to not answered.
At the end of your message, return the scores for each question in the following format: Example for four questions: 0 1 1 0. The number of values must exactly match the number of questions. Do not include any additional text or explanations."""

story1 = """
Scoring
Max. points for this task: 9
If participants answer No to the first question, the next four questions (questions two to five) are skipped, but
still need to be scored.
For the story containing a faux pas, questions two to five are awarded zero points if they are
not answered. If answers are provided, they are scored as normal by checking whether they
are correct or incorrect. 
For the control story (without a faux pas), questions two to five are
awarded one point if they remain unanswered and zero points if an answer is provided.
Task:
Faux Pas Task: with Faux Pas
Story:
Sarah is at the supermarket to buy cat food. She got a little black cat called “Daisy” last
week. She meets her work colleague Kevin at the supermarket and starts a conversation with
him. Sarah says: “Hi! I’m here to buy food for my new cat.” Kevin replies: “I hope it’s not
a black cat, they’re ugly and mean bad luck.”
Questions:
1. Did anyone say something they shouldn’t have said or something awkward? - Yes/No
question
2. Who said something they shouldn’t have said or something awkward? - Open text
question
3. What did they say that they should not have said? - Open text question
4. Why shouldn’t they have said it or why was it awkward? - Open text question
5. Why do you think they said it? - Open text question
6. Is it more likely that Kevin knew or did not know that Sarah has a black cat? - Open
text question
7. How do you think Sarah felt? - Open text question
8. Where did Sarah start the conversation with Kevin? - Open text question
9. What did Sarah want to buy? - Open text question"""

story2 = """
Scoring
Max. points for this task: 9
If participants answer No to the first question, the next four questions (questions two to five) are skipped, but
still need to be scored.
For the story containing a faux pas, questions two to five are awarded zero points if they are
not answered. If answers are provided, they are scored as normal by checking whether they
are correct or incorrect. 
For the control story (without a faux pas), questions two to five are
awarded one point if they remain unanswered and zero points if an answer is provided.
Task:
Faux Pas Task: without Faux Pas - Control Story
Story:
Today is Caleb’s first day at his new job. He has brought cupcakes for all his colleagues
and left them in the meeting room. As Caleb walks past an open office door, he hears two
colleagues talking to each other inside. One of them says: “Have you tried the cupcakes in
the meeting room yet?”. The other replies: “No, not yet. I think the new employee brought
them.”
Questions:
1. Did anyone say something they shouldn’t have said or something awkward? - Yes/No
question
2. Who said something they shouldn’t have said or something awkward? - Open text
question
3. What did they say that they should not have said? - Open text question
4. Why shouldn’t they have said it or why was it awkward? - Open text question
5. Why do you think they said it? - Open text question
6. Is it more likely that Caleb’s colleagues knew or did not know that Caleb could hear
them? - Open text question
7. How do you think Caleb felt? - Open text question
8. Where were Caleb’s colleagues during their conversation? - Open text question
9. What did Caleb bring to the office? - Open text question"""

story3 = """
Scoring
Max. points for this task: 5
Task:
Irony Task: with irony
Story:
Hanna is the boss of a large company. As she walks through the office, she sees one of her
employees sitting relaxed at his desk watching a movie. Hanna says to him: “I see you’re
working particularly hard today.”
Questions:
1. Did Hanna think that her employee was working hard today? - Yes/No question
2. What could have been a reason for Hanna to say this? - Open text question
3. How could Hanna have felt in this situation? - Open text question
4. Who was the boss of the company? - Open text question
5. What did the employee do when Hanna walked by? - Open text question"""

story4 = """
Scoring
Max. points for this task: 5
Irony Task: without irony - Control Story
Story:
Paul went shopping because he wants to cook dinner with his wife tonight. He bought so
much that the whole table is full of food. When his wife sees this, she says: “Great, you
bought everything we need.”
Questions:
1. Did Paul’s wife think he had bought everything they needed? - Yes/No question
2. What could have been a reason for Paul’s wife to say this? - Open text question
3. How could Paul’s wife have felt in this situation? - Open text question
4. What did Paul buy? - Open text question
5. What did Paul and his wife want to do that evening? - Open text question
"""

story5 = """
Hinting Task: with hint
Story:
Clara and her husband recently adopted a puppy called “Max”. They are both working on
their laptops when the puppy starts whining. Clara says to her husband: “I think Max needs
to go outside soon but I have so much work to do.”
Questions:
1. What did Clara really mean when she said this? - Open text question
2. What reaction could Clara have hoped for? - Open text question
3. Did Clara want her husband to do something? - Yes/No question
4. What did Clara and her husband do? - Open text question
5. What was the puppy’s name? - Open text question
"""

story6 = """
Hinting Task: without hint - Control Story
Story:
Olivia and Charlie work in the same office. It’s already late in the evening and they still
have a lot of work to do. Charlie says to Olivia: “I have a really bad headache. I’m going to
take a painkiller and I’ll be right back.”
Questions:
1. What did Charlie really mean when he said this? - Open text question
2. What reaction could Charlie have hoped for? - Open text question
3. Did Charlie want Olivia to do something? - Yes/No question
4. What did Olivia and Charlie do? - Open text question
5. What kind of pain did Charlie experience? - Open text question
"""

story7 = """
Scoring
Max. points for this task: 6
Task:
Strange stories Task: White Lie
Story:
Thomas has been working in his current job for a very long time. He knows that his boss
George doesn’t like it when people disagree with him. When Thomas comes into the office,
his boss says to him: “Look at the new chairs I’ve bought for the company. Aren’t they
nice?” Thomas doesn’t like the color and thinks they look uncomfortable. Thomas replies:
“Yes, they look great. I particularly like the color.”
Questions:
1. Did Thomas mean what he said about the chairs? - Yes/No question
2. Why did Thomas say that? - Open text question
3. How could Thomas have felt in this situation? - Open text question
4. How could Thomas’s boss have felt when he heard Thomas’s response? - Open text
question
5. Who did Thomas talk to? - Open text question
6. What did Thomas’s boss buy for the office? - Open text question
"""

story8 = """
Task:
Strange stories Task: no White Lie - Control Story
Story:
Melissa is invited to lunch at her friend Camila’s house. When Melissa arrives, Camila says
to her: “Hi! I’ve cooked lasagna. I hope you like it?”. As a child, Melissa couldn’t stand
lasagna, but now she enjoys it. Melissa replies: “Yes. I love lasagna.”
Questions:
1. Did Melissa mean what she said? - Yes/No question
2. Why did Melissa say that? - Open text question
3. How could Melissa have felt in this situation? - Open text question
4. How could Melissa’s friend have felt when she heard Melissa’s response? - Open text
question
5. Who did Melissa talk to? - Open text question
6. What did Melissa’s friend cook? - Open text question
"""

stories = [story1, story2, story3, story4, story5, story6, story7, story8]

# Show all sheets in file
xls = pd.ExcelFile(file_path)
print("num Sheets:", xls.sheet_names)

sheets = xls.sheet_names
sheets.pop(0) #  drop Questionnaire sheet
print(sheets)

elapsed_time = 0.0
for sheet in sheets:
    user_id = sheet
    # load specific sheet
    df = pd.read_excel(file_path, sheet_name=user_id)
    
    #if index > 0:
    #    continue
    #index = index + 1

    story_index = 0
    for story_index_column in range(0, 16, 2):
        count = df.iloc[:, story_index_column].count()
        print(count)

        value = df.iloc[0:count, story_index_column]
        string_list = [str(v) for v in value]

        split_list = [item.split('?:') for item in string_list]

        split_list_as_string = '\n'.join([f"{q.strip()}: {a.strip()}" for q, a in split_list])
        #split_list_as_string = '\n'.join([f"{a.strip()}" for q, a in split_list])
        print(split_list_as_string)

        query_introduction = 'Instruction:\n' + instruction + '\n\nStory:\n' + stories[story_index] + '\n\nEvaluate this response:\n\n' + split_list_as_string
        story_index = story_index + 1
        start_time = time.time()  # Start timing
        #'-'*9
        messages = AIMessages()
        message = AIMessage(message=query_introduction, role="user", class_type="MessageAI", sender="user")
        messages.add_message(message)
        query_result = wrapped_model.query_text(messages)
        end_time = time.time()
        elapsed_time = elapsed_time + end_time - start_time
        print(query_result)

        last_line = query_result.strip().split('\n')[-1]
        scores_str = [int(x) for x in last_line.split()]
        scores = list(map(int, scores_str))
        print(scores)

        df.iloc[:len(scores), story_index_column+1] = scores

    #df.to_excel('LLM_Character/games/SocialTraining/data/ToM_may_part1_result.xlsx', sheet_name=user_id, index=False, mode='a')
    file_path_result = 'LLM_Character/games/SocialTraining/data/ToM_result_all_1.xlsx' 
    if os.path.exists(file_path_result):
        # Append as new sheet
        with pd.ExcelWriter(file_path_result, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
            df.to_excel(writer, sheet_name=user_id, index=False)
    else:
        # Create a new file with the first sheet
        with pd.ExcelWriter(file_path_result, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=user_id, index=False)



print(f"Execution time: {elapsed_time:.6f} seconds")
