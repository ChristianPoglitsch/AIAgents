#

Large language model evaluating theory of mind tasks in a gamified environment Supplementary Information Files


https://github.com/ChristianPoglitsch/AIAgents/tree/dev/LLM_Character/games/SocialTraining
Provide file for evaluating with LLM
Provide file to compute statistic


https://github.com/ChristianPoglitsch/AIAgents/tree/dev/LLM_Character/games/SocialTraining/Supplementary/detailed_scoring_information.pdf
Provides comprehensive scoring guidelines used by both human experts and the Large Language Model for each individual question across all tasks.

https://github.com/ChristianPoglitsch/AIAgents/tree/dev/LLM_Character/games/SocialTraining/Supplementary/Fauxpas.pdf
Contains general instructions and evaluation criteria specific to the Faux Pas tasks.

https://github.com/ChristianPoglitsch/AIAgents/tree/dev/LLM_Character/games/SocialTraining/Supplementary/sample_dataset.xlsx
A structured sample dataset illustrating the input format used for LLM evaluation, including participant responses and associated metadata.

https://github.com/ChristianPoglitsch/AIAgents/tree/dev/LLM_Character/games/SocialTraining/Supplementary/ToM_Tasks.pdf
Includes full descriptions of all Theory of Mind tasks used in the study, along with their corresponding questions.




# EmpathicAgents

## installation guide

**1. Python requirements**

```bash
pip install -r requirements.txt    
```

**2. Further requirements**

```bash
pip install --editable .
```

**3. OpenAI key**
For OpenAI key use .env file and add OPENAI_API_KEY= 


**4. Hugging face key**
Add key according to https://huggingface.co/docs/huggingface_hub/quick-start.
Define a fine-grained token and set env HF_TOKEN

**5. (Optional) Simple WebSocket**
start webSocketServer.py

## Authors

- Christian Poglitsch

   