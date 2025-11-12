# Large language models for autism: Evaluating theory of mind tasks in a gamified environment

Provide file for evaluating with LLM:
https://github.com/ChristianPoglitsch/AIAgents/tree/dev/LLM_Character/games/SocialTraining


Provides comprehensive scoring guidelines used by both human experts and the Large Language Model for each individual question across all tasks:
https://github.com/ChristianPoglitsch/AIAgents/tree/dev/LLM_Character/games/SocialTraining/Supplementary/detailed_scoring_information.pdf


Contains general instructions and evaluation criteria specific to the Faux Pas tasks:
https://github.com/ChristianPoglitsch/AIAgents/tree/dev/LLM_Character/games/SocialTraining/Supplementary/Fauxpas.pdf


A structured sample dataset illustrating the input format used for LLM evaluation, including participant responses and associated metadata:
https://github.com/ChristianPoglitsch/AIAgents/tree/dev/LLM_Character/games/SocialTraining/Supplementary/sample_dataset.xlsx


Includes full descriptions of all Theory of Mind tasks used in the study, along with their corresponding questions:
https://github.com/ChristianPoglitsch/AIAgents/tree/dev/LLM_Character/games/SocialTraining/Supplementary/ToM_Tasks.pdf


The game Social training is available at: https://tom.gamelabgraz.at/


# Paper

The associated research paper is published in *Scientific Reports* (Nature):  
https://www.nature.com/articles/s41598-025-18608-4



# Training compact language models for artificial emotional intelligence: from bluffing to trust in a social deduction game

Code for training and results can be found under LLM_Character/games/BotC

1) Play the game starting using botc.py

Results are stores as .pkl file and in a subfolder with test and train data

2) Train the model starting training.py

3) Merge original model with trained data using merge_model_adapter.py


# Paper (Draft)

The associated paper draft is published on Research Square:  
https://www.researchsquare.com/article/rs-7348060/v1


# Installation Guide

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


