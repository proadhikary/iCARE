# Mental Health Summarization

This repo contains the code and data for baseline models.

## Setup
Install the required packages
```sh
pip install -r requirements.txt
```

To generate the train and test data
```sh
python3 Train_Test_Builder.py
```

Run the baselines [OS = {Llama3, Llama3_1, Mentallama, Mistral*, Phi3, Orca2, Vicuna, Zephyr, Gemma} & CS = {GeminiPro*, GPT4omini}] and [prompt type = {zeroshot, oneshot*}]
```sh
python3 Baselines.py <OS model name> <prompt type>
python3 Baselines_ClosedModels.py <CS model name> <prompt type>
```

To evaluate the model outputs
```sh
python3 Scores.py
```
Scores with different eval metrics (BLEU, METEOR, ROUGE, BERTScore, InfoLM, etc.) are stored in `scores.xlsx`.

## Data Information
1. `instruction.json` contains the instructions for the prompting (these prompts are approved by the AIIMS Delhi's doctors). Kindly note that the order of prompts is the same as the aspects mentioned in the train/test data (17 aspects in total).

2. Major fields in train/test are Summary Number and Summary Aspects (17 in number).

3. `Data` folder contains training and testing dialogs and gold-annotated data by experts. 

## Cite
