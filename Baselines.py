import pandas as pd
import glob
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import transformers
import sys
import os

##############################################################################
'''
This block defines the major high level args for baselines
'''
model_name = sys.argv[1] #'Llama3' or Mentallama or Mistral or Phi3
prompt_type = sys.argv[2] #'zeroshot' or 'oneshot'
sample_type = sys.argv[3] #'all' or 'test'
instruction_type = 'doctors_prompts'# sys.argv[4] 

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
print(model_name, prompt_type, instruction_type, device)

##############################################################################
'''
Loads instructions for ICL prompt setting
'''
with open('instructions.json') as json_file:
    inst_file = json.load(json_file)

InstructionsForSummary = inst_file[instruction_type]

##############################################################################
'''
Loads train and test data: finally including only Summary No. and the desired features

Note: the feature order is same as the instructions
'''
df_test = pd.read_excel('test.xlsx')
features = list(df_test.columns[7:24])
df_test = df_test[["Summary Number"]+features]
# df_test.fillna("Nil", inplace=True)

df_train = pd.read_excel('train.xlsx')
# print(df_train.columns)
df_train = df_train[["Summary Number"]+features]
# df_train.fillna("Nil", inplace=True)

if(sample_type=='all'):
    df = pd.concat([df_test, df_train])
    df_test = df.copy()
    df_train = df.copy()

##############################################################################
'''
Defining models and tokenizers
For Phi3: trust_remote_code=True is a must arg
For all LLMs, default tempr values are used (i.e. as it is given in config.json)
'''
models = {'Llama3':'/home/models/Meta-Llama-3-8B-Instruct',
          'Llama3_1':'/home/models/Meta-Llama-3.1-8B-Instruct',
          'Mentallama':'/home/models/MentaLLaMA-chat-7B',
          'Mistral':'/home/models/Mistral-7B-Instruct-v0.2',
          'Phi3':'/home/models/Phi-3-small-8k-instruct',
          'Orca2':'/home/models/Orca-2-7b',
          'Vicuna':'/home/models/vicuna-7b-v1.5',
          'Zephyr':'/home/models/zephyr-7b-beta',
          'Gemma':'/home/models/gemma-7b-it'}

if(model_name=='Phi3'):
    tokenizer = transformers.AutoTokenizer.from_pretrained(models[model_name], trust_remote_code=True)#, padding_side='left')
    model = transformers.AutoModelForCausalLM.from_pretrained(models[model_name], torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(models[model_name])#, padding_side='left')
    model = transformers.AutoModelForCausalLM.from_pretrained(models[model_name], torch_dtype=torch.bfloat16).to(device)

tokenizer.pad_token_id = tokenizer.eos_token_id

##############################################################################
def CSVtoStringDialog(file_name):
    """
    input: (int) file/folder number
    return: (str) concated dialog

    This function works for both test and train, it automatically detects the file type (train or test) using the try-catch condition
    """
    file_name = str(file_name)

    try:
        dfS = pd.read_csv('Data/test/' + file_name + "/" + file_name + '.csv')
    except FileNotFoundError:
        dfS = pd.read_csv('Data/train/' + file_name + "/" + file_name + '.csv')

    dialog = ""
    old = ""
    for i in range(len(dfS)):
        rowS = dfS.iloc[i]

        if(rowS.speaker=='SPEAKER_00'):
            if(old=='SPEAKER_00'):
                dialog += rowS.text + " "
            else:
                dialog += "#Therapist#: " + rowS.text + " "
                old = 'SPEAKER_00'
        elif(rowS.speaker=='SPEAKER_01'):
            if(old=='SPEAKER_01'):
                dialog += rowS.text + " "
            else:
                dialog += "#Patient#: " + rowS.text + " "
                old = 'SPEAKER_01'

    return dialog

def ExampleForICL(file_name):
    """
    input: (int) test file/folder name
    return: (list of strs) example dialog + gold(human written summary) text(s)

    First it randomly choose one example from train data with random_state as test_file_name
    Then it extract the dialog for that file number and concat it with gold label (for all features)
    """
    example = df_train.sample(n=1, random_state=file_name).iloc[0]

    # if we get same test and incontext datapoint
    tmp = 1
    while(example.loc['Summary Number']==file_name):
        example = df_train.sample(n=1, random_state=file_name+tmp).iloc[0]
        tmp += 1

    dialog = CSVtoStringDialog(example.loc['Summary Number'])
    ex_prompt = []

    for f in features:
        ex_prompt.append( "###Dialog###: " + dialog + "###Summary Response###: " + example.loc[f] )

    return ex_prompt

def PromptGen(file_name, ptype):
    """
    input: (int) test file name
           (str) prompt type (zeroshot or oneshot)
    return: (list of strs) full prompt for ICL
    """
    dialog = CSVtoStringDialog(file_name)
    prompt = ["###Dialog###: " + dialog + "###Summary Response###: "]*len(features)

    if(ptype=="oneshot"):
        example = ExampleForICL(file_name)
        for i in range(len(features)):
            prompt[i] = example[i] + " " +prompt[i]

    for i in range(len(features)):
        prompt[i] = "You're a helpful mental health assistant and follow the given instructions carefully. "+ InstructionsForSummary[i] + " If no information is available, write 'Nil' only. " + prompt[i]
        
    return prompt

#############################################################################
def ModelOutput(prompt):
    """
    input: (str) instruction prompt
    return: (str) LLM's response
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens= 100, do_sample=False, pad_token_id=tokenizer.pad_token_id)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt):]

#############################################################################
# '''Testing prompt generation'''
# test_prompt = PromptGen(22, 'zeroshot')
# print("TEST PROMPT (ZEROSHOT)")
# for i in test_prompt:
#     print(i)
#     break

# print("\n\n\n\n")
# test_prompt = PromptGen(22, 'oneshot')
# print("TEST PROMPT (ONESHOT)")
# for i in test_prompt:
#     print(i)
#     break

# exit()
###############################################################################
Summ = [ [] for i in range(len(features))]
for i in tqdm(df_test["Summary Number"]):
    """
    Iterating over the test files and saving the response in Summ list Summ[0] represent the resp for first aspect and so on
    """
    dialog = CSVtoStringDialog(i)

    summ_prompts = PromptGen(i, prompt_type)

    for j in range(len(summ_prompts)):
        # Exception handling: Phi3 throws AssertionError when text size go beyond 8192
        try:
            Summ[j].append(ModelOutput(summ_prompts[j]))
        except AssertionError:
            Summ[j].append("")

for i, j in enumerate(features):
    """
    appending the model resp to dataframe
    """
    df_test["By Model " + j] = Summ[i]

# Saving the output results
if(sample_type=='all'):
    df_test.to_excel("Outputs/HumanEval/" + model_name + "_" + instruction_type + "_" + prompt_type + '.xlsx', index=False)
else:
    df_test.to_excel("Outputs/" + model_name + "_" + instruction_type + "_" + prompt_type + '.xlsx', index=False)
