import pandas as pd
import glob
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import transformers
import sys
import os
import time
import google.generativeai as genai
from openai import OpenAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold

##############################################################################
genai.configure(api_key = "AIzaSyAFA_5T0JDIQZt_Tf59fto4AYNaJwJFhSk")#"Enter_GeminiAPI_Key")
openai_client = OpenAI(api_key= "Enter_OpenAI_Key")

##############################################################################
'''
This block defines the major high level args for baselines
'''
model_name = sys.argv[1] #GeminiPro or GPT4omini
prompt_type = sys.argv[2] #'zeroshot' or 'oneshot'
sample_type = sys.argv[3] #'all' or 'test'
instruction_type = 'doctors_prompts'# sys.argv[4]

print(model_name, prompt_type, instruction_type)

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
# df_test = df_test.sample(n=1)
# df_test.fillna("Nil", inplace=True)

df_train = pd.read_excel('train.xlsx')
# print(df_train.columns)
df_train = df_train[["Summary Number"]+features]
# df_train.fillna("Nil", inplace=True)

if(sample_type=='all'):
    df = pd.concat([df_test, df_train])
    # df = df.head(3) # for testing if required
    df_test = df.copy()
    df_train = df.copy()

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
        ex_prompt.append( "###Dialog###: " + dialog + "###Summary  Response###: " + example.loc[f] )

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
if(model_name=='GeminiPro'):
    model_gemini = genai.GenerativeModel('gemini-pro') # Note: gemini-pro is an alias for gemini-1.0-pro.
    # defines safety settings for gemini model (as the api was initially throwing an error)
    ss = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    def ModelOutput(prompt):
        try:
            response = model_gemini.generate_content(prompt, safety_settings=ss)
        except:
            # print("Sleep for 60 sec")
            time.sleep(60)
            response = model_gemini.generate_content(prompt, safety_settings=ss)

        # sometimes because of safety rating context can be blocked
        try:
            return response.text
        except:
            print("There was an error in response")
            return ""
    
elif(model_name=='GPT4omini'):
    def ModelOutput(prompt):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ]
                )
        except:
            # print("Sleep for 60 sec")
            time.sleep(60)
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ]
                )
            
        return response.choices[0].message.content
        
###############################################################################
try:
    if(sample_type=='all'):
        df_out = pd.read_excel("Outputs/HumanEval/" + model_name + "_" + instruction_type + "_" + prompt_type + '.xlsx')
    else:
        df_out = pd.read_excel("Outputs/" + model_name + "_" + instruction_type + "_" + prompt_type + '.xlsx')
except:
    df_out = pd.DataFrame(columns=["Summary Number"]+ features + ["By Model "+ i for i in features])

# Summ = [ [] for i in range(len(features))]
for i in tqdm(range(len(df_out), len(df_test))):
    """
    Iterating over the test files and saving the response in Summ list Summ[0] represent the resp for first aspect and so on
    """
    # dialog = CSVtoStringDialog(i)

    summ_prompts = PromptGen(df_test.iloc[i]["Summary Number"], prompt_type)

    Summ = []
    try:
        for j in range(len(summ_prompts)):
            Summ.append(ModelOutput(summ_prompts[j]))

        df_out.loc[len(df_out)] = list(df_test.iloc[i]) + Summ
    except:
        break

# for i,j in enumerate(features):
#     """
#     appending the model resp to dataframe
#     """
#     df_test["By Model " + j] = Summ[i]

# Saving the output results
if(sample_type=='all'):
    df_out.to_excel("Outputs/HumanEval/" + model_name + "_" + instruction_type + "_" + prompt_type + '.xlsx', index=False)
else:
    df_out.to_excel("Outputs/" + model_name + "_" + instruction_type + "_" + prompt_type + '.xlsx', index=False)