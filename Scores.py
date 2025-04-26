import numpy as np
import pandas as pd
import glob
from evaluate import load
import nltk
from tqdm import tqdm
from torchmetrics.text.infolm import InfoLM
nltk.download('punkt_tab')

# load eval metrices
bleu_metric = load("bleu")
meteor_metric = load('meteor')
rouge_L_metric = load('rouge')
bleurt_metric = load('bleurt')
bert_metric = load("bertscore")
infolm = InfoLM('google/bert_uncased_L-2_H-128_A-2', information_measure='fisher_rao_distance',idf=False)

##################################################
def get_scores(out_file):
    print("### Scores for ", out_file, " ###")

    df = pd.read_excel(out_file)
    df.fillna('Nil', inplace=True)
    
    aspects = []
    feature_col = df.columns[1:]
    total_features = int(len(feature_col)/2)

    for i in range(total_features):
        """
        creating human-model generated text pair for diff aspects
        """
        # test feature map:
        # print(feature_col[i], feature_col[total_features + i])
        aspects.append( (df[feature_col[i]], df[feature_col[total_features + i]]) )

    for i in range(total_features):
        """
        calculating the scores wrt to diff features (order of features is same as feature_col !!)
        """
        references = aspects[i][0].lower()
        predictions = aspects[i][1].lower()

        B = bleu_metric.compute(predictions=predictions, references=references)['bleu']
        M = meteor_metric.compute(predictions=predictions, references=references)['meteor']
        RL = rouge_L_metric.compute(predictions=predictions, references=references)['rougeL']
        BS = np.mean(bert_metric.compute(predictions=predictions, references=references, lang="en", model_type="roberta-large")['f1'])
        BL = np.mean(bleurt_metric.compute(predictions=predictions, references=references)['scores'])
        ILM = infolm(predictions, references).item()

        tmp_scores =  {'model': out_file, 'feature': feature_col[i],
                'bleu': B,
                'meteor': M,
                'rougel': RL,
                'bertscore': BS,
                'bleuert': BL,
                'infolm': ILM}
        
        # dynamically updating the global df: score_sheet
        score_sheet.loc[len(score_sheet)] = tmp_scores

#########################################################################
output_files = glob.glob('Outputs/*')
output_files.sort()

# this is a global DF -- adding rows to it within get_scores function
score_sheet = pd.DataFrame(columns=['model', 'feature', 'bleu', 'meteor', 'rougel', 'bertscore', 'bleuert', 'infolm'])
for f in tqdm(output_files):
    get_scores(f)

score_sheet.to_excel('scores.xlsx', index=False)

# scores when all samples were taken (top 3 models w/ oneshot learning FOR human eval)
output_files = glob.glob('Outputs/HumanEval/*')
output_files.sort()

# this is a global DF -- adding rows to it within get_scores function
score_sheet = pd.DataFrame(columns=['model', 'feature', 'bleu', 'meteor', 'rougel', 'bertscore', 'bleuert', 'infolm'])
for f in tqdm(output_files):
    get_scores(f)

score_sheet.to_excel('scores_full_data.xlsx', index=False)