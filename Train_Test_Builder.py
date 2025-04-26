import pandas as pd
import glob

for type in ['train', 'test']:
    folders = glob.glob('Data/'+type+'/*')

    for f in folders:
        tmp = pd.read_csv(f + "/" + f.split("/")[-1] + "_AIIMS_response.csv")
        try:
            df = pd.concat([df, tmp])
        # first time intialize of df
        except NameError:
            df = tmp.copy()

    df.fillna("Nil", inplace=True)

    df.to_excel(type+'.xlsx', index=False)

    # to clean the df
    del df