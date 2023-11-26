import os
import json
import pandas as pd
from pypdf import PdfReader
from tqdm import tqdm
from enum import Enum
import requests
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

headers = {"Authorization": "Bearer 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"}
API_URL_climate_related = "https://api-inference.huggingface.co/models/climatebert/distilroberta-base-climate-detector"
API_URL_ron = "https://api-inference.huggingface.co/models/climatebert/distilroberta-base-climate-sentiment"
API_URL_tcfd = "https://api-inference.huggingface.co/models/climatebert/distilroberta-base-climate-tcfd"

class TCFDDomain(Enum):
    Governance = "Governance"
    Strategy = "Strategy"
    RiskManagement = "RiskManagement" 
    MetricsTargets = "MetricsTargets"
    NotDefined = "NotDefined"

# risk, opportunity, neutral
class Ron(Enum):
    Risk = "Risk"
    Opportunity = "Opportunity"
    Neutral = "Neutral"

"""
GOAL:
    labeled df of form: 
    # pdf_name | text | page_nr | climate_related | domain | r.o.n. | ?transition 
    #                           | no              | nan 
"""

def pipeline(df: pd.DataFrame) -> pd.DataFrame:
    # extraction: pdf -> json of pages 

    for index, row in df.iterrows():
        
        print("Row: ", row)

        # climate related label, update row in df with output of add_climate_related_label 
        row = add_climate_related_label(row)
        df.loc[index] = row
        
        if row["climate_related"] == 'True': 
            print("Climate related")
            row = add_domain_label(row)
            df.loc[index] = row

            if row["domain"] == TCFDDomain.Strategy.value:
                print("Strategy")
                row = add_ron_label(row)
                df.loc[index] = row
    
    return df

def add_climate_related_label(row: pd.Series) -> pd.Series:
    response = request_label(row["text"], API_URL_climate_related)

    if find_highest_score(response[0]) == "no":
        row["climate_related"] = 'False'
    elif find_highest_score(response[0]) == "yes":
        row["climate_related"] = 'True'
    else: 
        row["climate_related"] = 'api_failed'
    return row
    
def add_domain_label(row: pd.Series) -> pd.Series:
    
    response = request_label(row["text"], API_URL_tcfd)
    
    if find_highest_score(response[0]) == "governance":
        row["domain"] = TCFDDomain.Governance.value
    elif find_highest_score(response[0]) == "strategy":
        row["domain"] = TCFDDomain.Strategy.value
    elif find_highest_score(response[0]) == "risk":
        row["domain"] = TCFDDomain.RiskManagement.value
    elif find_highest_score(response[0]) == "metrics":
        row["domain"] = TCFDDomain.MetricsTargets.value
    else:
        row["domain"] = "api_failed"
    return row

def add_ron_label(row: pd.Series) -> pd.Series:
    response = request_label(row["text"], API_URL_ron)

    if find_highest_score(response[0]) == "risk":
        row["ron"] = Ron.Risk.value
    elif find_highest_score(response[0]) == "opportunity":
        row["ron"] = Ron.Opportunity.value
    elif find_highest_score(response[0]) == "neutral":
        row["ron"] = Ron.Neutral.value
    else: 
        row["ron"] = "api_failed"
    return row

def request_label(text: str, api_url: str) -> str:
    payload = {"inputs": text}
    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        print(response.status_code)
        return [[{'label': 'api_failed', 'score': 1}]]

def find_highest_score(json: list) -> str:
    highest_score = 0
    highest_score_label = ""
    for i in json: 
        if i["score"] > highest_score:
            highest_score = i["score"]
            highest_score_label = i["label"]
    return highest_score_label

def extract(pdf_path: str) -> pd.DataFrame:

    df = pd.DataFrame(columns=["pdf_name","page_nr", "climate_related","domain" ,"ron", "transition", "text"])
    df["climate_related"] = df["climate_related"].astype(str)
    df["domain"] = df["domain"].astype(str)
    df["ron"] = df["ron"].astype(str)

    reader = PdfReader(pdf_path)
    
    for i, page in enumerate(tqdm(reader.pages, desc="Extracting text")):
    
        df = df._append({"pdf_name": pdf_path.split(".")[0].split("/")[-1],"text":page.extract_text(),"page_nr":  i + 1}, ignore_index = True)

    return df 

def store_df(df: pd.DataFrame): 
    df.to_csv(f"outputs/{df['pdf_name'][0]}.csv")

def filter(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[(df['climate_related'] == True) & (df['domain'] == TCFDDomain.Strategy.value) & (df['ron'] == Ron.Risk.value)]

if "__main__" == __name__: 

    


    '''folder_path = 'pdfs'
    files = os.listdir(folder_path)
    document_names = [file for file in files if file.endswith('.pdf')]

    folder_path = 'outputs'
    files = os.listdir(folder_path)
    csv_names = [file.split(".")[-2] for file in files if file.endswith('.csv')]

    print(csv_names)

    document_names = ["Commerzbank_2022_EN-2.pdf"]

    for document_name in document_names:

        if document_name.split(".")[-2] in csv_names: 
            print(f"Load {document_name.split('.')[-2]}")
            df = pd.read_csv(f"outputs/{document_name.split('.')[-2]}.csv", index_col=0)
        else:
            print(f"Extract {document_name.split('.')[-2]}")
            df = extract(f"pdfs/{document_name}")
        df = pipeline(df)
        store_df(df)
    
    print(filter(df))'''
    print("start")
    
    df = pd.DataFrame({"text": ["Climate Stategy"], "climate_related": [pd.NA], "domain": [pd.NA], "ron": [pd.NA]})

    df = pipeline(df)

    print(df)