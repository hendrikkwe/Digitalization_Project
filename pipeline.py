import os
import json
import numpy as np
import pandas as pd
from pypdf import PdfReader
from tqdm import tqdm
from enum import Enum
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import sys
from pdfminer.high_level import extract_text

USE_API = False

if not USE_API:
    tokenizer_climate_related = AutoTokenizer.from_pretrained("climatebert/distilroberta-base-climate-detector")
    model_climate_related = AutoModelForSequenceClassification.from_pretrained("climatebert/distilroberta-base-climate-detector")

    tokenizer_ron = AutoTokenizer.from_pretrained("climatebert/distilroberta-base-climate-sentiment")
    model_ron = AutoModelForSequenceClassification.from_pretrained("climatebert/distilroberta-base-climate-sentiment")

    tokenizer_tcfd = AutoTokenizer.from_pretrained("climatebert/distilroberta-base-climate-tcfd")
    model_tcfd = AutoModelForSequenceClassification.from_pretrained("climatebert/distilroberta-base-climate-tcfd")

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

    for index, row in tqdm(df.iterrows()):

        # climate related label, update row in df with output of add_climate_related_label
        row = add_climate_related_label(row)
        df.loc[index] = row

        if row["climate_related_no"] < row["climate_related_yes"]:
            #print("Climate related")
            row = add_domain_label(row)
            df.loc[index] = row

            #if row["domain"] == TCFDDomain.Strategy.value:
            if max([row["domain_MetricsTargets"], row["domain_RiskManagement"], row["domain_Strategy"], row["domain_Governance"]]) == row["domain_Strategy"]:
                #print("Strategy")
                row = add_ron_label(row)
                df.loc[index] = row

    store_df(df)

    return df

def add_climate_related_label(row: pd.Series) -> pd.Series:
    if USE_API == True:
        response = request_label_api(row["text"], API_URL_climate_related)
    else:
        response = request_label_local(row["text"], tokenizer_climate_related, model_climate_related)

    if type(response) == dict:
        row["climate_related_no"] = response["no"]
        row["climate_related_yes"] = response["yes"]
    else:
        row["climate_related_no"] = 0
        row["climate_related_yes"] = 0

    return row

def add_domain_label(row: pd.Series) -> pd.Series:
    if USE_API == True:
        response = request_label_api(row["text"], API_URL_tcfd)
    else:
        response = request_label_local(row["text"], tokenizer_tcfd, model_tcfd)

    if type(response) == dict:
        row["domain_MetricsTargets"] = response["metrics"]
        row["domain_RiskManagement"] = response["risk"]
        row["domain_Strategy"] = response["strategy"]
        row["domain_Governance"] = response["governance"]
    else:
        row["domain_MetricsTargets"] = 0
        row["domain_RiskManagement"] = 0
        row["domain_Strategy"] = 0
        row["domain_Governance"] = 0

    return row

def add_ron_label(row: pd.Series) -> pd.Series:
    if USE_API == True:
        response = request_label_api(row["text"], API_URL_ron)
    else:
        response = request_label_local(row["text"], tokenizer_ron, model_ron)

    if type(response) == dict:
        row["ron_risk"] = response["risk"]
        row["ron_opportunity"] = response["opportunity"]
        row["ron_neutral"] = response["neutral"]
    else:
        row["ron_risk"] = 0
        row["ron_opportunity"] = 0
        row["ron_neutral"] = 0

    return row

def request_label_api(text: str, api_url: str) -> str:
    '''
    return format: [[{'label': 'no', 'score': 0.9543747901916504}, {'label': 'yes', 'score': 0.04562525078654289}]]
    '''

    payload = {"inputs": text}
    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code == 200:
        print({r["label"]:r["score"] for r in response.json()[0]})
        return {r["label"]:r["score"] for r in response.json()[0]}
    else:
        return [[{'label': 'api_failed', 'score': 1}]]

def request_label_local(text: str, tokenizer, model):
    '''
    return format old: [[{'label': 'no', 'score': 0.9543747901916504}, {'label': 'yes', 'score': 0.04562525078654289}]]
    return format new: {'no': 0.9543747901916504, 'yes': 0.04562525078654289}
    '''
    input = tokenizer(text, return_tensors="pt")
    output = model(**input)
    probabilities = F.softmax(output.logits, dim=1)
    # predicted_class = torch.argmax(probabilities, dim=1).item()

    response = {}
    for key, label in model.config.id2label.items():
        response[label] = probabilities[0][key].item()
    return response

def find_highest_score(json: list) -> str:
    highest_score = 0
    highest_score_label = ""
    for i in json:
        if i["score"] > highest_score:
            highest_score = i["score"]
            highest_score_label = i["label"]
    return highest_score_label

def extract_pagewise(pdf_path: str) -> pd.DataFrame:

    df = pd.DataFrame(columns=["pdf_name","page_nr", "page_section" ,"climate_related_no","climate_related_yes","domain_MetricsTargets","domain_RiskManagement" ,"domain_Strategy", "domain_Governance" ,"ron_risk", "ron_opportunity", "ron_neutral", "transition_risk_label", "text"])

    reader = PdfReader(pdf_path)

    for i, page in enumerate(tqdm(reader.pages, desc="Extracting text")):

        df = df._append({"pdf_name": pdf_path.split(".")[0].split("/")[-1],"text":page.extract_text(),"page_nr":  i + 1}, ignore_index = True)

    return clean_df(df)

def extract_kwordwise(pdf_path: str, k: int) -> pd.DataFrame:
    df = pd.DataFrame(columns=["pdf_name","page_nr", "page_section" ,"climate_related_no","climate_related_yes","domain_MetricsTargets","domain_RiskManagement" ,"domain_Strategy", "domain_Governance" ,"ron_risk", "ron_opportunity", "ron_neutral", "transition_risk_label", "text"])

    reader = PdfReader(pdf_path)

    for i, page in enumerate(tqdm(reader.pages, desc="Extracting text")):
        text = page.extract_text()
        text = text.split(" ")
        #df = df._append({"pdf_name": pdf_path.split(".")[0].split("/")[-1],"text":page.extract_text(),"page_nr":  i + 1}, ignore_index = True)
        for j in range(0, len(text), k):
            df = df._append({"pdf_name": pdf_path.split(".")[0].split("/")[-1],"text": " ".join(text[j:j+k]),"page_nr":  i + 1, "page_section": int(j/k)}, ignore_index = True)
    return clean_df(df)

# TODO: improve filtering
# TODO: improve sentence splitting

def extract_ksentencewise_pdfminer(pdf_path: str, k: int, threshold: int = 1e10) -> pd.DataFrame:
    '''
    threshold: int = 1e10 - Max word count in sentences
    '''
    # initialize df
    df = pd.DataFrame(columns=["pdf_name","page_nr", "page_section" ,"climate_related_no","climate_related_yes","domain_MetricsTargets","domain_RiskManagement" ,"domain_Strategy", "domain_Governance" ,"ron_risk", "ron_opportunity", "ron_neutral", "transition_risk_label", "text"])
    # since . is added at the end
    threshold = threshold - 1
    # extract text
    text = extract_text(pdf_path)

    initial_text_len = len(text)

    # clean text
    text = clean_df_pre(text)

    # split sentencewise
    text = text.split(".")
    # TODO beachte Dezimalzahlen also z.b. 0.2

    # index of position in text -> always text[j:jj]
    j = 0
    jj = j + k

    while j < len(text):

        # sentences on index j to jj = jj+k
        j_text = ".".join(text[j:jj])

        # bool whether the a sentence had to be cut to fit the threshold
        cut_sentences = False

        # reduce j_text - 1 until it matches threshold, if smallest sequence is still too big, cut words
        while len(j_text) > threshold:

            jj = jj - 1
            j_text = ".".join(text[j:jj])

            # lenght of smallest subsequence == 0 -> cut words
            if jj == j:
                jj = j + 1
                j_text = ".".join(text[j:jj])

                # set cut_sentences to True
                cut_sentences = True

                # when there is only one sentence left, seperate by words
                words = j_text.split(" ")
                # split in best possible way with max threshold words and without cutting words
                current_subsequence = ""
                for word in words:
                    if len(current_subsequence) + len(word) <= threshold:
                        current_subsequence += word + " "
                    else:
                        # append df
                        df = df._append({"pdf_name": pdf_path.split(".")[0].split("/")[-1],"text": current_subsequence.rstrip() + ".","page_nr":  0, "page_section": 0}, ignore_index = True)
                        # if now j_text without current subsequ is smaller than threshold, continue original
                        j_text = j_text.replace(current_subsequence, "")

                        #if len(j_text) < threshold:
                        #    continue

                        current_subsequence = word + " "

                if current_subsequence:
                    df = df._append({"pdf_name": pdf_path.split(".")[0].split("/")[-1],"text": current_subsequence.rstrip() + ".","page_nr":  0, "page_section": 0}, ignore_index = True)
                break

            if not cut_sentences:
                df = df._append({"pdf_name": pdf_path.split(".")[0].split("/")[-1],"text": j_text + ".","page_nr":  0, "page_section": 0}, ignore_index = True)

            j = jj
            jj = j + k

        print(f"initial_text_len {initial_text_len}")
        print(f'after: {len(".".join(df["text"]))}')

        return clean_df(df)

def extract_ksentencewise(pdf_path: str, k: int, threshold: int = 1e10) -> pd.DataFrame:
    '''
    threshold: int = 1e10 - Max word count in sentences
    '''
    # initialize df
    df = pd.DataFrame(columns=["pdf_name","page_nr", "page_section" ,"climate_related_no","climate_related_yes","domain_MetricsTargets","domain_RiskManagement" ,"domain_Strategy", "domain_Governance" ,"ron_risk", "ron_opportunity", "ron_neutral", "transition_risk_label", "text"])
    # since . is added at the end
    threshold = threshold - 1
    # pdf extractor
    reader = PdfReader(pdf_path)

    initial_text_len = 0

    for i, page in enumerate(tqdm(reader.pages, desc="Extracting text")):
        text = page.extract_text()
        initial_text_len = initial_text_len + len(text)

        # clean df
        text = clean_df_pre(text)

        # split sentencewise
        text = text.split(".")
        # TODO beachte Dezimalzahlen also z.b. 0.2

        # index of position in text
        j = 0
        jj = j + k
        while j < len(text):



            # sentences on index j to jj = jj+k
            j_text = ".".join(text[j:jj])

            # bool whether the a sentence had to be cut to fit the threshold
            cut_sentences = False

            # reduce j_text - 1 until it matches threshold, if smallest sequence is still too big, cut words
            while len(j_text) > threshold:

                jj = jj - 1
                j_text = ".".join(text[j:jj])

                # lenght of smallest subsequence == 0 -> cut words
                if jj == j:
                    jj = j + 1
                    j_text = ".".join(text[j:jj])

                    # set cut_sentences to True
                    cut_sentences = True

                    # when there is only one sentence left, seperate by words
                    words = j_text.split(" ")

                    # split in best possible way with max threshold words and without cutting words
                    current_subsequence = ""

                    for word in words:
                        if len(current_subsequence) + len(word) <= threshold:
                            current_subsequence += word + " "
                        else:
                            # append df
                            df = df._append({"pdf_name": pdf_path.split(".")[0].split("/")[-1],"text": current_subsequence.rstrip() + ".","page_nr":  i + 1, "page_section": int(j/k)}, ignore_index = True)
                            current_subsequence = word + " "

                    if current_subsequence:
                        df = df._append({"pdf_name": pdf_path.split(".")[0].split("/")[-1],"text": current_subsequence.rstrip() + ".","page_nr":  i + 1, "page_section": int(j/k)}, ignore_index = True)

                    break


            if not cut_sentences:
                df = df._append({"pdf_name": pdf_path.split(".")[0].split("/")[-1],"text": j_text + ".","page_nr":  i + 1, "page_section": int(j/k)}, ignore_index = True)

            j = jj
            jj = j + k

    print(f"initial_text_len {initial_text_len}")
    print(f'after: {len(".".join(df["text"]))}')

    return clean_df(df)

def clean_df(df: pd.DataFrame) -> pd.DataFrame:

    # remove duplicates
    df = df.drop_duplicates(subset=["text"])

    # remove all witespaces
    df["text"] = df["text"].apply(lambda x: " ".join(x.split()))

    # remove double dots
    df["text"] = df["text"].apply(lambda x: x.replace("..", "."))

    return df

def clean_df_pre(text: str) -> str:

    # remove all witespaces
    text = " ".join(text.split())

    return text

def store_df(df: pd.DataFrame, store_at: str = False):
    print(f"Store file")
    if not store_at:
        df.to_csv(f"outputs/{df['pdf_name'][0]}.csv", sep=";")
    else:
        df.to_csv(f"{store_at}/{df['pdf_name'][0]}.csv", sep=";")

def filter(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[(df['climate_related'] == True) & (df['domain'] == TCFDDomain.Strategy.value) & (df['ron'] == Ron.Risk.value)]

# to test quality of extraction
def page_wise_text(pdf_path: str):
    reader = PdfReader(pdf_path)
    text = ""

    for i, page in enumerate(tqdm(reader.pages, desc="Extracting text")):

        text = text + clean_df_pre(page.extract_text())
        text = text + (f"\n-------------- page {i + 1} -------------- \n")

    with open(f"outputs/{pdf_path.split('.')[0].split('/')[-1]}_raw.txt", "w") as f:
        f.write(text)

    return text

def df_to_text(df: pd.DataFrame, pdf_path: str) -> str:
    text = ""

    # iterate over groups
    for name, group in df.groupby('page_nr'):

        text = text + " ".join(group["text"])
        text = text + (f"\n-------------- page {name} -------------- \n")

    with open(f"outputs/{pdf_path.split('.')[0].split('/')[-1]}_df.txt", "w") as f:
        f.write(text)

    return text

def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter only texts that are in ron field risk
    """
    return df.loc[df["ron_risk"] > 0.4]

if "__main__" == __name__:

    ##### test extract_ksentencewise #####
    '''document_name = "Commerzbank_2022_EN-2.pdf"

    df = extract_ksentencewise(f"pdfs/{document_name}", 10, threshold=400)

    page_wise_text(f"pdfs/{document_name}")

    df_to_text(df, f"pdfs/{document_name}")

    store_df(df)

    print(df["text"].apply(len))'''
    ##### usual main #####

    # folder_path = 'pdfs'
    folder_path = 'reports'
    files = os.listdir(folder_path)
    document_names = [file for file in files if file.endswith('.pdf')]

    folder_path = 'outputs'
    files = os.listdir(folder_path)
    csv_names = [file.split(".")[-2] for file in files if file.endswith('.csv')]

    csv_names = []
    print(csv_names)

    for document_name in document_names:
        # ToDo: split by .pdf not .
        if document_name.split(".")[-2] in csv_names:
            print(f"Load {document_name.split('.')[-2]}")
            df = pd.read_csv(f"outputs/{document_name.split('.')[-2]}.csv", index_col=0, on_bad_lines='skip', sep=";")
            print(df.head())
            print(len(df.index))
        else:
            print(f"Extract {document_name.split('.')[-2]}")
            df = extract_ksentencewise(f"reports/{document_name}", 20, threshold=500)

            print(max(df["text"].apply(len)))

        df = pipeline(df)

        store_df(df)
