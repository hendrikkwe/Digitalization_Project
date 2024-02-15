import os

import pandas as pd
import requests
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from enums import Ron, TCFDDomain
from pd_helpers import *
from pdf_extraction import extract_ksentencewise

USE_API = False

# Paths for output and input
PDF_FOLDER_PATH = "pdfs"
CSV_FOLDER_PATH = "outputs"

# API URL and headers, add your own token
BEARER_TOKEN = ""
headers = {"Authorization": "Bearer {BEARER_TOKEN}"}
API_URL_CLIMATE_RELATED = "https://api-inference.huggingface.co/models/climatebert/distilroberta-base-climate-detector"
API_URL_RON = "https://api-inference.huggingface.co/models/climatebert/distilroberta-base-climate-sentiment"
API_URL_TCFD = "https://api-inference.huggingface.co/models/climatebert/distilroberta-base-climate-tcfd"


##################################################################
#    Models: need to be outsourced to other file
##################################################################


if not USE_API:
    tokenizer_climate_related = AutoTokenizer.from_pretrained(
        "climatebert/distilroberta-base-climate-detector"
    )
    model_climate_related = AutoModelForSequenceClassification.from_pretrained(
        "climatebert/distilroberta-base-climate-detector"
    )

    tokenizer_ron = AutoTokenizer.from_pretrained(
        "climatebert/distilroberta-base-climate-sentiment"
    )
    model_ron = AutoModelForSequenceClassification.from_pretrained(
        "climatebert/distilroberta-base-climate-sentiment"
    )

    tokenizer_tcfd = AutoTokenizer.from_pretrained(
        "climatebert/distilroberta-base-climate-tcfd"
    )
    model_tcfd = AutoModelForSequenceClassification.from_pretrained(
        "climatebert/distilroberta-base-climate-tcfd"
    )


def add_climate_related_label(row: pd.Series) -> pd.Series:
    """Adds the labels for climate related to the row"""
    if USE_API:
        response = request_label_api(row["text"], API_URL_CLIMATE_RELATED)
    else:
        response = request_label_local(
            row["text"], tokenizer_climate_related, model_climate_related
        )

    if isinstance(type(response), dict):
        row["climate_related_no"] = response["no"]
        row["climate_related_yes"] = response["yes"]
    else:
        row["climate_related_no"] = 0
        row["climate_related_yes"] = 0

    return row


def add_domain_label(row: pd.Series) -> pd.Series:
    """Adds the labels for the TCFD domains to the row"""
    if USE_API:
        response = request_label_api(row["text"], API_URL_TCFD)
    else:
        response = request_label_local(row["text"][0:30], tokenizer_tcfd, model_tcfd)

    if isinstance(type(response), dict):
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
    """Adds the label for Risk Opportunity Neutral (RON) to the row

    Args:
        row (pd.Series): row to label

    Returns:
        pd.Series: labeled row
    """
    if USE_API:
        response = request_label_api(row["text"], API_URL_RON)
    else:
        response = request_label_local(row["text"], tokenizer_ron, model_ron)

    if isinstance(type(response), dict):
        row["ron_risk"] = response["risk"]
        row["ron_opportunity"] = response["opportunity"]
        row["ron_neutral"] = response["neutral"]
    else:
        row["ron_risk"] = 0
        row["ron_opportunity"] = 0
        row["ron_neutral"] = 0

    return row


def request_label_api(text: str, api_url: str) -> dict:
    """
    return format: [[{'label': 'no', 'score': 0.9543747901916504}, {'label': 'yes', 'score': 0.04562525078654289}]]


    Args:
        text (str): text to classify
        api_url (str): api url to send the request to

    Returns:
        dict: dict with structure {"label": "score"}
    """
    payload = {"inputs": text}
    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code == 200:
        print({r["label"]: r["score"] for r in response.json()[0]})
        return {r["label"]: r["score"] for r in response.json()[0]}
    else:
        return [[{"label": "api_failed", "score": 1}]]


def request_label_local(text: str, tokenizer, model) -> dict:
    """runs the model locally on the text and returns the probabilities for
    the labels

    return format for e.g. climate related:
        {'no': 0.9543747901916504, 'yes': 0.04562525078654289}

    Args:
        text (str): text to be classified
        tokenizer (_type_): tokenizer for the model
        model (_type_): model used

    Returns:
        (_dict_): dict with probabilities for the labels
    """
    input = tokenizer(text, return_tensors="pt")
    output = model(**input)
    probabilities = F.softmax(output.logits, dim=1)
    # predicted_class = torch.argmax(probabilities, dim=1).item()

    response = {}
    for key, label in model.config.id2label.items():
        response[label] = probabilities[0][key].item()
    return response


def pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Iterates through the dataframe and adds labels for climate related, domain and ron. Drops
    irrelevant chunks as descibed in the documentation.

    Args:
        df (pd.DataFrame): Dataframe with the extracted text chunks

    Returns:
        pd.DataFrame: _description_
    """
    for index, row in tqdm(df.iterrows()):

        # climate related label, update row in df with output of add_climate_related_label
        row = add_climate_related_label(row)
        df.loc[index] = row

        if row["climate_related_no"] < row["climate_related_yes"]:
            # print("Climate related")
            row = add_domain_label(row)
            df.loc[index] = row

            # if row["domain"] == TCFDDomain.Strategy.value:
            if (
                max(
                    [
                        row["domain_MetricsTargets"],
                        row["domain_RiskManagement"],
                        row["domain_Strategy"],
                        row["domain_Governance"],
                    ]
                )
                == row["domain_Strategy"]
            ):
                # print("Strategy")
                row = add_ron_label(row)
                df.loc[index] = row

    store_df(df)

    return df


def main(pdf_folder_path: str, csv_folder_path: str, reextract: bool = True):
    """Implements the pipeline. All pdfs in pdf_folder are extracted and processed.
    Resulting Dataframes are stored in csv_folder.


    Args:
        pdf_folder_path (str): _description_
        csv_folder_path (str): _description_
        reextract (bool, optional): If False, the csv files in csv_folder are
        loaded and processed, else only the lables are set again. Defaults to True.
    """
    files = os.listdir(pdf_folder_path)
    document_names = [file for file in files if file.endswith(".pdf")]
    # document_names = ["Allianz Global Investors GmbH_Asset Manager_EN_2022.pdf"]
    document_names = ["nachhaltigkeitsbericht-2021-2.pdf"]

    if reextract:
        files = os.listdir(csv_folder_path)
        csv_names = [file.split(".")[-2] for file in files if file.endswith(".csv")]
    else:
        csv_names = []

    for document_name in document_names:
        if document_name.split(".")[-2] in csv_names:
            print(f"Load {document_name.split('.')[-2]}")
            df = pd.read_csv(
                f"{csv_folder_path}/{document_name.split('.')[-2]}.csv",
                index_col=0,
                on_bad_lines="skip",
                sep=";",
            )
            print(df.head())
            print(len(df.index))
        else:
            print(f"Extract {document_name.split('.')[-2]}")
            df = extract_ksentencewise(
                f"{pdf_folder_path}/{document_name}", 20, threshold=500
            )
            print(max(df["text"].apply(len)))

        df = pipeline(df)
        store_df(df)


if "__main__" == __name__:

    main(PDF_FOLDER_PATH, CSV_FOLDER_PATH)
