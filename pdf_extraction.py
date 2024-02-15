import pandas as pd
from pdfminer.high_level import extract_text
from pypdf import PdfReader
from tqdm import tqdm


def extract_pagewise(pdf_path: str) -> pd.DataFrame:

    df = pd.DataFrame(
        columns=[
            "pdf_name",
            "page_nr",
            "page_section",
            "climate_related_no",
            "climate_related_yes",
            "domain_MetricsTargets",
            "domain_RiskManagement",
            "domain_Strategy",
            "domain_Governance",
            "ron_risk",
            "ron_opportunity",
            "ron_neutral",
            "transition_risk_label",
            "text",
        ]
    )

    reader = PdfReader(pdf_path)

    for i, page in enumerate(tqdm(reader.pages, desc="Extracting text")):

        df = df._append(
            {
                "pdf_name": pdf_path.split(".")[0].split("/")[-1],
                "text": page.extract_text(),
                "page_nr": i + 1,
            },
            ignore_index=True,
        )

    return clean_df(df)


def extract_kwordwise(pdf_path: str, k: int) -> pd.DataFrame:
    df = pd.DataFrame(
        columns=[
            "pdf_name",
            "page_nr",
            "page_section",
            "climate_related_no",
            "climate_related_yes",
            "domain_MetricsTargets",
            "domain_RiskManagement",
            "domain_Strategy",
            "domain_Governance",
            "ron_risk",
            "ron_opportunity",
            "ron_neutral",
            "transition_risk_label",
            "text",
        ]
    )

    reader = PdfReader(pdf_path)

    for i, page in enumerate(tqdm(reader.pages, desc="Extracting text")):
        text = page.extract_text()
        text = text.split(" ")
        # df = df._append({"pdf_name": pdf_path.split(".")[0].split("/")[-1],"text":page.extract_text(),"page_nr":  i + 1}, ignore_index = True)
        for j in range(0, len(text), k):
            df = df._append(
                {
                    "pdf_name": pdf_path.split(".")[0].split("/")[-1],
                    "text": " ".join(text[j : j + k]),
                    "page_nr": i + 1,
                    "page_section": int(j / k),
                },
                ignore_index=True,
            )
    return clean_df(df)


# TODO: improve filtering
# TODO: improve sentence splitting


def extract_ksentencewise_pdfminer(
    pdf_path: str, k: int, threshold: int = 1e10
) -> pd.DataFrame:
    """
    threshold: int = 1e10 - Max word count in sentences
    """
    # initialize df
    df = pd.DataFrame(
        columns=[
            "pdf_name",
            "page_nr",
            "page_section",
            "climate_related_no",
            "climate_related_yes",
            "domain_MetricsTargets",
            "domain_RiskManagement",
            "domain_Strategy",
            "domain_Governance",
            "ron_risk",
            "ron_opportunity",
            "ron_neutral",
            "transition_risk_label",
            "text",
        ]
    )
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
                        df = df._append(
                            {
                                "pdf_name": pdf_path.split(".")[0].split("/")[-1],
                                "text": current_subsequence.rstrip() + ".",
                                "page_nr": 0,
                                "page_section": 0,
                            },
                            ignore_index=True,
                        )
                        # if now j_text without current subsequ is smaller than threshold, continue original
                        j_text = j_text.replace(current_subsequence, "")

                        # if len(j_text) < threshold:
                        #    continue

                        current_subsequence = word + " "

                if current_subsequence:
                    df = df._append(
                        {
                            "pdf_name": pdf_path.split(".")[0].split("/")[-1],
                            "text": current_subsequence.rstrip() + ".",
                            "page_nr": 0,
                            "page_section": 0,
                        },
                        ignore_index=True,
                    )
                break

            if not cut_sentences:
                df = df._append(
                    {
                        "pdf_name": pdf_path.split(".")[0].split("/")[-1],
                        "text": j_text + ".",
                        "page_nr": 0,
                        "page_section": 0,
                    },
                    ignore_index=True,
                )

            j = jj
            jj = j + k

        print(f"initial_text_len {initial_text_len}")
        print(f'after: {len(".".join(df["text"]))}')

        return clean_df(df)


def extract_ksentencewise(pdf_path: str, k: int, threshold: int = 1e10) -> pd.DataFrame:
    """
    threshold: int = 1e10 - Max word count in sentences
    """
    # initialize df
    df = pd.DataFrame(
        columns=[
            "pdf_name",
            "page_nr",
            "page_section",
            "climate_related_no",
            "climate_related_yes",
            "domain_MetricsTargets",
            "domain_RiskManagement",
            "domain_Strategy",
            "domain_Governance",
            "ron_risk",
            "ron_opportunity",
            "ron_neutral",
            "transition_risk_label",
            "text",
        ]
    )
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
                            df = df._append(
                                {
                                    "pdf_name": pdf_path.split(".")[0].split("/")[-1],
                                    "text": current_subsequence.rstrip() + ".",
                                    "page_nr": i + 1,
                                    "page_section": int(j / k),
                                },
                                ignore_index=True,
                            )
                            current_subsequence = word + " "

                    if current_subsequence:
                        df = df._append(
                            {
                                "pdf_name": pdf_path.split(".")[0].split("/")[-1],
                                "text": current_subsequence.rstrip() + ".",
                                "page_nr": i + 1,
                                "page_section": int(j / k),
                            },
                            ignore_index=True,
                        )

                    break

            if not cut_sentences:
                df = df._append(
                    {
                        "pdf_name": pdf_path.split(".")[0].split("/")[-1],
                        "text": j_text + ".",
                        "page_nr": i + 1,
                        "page_section": int(j / k),
                    },
                    ignore_index=True,
                )

            j = jj
            jj = j + k

    print(f"initial_text_len {initial_text_len}")
    print(f'after: {len(".".join(df["text"]))}')

    return clean_df(df)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:

    # remove duplicates
    df = df.drop_duplicates(subset=["text"])

    # remove all witespaces
    # df.loc[["text"],:] = df.loc[["text"],:].apply(lambda x: " ".join(x.split()))

    # remove double dots
    # df.loc[["text"],:] = df.loc[["text"],:].apply(lambda x: x.replace("..", "."))

    # remove \n from text
    df["text"] = df["text"].str.replace("\n", " ")

    # remove text at the beginning and the end of the text
    df["text"] = df["text"].str.strip()

    return df


def clean_df_pre(text: str) -> str:

    # remove all witespaces
    text = " ".join(text.split())

    return text


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
    for name, group in df.groupby("page_nr"):

        text = text + " ".join(group["text"])
        text = text + (f"\n-------------- page {name} -------------- \n")

    with open(f"outputs/{pdf_path.split('.')[0].split('/')[-1]}_df.txt", "w") as f:
        f.write(text)

    return text
