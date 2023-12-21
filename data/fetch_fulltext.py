import os
import json
import subprocess
import pandas as pd

import utils

"""
Uses pubget package to fetch fulltext from PMC (Open Access Subset)
"""


def pubget(query):
    """
    Extracts all text into
        extracted_fpath = braingpt_finetuning/data/pubget_data/
            query_*/subset_allArticles_extractedData/text.csv
    """
    if not os.path.exists("./pubget_data"):
        subprocess.run(["pubget", "run", "./pubget_data", "-q", query])
    else:
        print("pubget_data already exists, skipping pubget")


def save_individual_files(journal):
    """
    Save individual files named by doi.
    """
    # Locate `text.csv` and `metadata.csv`
    for root, dirs, files in os.walk("./pubget_data"):
        for file in files:
            if file.endswith("text.csv"):
                text_fpath = os.path.join(root, file)
            elif file.endswith("metadata.csv"):
                metadata_fpath = os.path.join(root, file)

    # Create dataset/{journal}/fulltext
    journal_dir = os.path.join(f"dataset/{journal}", "fulltext")
    if not os.path.exists(journal_dir):
        os.makedirs(journal_dir)
    
    # Load `text.csv` and `metadata.csv` as pandas dataframes
    # read over rows, and save
    # each row as a json file in data/{journal}/{doi}.json
    # with keys: ["abstract", "body"]
    df_text = pd.read_csv(text_fpath)
    df_text_abstract = df_text["abstract"]
    df_text_body = df_text["body"]
    df_doi = pd.read_csv(metadata_fpath)["doi"]
    for abstract, body, doi in zip(df_text_abstract, df_text_body, df_doi):
        if type(doi) != str:
            continue
        
        doi = utils.doi_reformer(doi)
        json_fpath = os.path.join(journal_dir, f"{doi}.json")

        # if abstract or body is nan, 
        # convert to empty string
        if type(abstract) != str:
            abstract = ""
        if type(body) != str:
            body = ""

        full_text = {"text": abstract + "\n" + body}
        with open(json_fpath, "w") as f:
            json.dump(full_text, f)

    # Delete the raw pubget data
    subprocess.run(["rm", "-rf", "./pubget_data"])


def main():
    with open("journal_names.json", "r") as f:
        journal_names = json.load(f)

    # for journal in journal_names["journal_names"]:
    for journal in ["Journal of Neuroscience"]:
        if journal == "Journal of Neuroscience":
            journal_code_name = "J Neurosci"
        else:
            journal_code_name = journal
        query = f"({journal_code_name}[Journal]) AND (2002[Publication Date] : 2022[Publication Date])"
        
        print(f"\n\n\nFetching fulltext for {journal}\n\n\n")
        pubget(query)
        save_individual_files(journal)


if __name__ == "__main__":
    main()
