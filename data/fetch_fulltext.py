import os
import json
import subprocess
import multiprocessing
import pandas as pd

import utils


def pubget(query, journal):
    if not os.path.exists(f"./pubget_data_{journal}"):
        subprocess.run(["pubget", "run", f"./pubget_data_{journal}", "-q", query])
    else:
        print(f"pubget_data_{journal} already exists, skipping pubget")


def save_individual_file(text_fpath, metadata_fpath, journal_dir):
    df_text = pd.read_csv(text_fpath)
    df_text_abstract = df_text["abstract"]
    df_text_body = df_text["body"]
    df_doi = pd.read_csv(metadata_fpath)["doi"]
    
    for abstract, body, doi in zip(df_text_abstract, df_text_body, df_doi):
        if not isinstance(doi, str):
            continue
        
        doi = utils.doi_reformer(doi)
        json_fpath = os.path.join(journal_dir, f"{doi}.json")

        if not isinstance(abstract, str):
            abstract = ""
        if not isinstance(body, str):
            body = ""

        full_text = {"text": abstract + "\n" + body}
        with open(json_fpath, "w") as f:
            json.dump(full_text, f)


def process_journal(journal):
    journal_code_name = utils.journal_reformer(journal, mode="fulltext")
    query = f"({journal_code_name}[Journal]) AND (2002[Publication Date] : 2022[Publication Date])"
    query = utils.query_reformer(journal, query, mode="fulltext")

    print(f"\n\n\nFetching fulltext for {journal}\n\n\n")
    pubget(query, journal)

    for root, dirs, files in os.walk(f"./pubget_data_{journal}"):
        for file in files:
            if file.endswith("text.csv"):
                text_fpath = os.path.join(root, file)
            elif file.endswith("metadata.csv"):
                metadata_fpath = os.path.join(root, file)

    journal_dir = os.path.join(f"dataset/{journal}", "fulltext")
    if not os.path.exists(journal_dir):
        os.makedirs(journal_dir)

    save_individual_file(text_fpath, metadata_fpath, journal_dir)

    subprocess.run(["rm", "-rf", f"./pubget_data_{journal}"])


def main(num_processes):
    with open("journal_names.json", "r") as f:
        journal_names = json.load(f)

    with multiprocessing.Pool(num_processes) as pool:
        # Use apply_async to parallelize processing for each journal
        results = [
            pool.apply_async(
                process_journal, 
                args=(journal,)
            ) for journal in journal_names["journal_names"]
        ]
        
        # Wait for all processes to finish
        pool.close()
        pool.join()

        # Retrieve results if needed
        for result in results:
            result.get()


if __name__ == "__main__":
    main(num_processes=100)
