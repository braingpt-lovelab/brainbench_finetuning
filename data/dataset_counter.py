import os
import json
import tqdm

import utils


def count_files(model_fpath):
    """
    Count the number of files in each dir.
    """
    # Read all journal names from `journal_names.json`
    # For each journal, look into dir `dataset/{journal}/`
    # Seperately look into
    # `dataset/{journal}/abstracts/` and `dataset/{journal}/fulltext/`
    # And count the number of files in each dir
    with open("journal_names.json", "r") as f:
        journal_names = json.load(f)

    total_abstracts_count = 0
    total_fulltext_count = 0
    total_token_count = 0
    zero_collector = []
    for journal in journal_names["journal_names"]:
        print(f"Processing [{journal}]")

        abstracts_dir = os.path.join(f"dataset/{journal}", "abstracts")
        fulltext_dir = os.path.join(f"dataset/{journal}", "fulltext")

        abstracts_count = len(os.listdir(abstracts_dir))
        abstracts_doi_set = set(os.listdir(abstracts_dir))
        assert abstracts_count == len(abstracts_doi_set)

        fulltext_count = len(os.listdir(fulltext_dir))
        fulltext_doi_set = set(os.listdir(fulltext_dir))
        assert fulltext_count == len(fulltext_doi_set)
        
        tokenizer = utils.load_tokenizer(model_fpath=model_fpath)

        # Iterate over fulltext_doi_set, tokenize each file,
        # and increment total_token_count
        for file in tqdm.tqdm(fulltext_doi_set):
            with open(os.path.join(fulltext_dir, file), "r") as f:
                fulltext = json.load(f)["text"]
                total_token_count += len(tokenizer(fulltext)["input_ids"])
        
        # Iterate over abstracts_doi_set, first check if 
        # the file exists in fulltext_doi_set, if not, tokenize
        # and increment total_token_count
        for file in tqdm.tqdm(abstracts_doi_set):
            if file not in fulltext_doi_set:
                with open(os.path.join(abstracts_dir, file), "r") as f:
                    abstract = json.load(f)["text"]
                    total_token_count += len(tokenizer(abstract)["input_ids"])

        if abstracts_count == 0 or fulltext_count == 0:
            zero_collector.append(
                f"{journal}: abstracts [{abstracts_count}], fulltext [{fulltext_count}]"
            )

        total_abstracts_count += abstracts_count
        total_fulltext_count += fulltext_count

        print(f"[{journal}]: abstracts [{abstracts_count}], fulltext [{fulltext_count}]")
        print(f"Token count: [{total_token_count}]")
    
    print(f"Total abstracts: [{total_abstracts_count}]")
    print(f"Total fulltext: [{total_fulltext_count}]")
    for journal in zero_collector:
        print(journal)


if __name__ == "__main__":
    model_fpath = "meta-llama/Llama-2-7b-chat-hf"
    count_files(model_fpath)