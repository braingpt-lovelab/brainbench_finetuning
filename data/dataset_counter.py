import os
import json
import tqdm
import multiprocessing

import utils


def process_journal(journal, model_fpath, return_num_tokens):
    """
    Single process for counting the number of abstracts and fulltext files
    in a journal directory. Also counts the total number of tokens in the
    fulltext files and abstracts files (duplicate abstracts are not counted).
    """
    abstracts_dir = os.path.join(f"dataset/{journal}", "abstracts")
    fulltext_dir = os.path.join(f"dataset/{journal}", "fulltext")

    abstracts_count = len(os.listdir(abstracts_dir))
    abstracts_doi_set = set(os.listdir(abstracts_dir))
    assert abstracts_count == len(abstracts_doi_set)

    fulltext_count = len(os.listdir(fulltext_dir))
    fulltext_doi_set = set(os.listdir(fulltext_dir))
    assert fulltext_count == len(fulltext_doi_set)

    total_token_count = 0
    if return_num_tokens:
        tokenizer = utils.load_tokenizer(model_fpath=model_fpath)


        for file in tqdm.tqdm(fulltext_doi_set):
            with open(os.path.join(fulltext_dir, file), "r") as f:
                fulltext = json.load(f)["text"]
                total_token_count += len(tokenizer(fulltext)["input_ids"])

        for file in tqdm.tqdm(abstracts_doi_set):
            if file not in fulltext_doi_set:
                with open(os.path.join(abstracts_dir, file), "r") as f:
                    abstract = json.load(f)["text"]
                    total_token_count += len(tokenizer(abstract)["input_ids"])

    return journal, abstracts_count, fulltext_count, total_token_count


def main(model_fpath, num_processes, return_num_tokens):
    with open("journal_names.json", "r") as f:
        journal_names = json.load(f)

    journals = journal_names["journal_names"]
    total_abstracts_count = 0
    total_fulltext_count = 0
    total_token_count = 0
    zero_collector = []

    with multiprocessing.Pool(num_processes) as pool:
        results = []
        for journal in journals:
            results.append(
                pool.apply_async(
                    process_journal, 
                    args=[
                        journal, 
                        model_fpath, 
                        return_num_tokens
                    ]
                )
            )

        # Collect results
        for result in results:
            journal, abstracts_count, fulltext_count, token_count = result.get()

            if abstracts_count == 0 or fulltext_count == 0:
                zero_collector.append(
                    f"{journal}: abstracts [{abstracts_count}], fulltext [{fulltext_count}]"
                )

            total_abstracts_count += abstracts_count
            total_fulltext_count += fulltext_count
            total_token_count += token_count

            print(f"[{journal}]: abstracts [{abstracts_count}], fulltext [{fulltext_count}], tokens [{token_count}]")

        pool.close()
        pool.join()

    print(f"Total abstracts: [{total_abstracts_count}]")
    print(f"Total fulltext: [{total_fulltext_count}]")
    print(f"Token count (b): [{total_token_count / 10**9}]b")

    for journal in zero_collector:
        print(journal)


if __name__ == "__main__":
    model_fpath = "meta-llama/Llama-2-7b-chat-hf"
    return_num_tokens = False
    main(
        model_fpath, 
        num_processes=100, 
        return_num_tokens=return_num_tokens
    )