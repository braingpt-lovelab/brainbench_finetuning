import os
import json


def count_files():
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
    zero_collector = []
    for journal in journal_names["journal_names"]:
        abstracts_dir = os.path.join(f"dataset/{journal}", "abstracts")
        fulltext_dir = os.path.join(f"dataset/{journal}", "fulltext")

        abstracts_count = len(os.listdir(abstracts_dir))
        fulltext_count = len(os.listdir(fulltext_dir))

        if abstracts_count == 0 or fulltext_count == 0:
            zero_collector.append(
                f"{journal}: abstracts [{abstracts_count}], fulltext [{fulltext_count}]"
            )

        total_abstracts_count += abstracts_count
        total_fulltext_count += fulltext_count

        print(f"[{journal}]: abstracts [{abstracts_count}], fulltext [{fulltext_count}]")
    
    print(f"Total abstracts: [{total_abstracts_count}]")
    print(f"Total fulltext: [{total_fulltext_count}]")
    for journal in zero_collector:
        print(journal)


if __name__ == "__main__":
    count_files()