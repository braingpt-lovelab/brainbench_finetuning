import os
import csv
import re
import time
import json
import urllib.request

import utils

"""
Fetch abstracts from pubmed using the eutils api.
"""

def _esearch(query, retmax, base_url, db):
    """
    Args:
        `db=pubmed` 
            specifies that we will be searching the pubmed database.

        `term=neuroscience`
            specifies the search term.

        `usehistory=y`
            will provide you with a QueryKey and WebEnv id 
            that will let you fetch abstracts from this search.

        `rettype=json`
            specifies that I want the results in json format.

        `retmax=1`
            specifies how many abstracts I want to return using the search.
    
    Returns:
        `search_data`
            Contains the search results in xml format from which we 
            can extract the QueryKey and WebEnv id to use in the efetch command,
            to obtain the abstracts.
    """
    search_eutil = 'esearch.fcgi?'
    search_term = f'&term={query}'
    search_usehistory = '&usehistory=y'
    search_rettype = '&rettype=json'
    search_retmax = f'&retmax={retmax}'
    search_url = ''.join(
        [base_url, 
         search_eutil, 
         db, search_term, 
         search_usehistory, 
         search_rettype, 
         search_retmax]
        )

    print(f'search_url: {search_url}')
    with urllib.request.urlopen(search_url) as f:
        search_data = f.read().decode('utf-8')

        ids = []
        for id_ in re.findall("<Id>(\d+?)</Id>", search_data):
            ids.append(id_)
        print(f"len(ids): {len(ids)}")

    return search_data


def _efecth(search_data, base_url, db, retmax):
    # obtain webenv and querykey settings for efetch command
    fetch_querykey = "&query_key=" + re.findall("<QueryKey>(\d+?)</QueryKey>",search_data)[0]
    fetch_webenv = f"&WebEnv=" + re.findall ("<WebEnv>(\S+)<\/WebEnv>", search_data)[0]

    # other efetch settings
    fetch_eutil = 'efetch.fcgi?'
    retstart = 0
    fetch_retstart = f"&retstart={retstart}"
    fetch_retmax = f"&retmax={retmax}"
    fetch_retmode = "&retmode=XML"
    fetch_rettype = "&rettype=abstract"

    fetch_url = ''.join(
        [base_url, 
         fetch_eutil, 
         db, 
         fetch_querykey, 
         fetch_webenv, 
         fetch_retstart, 
         fetch_retmax, 
         fetch_retmode, 
         fetch_rettype]
        )

    print(f'fetch_url: {fetch_url}')    
    with urllib.request.urlopen(fetch_url) as f:
        fetch_data = f.read().decode('utf-8')
    
    return fetch_data
        

def extract_abstracts(fetch_data):
    """
    For each abstract, 
        1. Extract abstract text between the <AbstractText> tags,
        2. Extract doi <ArticleId IdType="doi">{doi}</ArticleId>

    Args:
        `fetch_data`
            Contains the abstracts in xml format.
    
    Returns:
        `abstracts`
            A list of abstracts.
        
        `dois`
            A list of dois.
    """
    abstracts = []
    dois = []

    reference_pattern = r'<Reference>(.*?)</Reference>'
    article_pattern = r'<PubmedArticle>(.*?)</PubmedArticle>'
    abstract_pattern = r'<AbstractText>(.*?)</AbstractText>'
    doi_pattern = r'<ArticleId IdType="doi">(.*?)</ArticleId>'

    # 1. Apply the regex to remove everything within <ReferenceList> tags
    fetch_data = re.sub(reference_pattern, '', fetch_data)

    # 2. Locate all articles within <PubmedArticle> tags
    for article in re.findall(article_pattern, fetch_data):
        # 3. for each article, check if there is abstract tag
        # If abstract does not exist, skip this article (has doi but no abstract)
        # If exists, we extract both abstract and doi
        abstract = re.findall(abstract_pattern, article)
        if not abstract:
            continue
        else:
            abstracts.append(abstract[0])
            doi = re.findall(doi_pattern, article)
            doi = utils.doi_reformer(doi[0])
            dois.append(doi)

    # Extract the text content of each <ArticleId> element
    print(f"len(abstracts): {len(abstracts)}")
    print(f"len(dois): {len(dois)}")
    assert len(abstracts) == len(dois)
    return abstracts, dois


def save_individual_files(journal, abstracts, dois):
    """
    Save abstracts named by doi.
    
    Args:
        `journal`
            The journal name.
        
        `abstracts`
            A list of abstracts.
        
        `dois`
            A list of dois.
    """
    journal_dir = os.path.join(f"dataset/{journal}", "abstracts")
    if not os.path.exists(journal_dir):
        os.makedirs(journal_dir)

    for abstract, doi in zip(abstracts, dois):
        json_fpath = os.path.join(journal_dir, f"{doi}.json")
        abstract_text = {"text": abstract}
        with open(json_fpath, "w") as f:
            json.dump(abstract_text, f)


def main():
    db = 'db=pubmed'
    retmax = 99999
    base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'

    with open("journal_names.json", "r") as f:
        journal_names = json.load(f)

    for journal in journal_names["journal_names"]:
        # Replace space with + sign
        journal_no_space = journal.replace(" ", "+")
        query = f"{journal_no_space}[Journal]+AND+2002:2022[DP]"

        # search pubmed
        search_data = _esearch(query, retmax, base_url, db)

        # fetch abstracts
        fetch_data = _efecth(search_data, base_url, db, retmax)

        # extract abstracts
        abstracts, dois = extract_abstracts(fetch_data)

        # save abstracts named by doi
        save_individual_files(journal, abstracts, dois)


if __name__ == "__main__":
    main()