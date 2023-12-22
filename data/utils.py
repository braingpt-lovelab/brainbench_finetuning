def doi_reformer(doi):
    """
    Change `/` to `--`, `.` to `-`, `()` to `=`
    """
    doi = doi.replace("/", "--")
    doi = doi.replace(".", "-")
    doi = doi.replace("(", "=")
    doi = doi.replace(")", "=")
    return doi


def journal_reformer(journal_name, mode="fulltext"):
    """
    Case-by-case mapper for journal names, 
    only if the original journal name is not
    searchable in the pubmed API.
    """
    if journal_name == "Journal of Neuroscience":
        journal_code = "J Neurosci"
    elif journal_name == "PNAS":
        journal_code = "Proc Natl Acad Sci U S A"
    else:
        journal_code = journal_name
    
    if mode == "abstract":
        journal_code = journal_code.replace(" ", "+")

    return journal_code


def query_reformer(journal, query, mode):
    """
    Case-by-case mapper for query, 
    only if we want further filtering
    on top of:

    if mode == "fulltext":
        {journal_code_name}[Journal]) AND (2002[Publication Date] : 2022[Publication Date])
    if mode == "abstract":
        {journal_code_name}[Journal]+AND+2002:2022[DP]
    """
    subset_journals = [
        "eLife", "Science Advances", "Nature Communications", 
        "PNAS", "Nature", "Cell", "Cell Reports", "EMBO J"
    ]
    if mode == "fulltext":
        if journal in subset_journals:
            query += " AND neuroscience"
        else:
            return query
        return query
    
    elif mode == "abstract":
        if journal in subset_journals:
            query += "+AND+neuroscience"
        else:
            return query
        return query
        

def load_tokenizer(model_fpath):
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_fpath,
    )
    return tokenizer