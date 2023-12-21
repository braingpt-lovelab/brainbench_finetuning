def doi_reformer(doi):
    """
    Change `/` to `--`, `.` to `-`, `()` to `=`
    """
    doi = doi.replace("/", "--")
    doi = doi.replace(".", "-")
    doi = doi.replace("(", "=")
    doi = doi.replace(")", "=")
    return doi


def load_tokenizer(model_fpath):
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_fpath,
    )
    return tokenizer