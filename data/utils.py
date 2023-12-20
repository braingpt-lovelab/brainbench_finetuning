def doi_reformer(doi):
    """
    Change `/` to `--`, `.` to `-`, `()` to `=`
    """
    doi = doi.replace("/", "--")
    doi = doi.replace(".", "-")
    doi = doi.replace("(", "=")
    doi = doi.replace(")", "=")
    return doi