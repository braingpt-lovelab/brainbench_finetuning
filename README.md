## Data
All regarding dataset download and curation is in `data`
1. `python fetch_journal_names.py` will extract top neuroscience journal names (based on https://research.com/journals-rankings/neuroscience) into `journal_names.json`
2. `python fetch_fulltext.py` will download articles from the above journals whose full-text versions are accessible from PubMed Central Open Access Subset.
3. `python fetch_abstract.py` will download article abstracts from the above journals that are available via PubMed E-utilities API.

### Dataset Structure
```
.
├── data
│   └── dataset
│       ├── {journal_name}
│            ├── fulltext
│            └── abstract
│   ├── fetch_journal_names.py
│   ├── fetch_fulltext.py
│   └── fetch_abstract.py
```
Both `fulltext/` and `abstract/` follow the same structure where each json file is an article named by its doi.
