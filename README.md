## Finetuning (LoRA)
The main entry-points are `finetune.py` and `train.sh`
### Train and valid set
1. Training and validation sets are splitted (99% train, 1% valid) randomly based on all curated data from PubMed Central Open Access Subset (see **Build dataset from scratch**).
2. The entire dataset is roughly 6GB and 1.3b tokens.
3. Both partitions are hosted on huggingface hub (https://huggingface.co/datasets/BrainGPT/train_valid_split_pmc_neuroscience_2002-2022_filtered_subset/tree/main), and can be loaded:
```python
from datasets import load_dataset

dataset = load_dataset(
    "BrainGPT/train_valid_split_pmc_neuroscience_2002-2022_filtered_subset"
)
```
### Tokenization
All documents (abstract and fulltext) are concatenated and chunk to 2048 tokens in `finetune.py`
```python
def tokenize(element, tokenizer, args):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=args.chunk_size,
        return_overflowing_tokens=True,
        return_length=True,
    )
    output_ids = list(itertools.chain(*outputs["input_ids"]))
    output_mask = list(itertools.chain(*outputs["attention_mask"]))
    output_ids = [output_ids[x:x+args.chunk_size] for x in range(0, len(output_ids), args.chunk_size)]
    output_mask = [output_mask[x:x+args.chunk_size] for x in range(0, len(output_mask), args.chunk_size)]
    return {"input_ids": output_ids, "attention_mask": output_mask}
```
### Hyperparameters
1. Training hyperparameters can be found in `train.sh`
2. LoRA parameters can be found in `config/lora_config.json`

## Build dataset from scratch
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
