# Searchapi provincial election qc 2026

## Folder

**searchapi_examples** : A collecton of te code snippet to extract various informations related to the quebec proincial election



## Data

The collected SearchApi.io data (Google News, Google Search rankings, Google
Trends, and ChatGPT/Gemini party-position answers) is published as a Kaggle
dataset:
[kaggle.com/datasets/jeanmidev/quebec-election-2026-search-attention](https://www.kaggle.com/datasets/jeanmidev/quebec-election-2026-search-attention).
One flat CSV per source, plus `DATA_DICTIONARY.md` — the full column
reference, including a few conventions worth reading first (e.g. Google
Trends' `value` vs `extracted_value` and the `"Breakout"` sentinel).

`import_from_kaggle.py` downloads it locally into `data/kaggle/` (gitignored):

```bash
pip install kaggle
# put your Kaggle API token at ~/.kaggle/kaggle.json
# (kaggle.com -> Account -> Settings -> Create New Token)

python import_from_kaggle.py                  # downloads + unzips into data/kaggle/
python import_from_kaggle.py --out some/dir    # download elsewhere
python import_from_kaggle.py --force           # re-download even if data/kaggle/ isn't empty
```

## Sources
* [Elections Quebec - data page](https://www.dgeq.org/donnees.html)