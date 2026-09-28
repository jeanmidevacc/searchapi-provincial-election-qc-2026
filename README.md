# Searchapi provincial election qc 2026

Code for my side project on the 2026 Quebec provincial election: I collected
Google News, Google Search, Google Trends, ChatGPT and Gemini results with
SearchApi.io every day, plus the open data from Elections Quebec, and built a
report on top of it.

* The article that recaps the whole project:
  [the-odd-dataguy.com](https://the-odd-dataguy.com/en/blog/2026/09/26/searchapi-qc-election-2026/)
* The live report:
  [searchapi-qc-election-2026.html](https://the-odd-dataguy-files.s3.us-east-1.amazonaws.com/briefing_reports/searchapi-qc-election-2026.html)

## Folder

* **searchapi_examples/** : one small script per SearchApi.io engine used to
  collect the data (more details in its readme)
* **import_from_kaggle.py** : downloads the Kaggle dataset into `data/kaggle/`
* **build_html_report.py** : builds the HTML report into `reports/report.html`
* **chatgpt_embeddings.py** : computes the embeddings of the ChatGPT answers
  used in the report, cached in `data/cache/`
* **requirements.txt** : the packages needed to run all of the above

## Data

All the data collected is on Kaggle:
[quebec-election-2026-search-attention](https://www.kaggle.com/datasets/jeanmidev/quebec-election-2026-search-attention).

There are two folders in the dataset:

* `searchapi/` : one CSV per source, with a `DATA_DICTIONARY.md` describing
  every column. Worth a read before starting, some columns have their own
  conventions (like `value` vs `extracted_value` in Google Trends, or the
  `"Breakout"` value).
* `dgeq/` : the Elections Quebec data (electors, candidates, district
  boundaries, parties), refreshed on every new version of the dataset.

To download it locally into `data/kaggle/`:

```bash
pip install kaggle
# add your Kaggle API token in ~/.kaggle/kaggle.json
# (kaggle.com -> Account -> Settings -> Create New Token)

python import_from_kaggle.py                  # download into data/kaggle/
python import_from_kaggle.py --out some/dir    # download somewhere else
python import_from_kaggle.py --force           # delete data/kaggle/ and download again
```

## Report

The report is built with [briefing](https://pypi.org/project/briefing/) and
has 5 pages:

* **DGEQ** : number of candidates per party, and a map of the electors per km²
  in each district
* **Search & News** : Google News articles per party per day, and the top
  Google result for each party leader
* **Trends** : what's trending in Quebec, and the Google Trends interest and
  related topics for the 5 main parties
* **ChatGPT** : the question asked to ChatGPT for each party and issue (in
  French and English), and a 3D map of the answers to see how they move day
  after day
* **ChatGPT sources** : the websites cited by ChatGPT in its answers

To build it (after downloading the data):

```bash
pip install -r requirements.txt
python build_html_report.py
```

For the ChatGPT map, each answer is turned into an embedding with
[granite-embedding-97m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2),
then reduced to 3D with a PCA. The model runs on CPU and the first run takes a
few minutes, so the embeddings are cached in `data/cache/` and only the new
answers are computed on the next runs. Delete the cache file to start from
scratch.

## Sources
* [Elections Quebec - data page](https://www.dgeq.org/donnees.html)
* [SearchApi.io](https://www.searchapi.io/)
