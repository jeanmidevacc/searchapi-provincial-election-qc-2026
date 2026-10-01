# SearchApi engines used by the election collector

All calls go through one REST endpoint, `https://www.searchapi.io/api/v1/search`,
differentiated by the `engine` query param. Each block below is a
copy-pasteable snippet (`requests`) showing how that engine is called and what
realistic parameters look like. Queries are mixed across parties/issues/languages
so no single one is singled out.

Requires `SEARCHAPI_API_KEY` set in the environment before running any of them:

```bash
export SEARCHAPI_API_KEY=xxxxxxxxxxxxxxxx
```

## google_news

News results for a search query, scoped by language / country / location.
Docs: <https://www.searchapi.io/docs/google-news?utm_source=Dev&utm_medium=Ambassador&utm_campaign=the-odd-dataguy.com>

```python
import os

import requests

URL = "https://www.searchapi.io/api/v1/search"
API_KEY = os.environ["SEARCHAPI_API_KEY"]

# Mixed sample: a generic topic + three different parties, alternating fr/en.
queries = [
    {"q": "élections Québec 2026", "hl": "fr", "lr": "lang_fr"},
    {"q": "Coalition Avenir Québec", "hl": "fr", "lr": "lang_fr"},
    {"q": "Parti conservateur du Québec", "hl": "fr", "lr": "lang_fr"},
    {"q": "Québec solidaire", "hl": "en", "lr": "lang_en"},
]

for extra in queries:
    params = {"engine": "google_news", "gl": "ca", "location": "Quebec, Canada", "api_key": API_KEY, **extra}
    response = requests.get(URL, params=params)
    print(extra["q"], "->", response.json())
```

## google_rank_tracking

Top-100 organic SERP results (position, domain, link, title, snippet) for a query.
Docs: <https://www.searchapi.io/docs/google-rank-tracking-api?utm_source=Dev&utm_medium=Ambassador&utm_campaign=the-odd-dataguy.com>

```python
import os

import requests

URL = "https://www.searchapi.io/api/v1/search"
API_KEY = os.environ["SEARCHAPI_API_KEY"]

# Mixed sample: a generic topic + two different parties, alternating fr/en.
queries = [
    {"q": "campagne électorale Québec 2026", "hl": "fr", "lr": "lang_fr"},
    {"q": "Parti Québécois", "hl": "fr", "lr": "lang_fr"},
    {"q": "Québec solidaire", "hl": "en", "lr": "lang_en"},
]

for extra in queries:
    params = {
        "engine": "google_rank_tracking",
        "gl": "ca",
        "location": "Quebec, Canada",
        "num": 100,
        "api_key": API_KEY,
        **extra,
    }
    response = requests.get(URL, params=params)
    print(extra["q"], "->", response.json())
```

## google_trends_trending_now

Not query-based: one call for a geo returns every currently-trending topic there.
Docs: <https://www.searchapi.io/docs/google-trends-trending-now-api?utm_source=Dev&utm_medium=Ambassador&utm_campaign=the-odd-dataguy.com>

```python
import os

import requests

URL = "https://www.searchapi.io/api/v1/search"
API_KEY = os.environ["SEARCHAPI_API_KEY"]

# Two lookback windows to illustrate the `time` param.
for time in ("past_24_hours", "past_7_days"):
    params = {"engine": "google_trends_trending_now", "geo": "CA-QC", "time": time, "api_key": API_KEY}
    response = requests.get(URL, params=params)
    print(time, "->", response.json())
```

## google_trends (query-based)

Same engine, four `data_type` modes. Values are relative indices, re-based per
request — only comparable within one comma-joined `q`, never across separate calls.
Docs: <https://www.searchapi.io/docs/google-trends?utm_source=Dev&utm_medium=Ambassador&utm_campaign=the-odd-dataguy.com>

```python
import os

import requests

URL = "https://www.searchapi.io/api/v1/search"
API_KEY = os.environ["SEARCHAPI_API_KEY"]

# One example per data_type, spread across different parties.
queries = [
    # TIMESERIES comparative — all 5 parties, one shared scale.
    {"data_type": "TIMESERIES", "q": "CAQ,PQ,PLQ,PCQ,QS", "geo": "CA-QC", "time": "today 3-m"},
    # TIMESERIES single-party bundle.
    {"data_type": "TIMESERIES", "q": "QS,Québec solidaire,Ruba Ghazal", "geo": "CA-QC", "time": "today 3-m"},
    # RELATED_QUERIES — `q` takes one string only.
    {"data_type": "RELATED_QUERIES", "q": "Parti Québécois", "geo": "CA-QC", "time": "today 3-m"},
    # RELATED_TOPICS — same shape, different response.
    {"data_type": "RELATED_TOPICS", "q": "Coalition Avenir Québec", "geo": "CA-QC", "time": "today 3-m"},
    # GEO_MAP — province breakdown (geo=CA, region=REGION).
    {"data_type": "GEO_MAP", "q": "Parti libéral du Québec", "geo": "CA", "region": "REGION", "time": "today 12-m"},
]

for extra in queries:
    params = {"engine": "google_trends", "api_key": API_KEY, **extra}
    response = requests.get(URL, params=params)
    print(extra["data_type"], extra["q"], "->", response.json())
```

## chatgpt

Sends a prompt to ChatGPT (with live web search) and returns its answer.
Docs: <https://www.searchapi.io/docs/chatgpt-api?utm_source=Dev&utm_medium=Ambassador&utm_campaign=the-odd-dataguy.com>

```python
import os

import requests

URL = "https://www.searchapi.io/api/v1/search"
API_KEY = os.environ["SEARCHAPI_API_KEY"]

# "Party's position on issue" prompts, mixing three parties/issues across fr/en.
prompts = [
    "Quelle est la position de la Coalition Avenir Québec sur le logement abordable?",
    "What is the Parti Québécois's position on immigration and admission levels?",
    "Quelle est la position de Québec solidaire sur la lutte contre les changements climatiques?",
]

for prompt in prompts:
    params = {"engine": "chatgpt", "q": prompt, "web_search": "true", "api_key": API_KEY}
    response = requests.get(URL, params=params)
    print(prompt, "->", response.json())
```

## gemini

Sends a prompt to Gemini and returns its answer.
Docs: <https://www.searchapi.io/docs/gemini-api?utm_source=Dev&utm_medium=Ambassador&utm_campaign=the-odd-dataguy.com>

```python
import os

import requests

URL = "https://www.searchapi.io/api/v1/search"
API_KEY = os.environ["SEARCHAPI_API_KEY"]

# Different parties/issues than chatgpt, so coverage stays spread out.
prompts = [
    "What is the Quebec Liberal Party's position on the health system and access to care?",
    "Quelle est la position du Parti conservateur du Québec sur l'économie et l'emploi?",
    "What is the Coalition Avenir Québec's position on the cost of living and purchasing power?",
]

for prompt in prompts:
    params = {"engine": "gemini", "q": prompt, "api_key": API_KEY}
    response = requests.get(URL, params=params)
    print(prompt, "->", response.json())
```
