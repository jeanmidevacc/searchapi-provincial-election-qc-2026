# google_trends engine (query-based) — https://www.searchapi.io/docs/google-trends
# Values are relative indices, re-based per request — only comparable within
# one comma-joined `q`, never across separate calls.

# Requires SEARCHAPI_API_KEY set in the environment:
#   export SEARCHAPI_API_KEY=xxxxxxxxxxxxxxxx
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
