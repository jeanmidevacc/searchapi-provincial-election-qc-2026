# google_rank_tracking engine — https://www.searchapi.io/docs/google-rank-tracking
# Top-100 organic SERP results for a query.

# Requires SEARCHAPI_API_KEY set in the environment:
#   export SEARCHAPI_API_KEY=xxxxxxxxxxxxxxxx
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
