# google_news engine — https://www.searchapi.io/docs/google-news

# Requires SEARCHAPI_API_KEY set in the environment:
#   export SEARCHAPI_API_KEY=xxxxxxxxxxxxxxxx
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
