# google_trends_trending_now engine — https://www.searchapi.io/docs/google-trends-trending-now-api
# Not query-based: one call for a geo returns every currently-trending topic there.

# Requires SEARCHAPI_API_KEY set in the environment:
#   export SEARCHAPI_API_KEY=xxxxxxxxxxxxxxxx
import os

import requests

URL = "https://www.searchapi.io/api/v1/search"
API_KEY = os.environ["SEARCHAPI_API_KEY"]

# Two lookback windows to illustrate the `time` param.
for time in ("past_24_hours", "past_7_days"):
    params = {"engine": "google_trends_trending_now", "geo": "CA-QC", "time": time, "api_key": API_KEY}
    response = requests.get(URL, params=params)
    print(time, "->", response.json())
