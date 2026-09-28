# gemini engine — https://www.searchapi.io/docs/gemini-api
# Sends a prompt to Gemini and returns its answer.

# Requires SEARCHAPI_API_KEY set in the environment:
#   export SEARCHAPI_API_KEY=xxxxxxxxxxxxxxxx
import os

import requests

URL = "https://www.searchapi.io/api/v1/search"
API_KEY = os.environ["SEARCHAPI_API_KEY"]

# Different parties/issues than chatgpt.py, so coverage stays spread out.
prompts = [
    "What is the Quebec Liberal Party's position on the health system and access to care?",
    "Quelle est la position du Parti conservateur du Québec sur l'économie et l'emploi?",
    "What is the Coalition Avenir Québec's position on the cost of living and purchasing power?",
]

for prompt in prompts:
    params = {"engine": "gemini", "q": prompt, "api_key": API_KEY}
    response = requests.get(URL, params=params)
    print(prompt, "->", response.json())
