# chatgpt engine — https://www.searchapi.io/docs/chatgpt
# Sends a prompt to ChatGPT (with live web search) and returns its answer.

# Requires SEARCHAPI_API_KEY set in the environment:
#   export SEARCHAPI_API_KEY=xxxxxxxxxxxxxxxx
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
