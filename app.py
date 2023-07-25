import os
from dotenv import load_dotenv

import http.client
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
import streamlit as st
from langchain.schema import SystemMessage

load_dotenv()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

#1.Tool for search
# def search(query):
#     url = "https://google.serper.def/search" 
    
#     payload = json.dumps({
#     "q": query
#     })
    
#     headers = {
#     'X-API-KEY': serper_api_key,
#     'Content-Type': 'application/json'
#     }
    
#     response = requests.request("POST",url,headers=headers, data=payload)
    
#     print(response.text)
    
#     return response.text

# search("what is meta's thread product")

#1.Tool for search
conn = http.client.HTTPSConnection("google.serper.dev")
def search(query):
    payload = json.dumps({
    "q": query
    })
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    data = res.read()
    print(data)
    return data

search("what is meta's thread product")

#2.Tool for scraping
def scrape_website(objective: str, url: str):
    print("Scraping wesite...")
    header = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    data = {
        "url": url
    }

    data_json = json.dumps(data)

    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers=header, data=data_json)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content,"html.parser")
        text = soup.get_text()
        print("Content:",text)

        return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")
    
scrape_website("what is langchain", "https://python.langchain.com/en/latest/index.html")
