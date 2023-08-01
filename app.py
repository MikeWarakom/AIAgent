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
        # encoded_text = text.encode('utf-8')
        print("Content:",text)

        if len(text) > 10000:
            output = summary(objective,text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")
    
def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])
    
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output

class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(description="The objective & task that users give to the agent")
    url: str= Field(description="The url of the website to be scraped")

class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and object"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self,objective: str, url: str):
        return scrape_website(objective,url)
    
    def _arun(self, url: str):
        raise NotADirectoryError("Error here")

tools = [
    Tool(
        name = "Search",
        func = search,
        description = "useful for when you need to answer questions about current event, data. You should ask targeted questions."
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results;
    you do not make things up, you wil try as hard as possible to gather facts and data to back up the research.

    Please make ure you complete the objective above with the following rules:
    1. You should do enough research to gather as much information as possible about the objective
    2. If there are url of relevant links & articles, you will scrape it to gather more information
    3. After scraping and search, you should think "is there any new things I should search and scraping based on the I collected to increase research quality?" If answer is yes, continue, but do not do this process more than 3 iterations."
    4. You should not make things up, you should only write facts and data that you have gathered
    5. In the final outpu, you should include all reference data and links to back up your research. You should include all references.
    5. In the final outpu, you should include all reference data and links to back up your research. You should include all references.
    """
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs = agent_kwargs,
    memory = memory,
)

def main():
    st.set_page_config(page_title="Ai research agent", page_icon=":bird:")

    st.header("AI research agent :bird:")
    query = st.text_input("Research goal")

    if query:
        st.write("Doing research for ", query)

        result = agent({"input": query})

        st.info(result['output'])

if __name__ == '__main__':
    main()