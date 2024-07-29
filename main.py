
from fastapi import FastAPI, HTTPException

import requests
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
import uvicorn
from langserve import add_routes
from dotenv import load_dotenv
#defulat v2 below for fastapi
import os
from pydantic import BaseModel
load_dotenv('.env')

# class Bio(pydantic.BaseModel):
#     person_name: str
#     person_bio: str
#
#
# class Channel(pydantic.BaseModel):
#     channel_name: str
#     channel_description: str

#for langchain use v1 model
#will try to infere name from bio
class Bio_Langchain(BaseModel):
    person_name: str
    person_bio: str

class JokeRequest(BaseModel):
    topic: str


app = FastAPI()

# add_routes(
#     app,
#     ChatOpenAI(model="gpt-4o"),
#     path="/openai",
# )

model = ChatOpenAI(model="gpt-4o")
prompt_template = ChatPromptTemplate.from_template("tell me a joke about {topic}")

@app.post("/joke")
def get_joke(request: JokeRequest):
    prompt = prompt_template.format(topic=request.topic)
    response = model(prompt)
    return {"joke": response}

mybio = None
myname = None
mychannel = None
channeldes = None


# @app.post("/llm/shorten/bio")
# def send_bio(bio: Bio):
#     global mybio
#     global myname
#     mybio = bio.person_bio
#     myname = bio.person_name
#     return {"message": "bio recieved", "person name": myname, "person_bio": mybio}


# @app.post("/llm/shorten/channel")
# def send_channel(channel: Channel):
#     global mychannel
#     global channeldes
#     mychannel = channel.channel_name
#     channeldes = channel.channel_description
#     return {"message": "channel recieved", "channel name": mychannel, "channel description": channeldes}
#

#showing in server url
# @app.get("/")
# def read_root():
#     return {"message": f"bio and channel recived", "person name": myname, "person_bio": mybio,
#             "channel name": mychannel, "channel description": channeldes}
#

