import pydantic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

#defulat v2 below
class Bio(pydantic.BaseModel):
    person_name: str
    person_bio: str


class Channel(pydantic.BaseModel):
    channel_name: str
    channel_description: str


app = FastAPI()

mybio = None
myname = None
mychannel = None
channeldes = None


@app.post("/llm/shorten/bio")
def send_bio(bio: Bio):
    global mybio
    global myname
    mybio = bio.person_bio
    myname = bio.person_name
    return {"message": "bio recieved", "person name": myname, "person_bio": mybio}


@app.post("/llm/shorten/deschannels")
def send_channel(channel: Channel):
    global mychannel
    global channeldes
    mychannel = channel.channel_name
    channeldes = channel.channel_description
    return {"message": "channel recieved", "channel name": mychannel, "channel description": channeldes}


#showing in server url
@app.get("/")
def read_root():
    return {"message": f"bio and channel recived", "person name": myname, "person_bio": mybio,
            "channel name": mychannel, "channel description": channeldes}
