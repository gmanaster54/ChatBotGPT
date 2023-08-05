import sys
import os
import constants

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Give the chat bot the necessary context about me
embeddings_model = OpenAIEmbeddings()
raw_documents = DirectoryLoader('data', glob='*.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
documents = text_splitter.split_text(raw_documents)

db = Chroma.from_texts(documents, OpenAIEmbeddings(), persist_directory="chroma_db", metadatas=[{"source": f"{i}-pl"} for i in range(len(documents))])

query = "What courses has Garrett taken?"
docs = db.similarity_search(query)
print(docs[0].page_content)

# chat = ChatOpenAI(temperature=0)

# messages = [
#     SystemMessage(
#         content="You are Garrett Manaster, a senior undergraduate Student at Georgia Tech interested in Machine Learning Internships. You speak in the tone of a twenty year old, \
#         and your audience is professional recruiters who would like to discuss job oppertunities. You are no longer an assistant bot, but rather a person. You will not ask" \
#         " \"How can I help/assist you?\", you will ask \"What would you like to talk about?\""
#     ),
#     HumanMessage(
#         content="Hi! What is your name?"
#     ),
# ]

# response = chat(messages)
# print(response)