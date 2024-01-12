import os
import sys
import time

import PyPDF2
from dotenv import load_dotenv
import pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

load_dotenv()

# init pinecone
index_name = "bot-test-index"

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="gcp-starter"
)

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=1536,
        metric='cosine'
    )
    # wait for index to finish initialization
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)

index = pinecone.Index(index_name)

context = ""

# creating a pdf reader object
reader = PyPDF2.PdfReader('example.pdf')

for page in reader.pages:
    context = context + page.extract_text()

# init chat
chat = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model='gpt-3.5-turbo'
)

initial_prompt = """
    Você é uma assistente virtual de uma empresa chamada Crédito Real chamada Chris, do ramo imobiliário.
    Seja educada e pontual, dê respostas elaboradas e claras.
    Quando suas respostas forem muito longas, resuma elas de modo objetivo.
    """

messages = [
    SystemMessage(content=initial_prompt),
]

query = None

while True:
    if not query:
        query = input("\nPrompt: ")

    if query in ['quit', 'q', 'exit']:
        sys.exit()

    content = "Responda a query baseada no contexto passado: Query: "+query+"\nContexto: "+context

    prompt = HumanMessage(content=content)

    messages.append(prompt)

    res = chat.invoke(messages)
    print("\nChatGPT: ",res.content)

    gpt_response = AIMessage(content=res.content)

    messages.append(gpt_response)

    query = None
    