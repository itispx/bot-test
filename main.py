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

index = pinecone.Index(index_name)

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

    print(query)

    embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    query_vector = embed_model.embed_documents(query)

    print(query_vector[0])

    query_results = index.query(queries=query_vector[0], top_k=3)

    print(query_results)

    context=""

    content = "Responda a query baseada no contexto passado: Query: "+query+"\nContexto: "+context

    prompt = HumanMessage(content=content)

    messages.append(prompt)

    res = chat.invoke(messages)
    print("\nChatGPT: ",res.content)

    gpt_response = AIMessage(content=res.content)

    messages.append(gpt_response)

    query = None
    