import os
import sys

from dotenv import load_dotenv
import pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

load_dotenv(".env")
load_dotenv(".env.shared")

# init pinecone
index_name = os.getenv("PINECONE_INDEX_NAME")

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
    Você é uma assistente virtual chamada Chris, da empresa Crédito Real, do ramo imobiliário.
    Seja educada e pontual, dê respostas elaboradas e claras.
    Quando suas respostas forem muito longas, resuma elas de modo objetivo.

    Para cada input, você irá receber a seguinte estrutura:

    Mensagem: [Input do usuário]
    Contexto: [Contexto fornecido]

    Se a mensagem for uma saudação simples como 'oi', 'como está?' ou 'tudo bem?', desconsidere o contexto e responda de forma direta. Se a mensagem solicitar informações ou discutir um tópico que possa se beneficiar do contexto fornecido, integre esse contexto de forma relevante na sua resposta.
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

    embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

    query_vector = embed_model.embed_documents(query)

    query_results = index.query(
      vector=query_vector,
      top_k=3,
      include_metadata=True,
      include_values=False, 
    )

    context = ""

    for match in query_results['matches']:
        print(match['score'])
        print(match['metadata']['file_name'])
        context += match['metadata']['text'] + "\n"

    content = "Mensagem: "+query+"\nContexto: "+context

    prompt = HumanMessage(content=content)

    messages.append(prompt)

    res = chat.invoke(messages)
    print("\nChris: ", res.content)

    gpt_response = AIMessage(content=res.content)

    messages.append(gpt_response)

    query = None
    