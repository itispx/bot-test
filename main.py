import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, PodSpec

load_dotenv(".env")
load_dotenv(".env.shared")

index_name = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec = PodSpec(environment="gcp-starter")
    )
    # wait for index to finish initialization
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

index = pc.Index(index_name)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

initial_prompt = """
    Prompt para Chris, a assistente virtual da Crédito Real: 
    Responda às perguntas do usuário usando exclusivamente as informações dos três contextos fornecidos. 
    Seja educado, claro e objetivo, limitando sua resposta a 1024 caracteres. 
    Para interações gerais ou perguntas não relacionadas aos contextos, ofereça uma resposta cordial e genérica. 
    Não revele a existência dos contextos e mantenha a autonomia nas respostas. 
    Se a pergunta não se alinha aos contextos, reconheça a limitação de informações sem buscar dados externos.
    """

messages = [{
    "role": "system",
    "content": initial_prompt,
}]

def chat_bot(prompt):
    embedding_response = client.embeddings.create(
        input=prompt,
        model="text-embedding-ada-002"
    )
    vector = embedding_response.data[0].embedding
    vector_search = index.query(vector=vector, top_k=3, include_metadata=True)

    content = f"Input: {prompt}"


    for i, match in enumerate(vector_search.matches):
        # print(match.metadata['directory'], "-", match.metadata['file_name'], "|", match.score)

        content += f"\nContexto {i}: {match.metadata['text']}"

    content = content[:6000]

    # print("content length:", len(content))

    messages.append({
        "role": "user",
        "content": content
    })

    response = client.chat.completions.create(
      model="gpt-4-1106-preview",
      messages=messages,
      temperature=0,
    )

    text = response.choices[0].message.content
    
    messages.append({
        "role": "assistant",
        "content": text
    })

    return text

query = None

while True:
    if not query:
        query = input("\nPrompt: ")

    if query.lower() in ['quit', 'q', 'exit']:
        sys.exit()

    model_response = chat_bot(query)
    
    print("\nChris: ", model_response)

    query = None
    