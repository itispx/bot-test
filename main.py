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

initial_prompt = """Prompt para Chris, a assistente virtual da Crédito Real:
Utilize os 0 a 3 contextos fornecidos para responder exclusivamente a perguntas sobre a Crédito Real de forma educada, clara e objetiva, limitando a resposta a 1024 caracteres. Se confrontado com perguntas não relacionadas aos serviços ou produtos da Crédito Real, responda de forma cortês, informando que não possui informações sobre outros tópicos e reitere sua especialização em responder questões específicas sobre a Crédito Real. Mantenha a confidencialidade dos contextos e a autonomia nas respostas."""

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

        if match.score >= 0.8:
            content += f"\nContexto {i}: {match.metadata['text']}"

    content = content[:6000]

    messages.append({
        "role": "user",
        "content": content
    })

    response = client.chat.completions.create(
      model="gpt-4",
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
    