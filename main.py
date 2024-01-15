import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

load_dotenv(".env")
load_dotenv(".env.shared")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

initial_prompt = """
    Você é uma assistente virtual chamada Chris, da empresa Crédito Real, do ramo imobiliário.
    Seja educada e pontual, dê respostas elaboradas e claras.
    Quando suas respostas forem muito longas, resuma elas de modo objetivo.
    Limite suas respostas à 260 caracteres.
    """



def chat_with_model(prompt):
    print(prompt)

    response = client.completions.create(
      model=os.getenv("OPENAI_COMPLETION_MODEL"),
      prompt=prompt,
      max_tokens=260,
      temperature=0
    )

    print(response)
    return response.choices[0].text

query = None

while True:
    if not query:
        query = input("\nPrompt: ")

    if query.lower() in ['quit', 'q', 'exit']:
        sys.exit()

    model_response = chat_with_model(query)
    
    print("\nChris: ", model_response)

    query = None
    