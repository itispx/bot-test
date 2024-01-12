import os
import sys
import time

import uuid
import PyPDF2
from dotenv import load_dotenv
import pinecone
from langchain_openai import OpenAIEmbeddings

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

embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# creating a pdf reader object
reader = PyPDF2.PdfReader('example.pdf')

for page in reader.pages:
    page_text = page.extract_text()

    paragraphs = page_text.split("\n")

    for paragraph in paragraphs:
        query_vector = embed_model.embed_documents(paragraph)
        
        data_to_insert = []
        
        for vector in query_vector:
            id = str(uuid.uuid4())
            data_to_insert.append((id, vector))

        index.upsert(vectors=data_to_insert)

index.close()