import os

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv(".env")
load_dotenv(".env.shared")

index_name = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

pc.delete_index(index_name)