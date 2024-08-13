import os
import time

import uuid
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv(".env")

index_name = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec = ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    # wait for index to finish initialization
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

index = pc.Index(index_name)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def estimate_token_count(text: str) -> int:
    words = text.split()
    return int(len(words) * 1.5)  # Rough estimation: 1.5 tokens per word

def embed_and_upload(file_path: str, text: str):
    try:
        split_text = text.split()
        batched_text = []
        batch_text_size = 500  # Adjusted to be safer within the token limit

        directory_name = file_path.split(os.sep)[-2]
        file_name = os.path.basename(file_path)

        for i in range(0, len(split_text), batch_text_size):
            text_batch = split_text[i:i+batch_text_size]

            if estimate_token_count(' '.join(text_batch)) > 8000:  # Ensure we stay under the limit
                batch_text_size = int(batch_text_size * 0.8)  # Reduce batch size and retry
                continue
            
            batched_text.append(text_batch)

        for text_array in batched_text:
            text = ' '.join(text_array)

            embedding_response = client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            vector = embedding_response.data[0].embedding  # Accessing the embedding correctly
            id = str(uuid.uuid4())
            metadata = {
                "text": text, 
                "directory": directory_name,
                "file_name": file_name
            }

            # Upload the embedding to Pinecone
            index.upsert(vectors=[(id, vector, metadata)])

            print(f"uploaded embedding for file: {file_name}, batch size: {len(text_array)} words")
    except Exception:
        pass


def from_text(file_path: str):
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

        text = "".join(lines)
        text = text.replace("\n", "")

        return text

def extract_text(file_path: str):
    file_name = os.path.basename(file_path)
    extension = os.path.splitext(file_name)[1]

    text = ""

    match extension:
        case ".txt":
            text = from_text(file_path)
        case _:
            print("Invalid extension: ", extension)

    embed_and_upload(file_path, text)

def read_folder(directory: str):

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            extract_text(file_path)
        elif os.path.isdir(file_path):
            read_folder(file_path)

directory = "knowledge-base"

read_folder(directory)
