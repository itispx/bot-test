import os
import time

import uuid
import PyPDF2
from dotenv import load_dotenv
import pinecone
from langchain_openai import OpenAIEmbeddings

load_dotenv(".env")
load_dotenv(".env.shared")

# init pinecone
index_name = os.getenv("PINECONE_INDEX_NAME")

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

def read_and_embbed(file_path: str):
    path_parts = file_path.split(os.sep)

    directory_name = path_parts[-2]

    reader = PyPDF2.PdfReader(file_path)

    for page in reader.pages:
        page_text = page.extract_text()
        split_text = page_text.replace("\n", "").split()

        batched_text = []

        batch_size = 50

        for i in range(0, len(split_text), batch_size):
            batched_text.append(split_text[i:i+batch_size])

        for text_array in batched_text:
            text = ' '.join(text_array)
            id = str(uuid.uuid4())
            vector = embed_model.embed_documents(text)
            file_name = os.path.basename(file_path)
            metadata = {
                "text": text, 
                "directory": directory_name,
                "file_name": file_name
            }

            print(file_name)

            index.upsert(vectors=[(id, vector[0], metadata)])
        
def read_folder(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            read_and_embbed(file_path)
        elif os.path.isdir(file_path):
            read_folder(file_path)

directory = "knowledge-base"

read_folder(directory)

index.close()
