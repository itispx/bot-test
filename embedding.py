import os
import time

import uuid
import PyPDF2
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, PodSpec
from bs4 import BeautifulSoup
import docx2txt

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
    api_key=os.getenv("OPENAI_API_KEY")
)

def read_and_embbed(file_path: str):
    path_parts = file_path.split(os.sep)

    directory_name = path_parts[-2]

    reader = PyPDF2.PdfReader(file_path)

    for page in reader.pages:
        page_text = page.extract_text()
        split_text = page_text.replace("\n", "").split()

        batched_text = []
        batch_text_size = 300

        for i in range(0, len(split_text), batch_text_size):
            batched_text.append(split_text[i:i+batch_text_size])

        for text_array in batched_text:
            text = ' '.join(text_array)

            embedding_response = client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            vector = embedding_response.data[0].embedding
            id = str(uuid.uuid4())
            file_name = os.path.basename(file_path)
            metadata = {
                "text": text, 
                "directory": directory_name,
                "file_name": file_name
            }

            index.upsert(vectors=[(id, vector, metadata)])

            print(file_name)


def embed_and_upload(file_path: str, text: str):
    split_text = text.split()
    batched_text = []
    batch_text_size = 600

    directory_name = file_path.split(os.sep)[-2]
    file_name = os.path.basename(file_path)

    for i in range(0, len(split_text), batch_text_size):
        batched_text.append(split_text[i:i+batch_text_size])
    
    for text_array in batched_text:
        text = ' '.join(text_array)

        embedding_response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        vector = embedding_response.data[0].embedding
        id = str(uuid.uuid4())
        metadata = {
            "text": text, 
            "directory": directory_name,
            "file_name": file_name
        }

        index.upsert(vectors=[(id, vector, metadata)])

        print(file_name)

def from_html(file_path: str):
    f = open(file_path, encoding="utf-8")     
    soup = BeautifulSoup(f, features="html.parser")

    body = soup.find("body")

    for script in body(["script", "style", "ul"]):
        script.extract()    # rip it out
    
    text = body.get_text().replace("\n", "")
    
    split = "deixe um comentário" # delete everything after this
    delete = ["FacebookTwitter", "Copyright © 2023. Desenvolvido por API.", "Baixe Nosso Aplicativo", "Siga a Imóveis Crédito Real", "Categorias"]

    text = text.split(split, 1)[0]

    for i in delete:
        text = text.replace(i, "")

    return text

def from_pdf(file_path: str):
    text = ""

    reader = PyPDF2.PdfReader(file_path)

    for page in reader.pages:
        text += page.extract_text().replace("\n", "")

    return text

def from_docx(file_path: str):
    with open(file_path, 'rb') as infile:
            text = docx2txt.process(infile)
            text = text.replace("\n", "")

            return text

def from_text(file_path: str):
    with open(file_path) as f:
        lines = f.readlines()

        text = "".join(lines)
        text = text.replace("\n", "")

        return text

def extract_text(file_path: str):
    file_name = os.path.basename(file_path)
    extension = os.path.splitext(file_name)[1]

    text = ""

    match extension:
        case ".html":
            text = from_html(file_path)
        case ".pdf":
            text = from_pdf(file_path)
        case ".docx":
            text = from_docx(file_path)
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
