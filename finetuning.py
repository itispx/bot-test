import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env")
load_dotenv(".env.shared")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

file = client.files.create(
    file=open("finetuning_prepared.jsonl", "rb"),
    purpose="fine-tune"
)

file_id = file.id

print("file_id",file_id)

client.fine_tuning.jobs.create(
  training_file=file_id, 
  model="davinci-002"
)