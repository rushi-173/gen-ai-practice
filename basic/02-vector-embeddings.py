from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

text = "Sachin Tendulkar is a famous cricketer and Ronaldinho is a famous footballer"

response = client.embeddings.create(
    input=text,
    model="text-embedding-3-small"
)

print("Vector Embeddings", response.data[0].embedding)