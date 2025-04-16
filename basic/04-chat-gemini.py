import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

response = client.models.generate_content(
    model='gemini-2.0-flash-001', contents='Name few famous cricketers in India'
)

print(response.text)