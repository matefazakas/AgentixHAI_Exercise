import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], # https://api.groq.com/openai/v1
api_key=os.environ["OPENAI_API_KEY"]) # dein Groq-Schlüssel
reply = client.chat.completions.create(model=os.environ["MODEL"], # openai/gpt-oss-20b
messages=[{"role": "user", "content": "How's the weather today in Ulm?"}])
print(reply.choices[0].message.content)
