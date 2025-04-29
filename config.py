import os
from dotenv import load_dotenv

load_dotenv()  # automatically loads variables from a .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_MODEL = "gpt-4-turbo"  # safest option
