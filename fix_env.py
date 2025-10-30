"""Check and fix .env file"""
import os
from pathlib import Path
from dotenv import load_dotenv

env_file = Path(__file__).parent / '.env'
print(f"Checking .env file: {env_file.absolute()}")

if env_file.exists():
    print("OK - .env file found")
    load_dotenv(env_file)
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if openai_key:
        print(f"OK - OPENAI_API_KEY loaded (starts with: {openai_key[:10]}...)")
    else:
        print("ERROR - OPENAI_API_KEY NOT LOADED!")
        print("Check .env file contents")

    if gemini_key:
        print(f"OK - GEMINI_API_KEY loaded")
    else:
        print("INFO - GEMINI_API_KEY not found (optional)")
else:
    print("ERROR - .env file NOT FOUND!")
    print(f"Creating new .env file...")

    # Create .env with template (without real keys!)
    env_content = """OPENAI_API_KEY=your-openai-api-key-here
GEMINI_API_KEY=your-gemini-api-key-here
USE_GEMINI=false
PORT=8000
"""
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write(env_content)
    print(f"OK - .env file created: {env_file}")

    # Check again
    load_dotenv(env_file)
    if os.getenv("OPENAI_API_KEY"):
        print("OK - Keys successfully written and loaded!")

