import os
from groq import Groq
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# 1. Ask the user for the language
target_lang = input("Enter the target language (e.g., Spanish, Hindi, French): ")

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text

file_path = "C:/Users/niles/Desktop/genai_pdf_chat/Shekhawati mission 100 Biology 12th 2025-26 (1).pdf"
document_text = extract_text_from_pdf(file_path)

# 2. Strict instruction to prevent the LLM from "talking back"
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system", 
            "content": f"You are a professional translator. Your task is to translate all provided text into {target_lang}. Do not ask follow-up questions. Do not provide explanations. Output ONLY the translated text."
        },
        {
            "role": "user", 
            "content": f"Document to translate into {target_lang}:\n\n{document_text}"
        }
    ],
    temperature=0.1  # Low temperature makes it more focused on the task
)

print(f"\n--- Result ({target_lang}) ---\n")
print(response.choices[0].message.content)
