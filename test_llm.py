import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

print("Chat started, (type exit to quit)")



while True:
    user = input("You: ")

    if user.lower() in ['exit']:
        print("Goodbye!")
        break


    response = client.chat.completions.create(
        messages=[{"role": "user", "content": user}],
        model="llama-3.3-70b-versatile",
    )


    print("AI: ", response.choices[0].message.content)