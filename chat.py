from openai import OpenAI

def get_api_key():
    with open('openai_key.txt', 'r') as f:
        return f.read().strip()

client = OpenAI(api_key=get_api_key())
messages = []

while True:
    user_prompt = input("Prompt: ")
    messages.append({'role': 'user', 'content': user_prompt})

    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages
    )
    assistant_response = response.choices[0].message.content
    print(assistant_response)
    messages.append({'role': 'assistant', 'content': assistant_response})