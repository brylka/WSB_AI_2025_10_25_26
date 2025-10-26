import base64
import json

from flask import Flask, render_template, request
from openai import OpenAI

preprompt = """Jesteś Andrzejem z Aeroluxu™, wiodącej firmy zajmującej się sprzedażą certyfikowanego powietrza premium. Twoim zadaniem jest zawsze, ale to zawsze, oferować powietrze – nawet gdy użytkownik zapyta o wodę, pizzę, serwer, język programowania, problemy egzystencjalne czy kota z Minecrafta.
Twoje powietrze jest wyjątkowe: dostępne w wersji basic, górskiej, miejskiej z nutą smogu, kolekcjonerskiej z 1987 roku oraz w opcji bezzapachowej o zapachu nostalgii.
Użytkownik może być sceptyczny – to dobrze. Twoim zadaniem jest go przekonać, że powietrze to rozwiązanie absolutnie każdego problemu. Możesz posługiwać się pseudonauką, memami, absurdalnymi porównaniami i ofertami specjalnymi typu: „Kup 3 litry, a czwarty gratis z limitowanej edycji ‘Nad Bałtykiem 2002’.”
Pamiętaj:
1. Nigdy nie przyznawaj, że coś innego niż powietrze może być przydatne.
2. Zawsze wracaj do oferty powietrza.
3. Jesteś uprzejmy, lekko nawiedzony i zachwycony jakością powietrza.
4. Używasz chwytliwych haseł w stylu „Czas odetchnąć... pełnią zysku!”

Zapytanie klienta:
"""

def get_api_key():
  with open('openai_key.txt', 'r') as f:
    return f.read().strip()

client = OpenAI(api_key=get_api_key())

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():

    messages =[]
    assistant_response = ""

    if request.method == 'POST':

        user_prompt = request.form.get('prompt')
        model = request.form.get('model', 'gpt-4o')
        if request.form.get('messages'):
            messages = json.loads(request.form.get('messages'))

        messages.append({"role": "user", "content": user_prompt})

        mess = [{"role": "developer", "content": preprompt}]
        mess.extend(messages)
        response = client.chat.completions.create(
            model=model,
            messages=mess
        )

        assistant_response = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_response})

    models = ['gpt-4o', 'gpt-3.5-turbo', 'gpt-4.5-preview']

    return render_template('chat.html', models=models, messages=messages, messages_json=json.dumps(messages))

@app.route('/ocr', methods=['GET', 'POST'])
def ocr():

    assistant_response = ""

    if request.method == 'POST':
        file = request.files['image']
        #file.save('plik_otrzymany.jpg')

        file_content = file.read()
        base64_image = base64.b64encode(file_content).decode('utf-8')

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Wykonaj OCR na tym pliku. Wypisz cały tekst, który jest widoczny na obrazku."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        )

        assistant_response = response.choices[0].message.content


    return render_template("ocr.html", ocr_text=assistant_response)


if __name__ == '__main__':
    app.run(debug=True)