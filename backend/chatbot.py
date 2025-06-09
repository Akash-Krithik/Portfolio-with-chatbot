from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import urllib.parse
import re

app = Flask(__name__)
CORS(app)

llm = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype="auto",
    device_map="auto",
    max_new_tokens=16
)

def fetch_web_content(query):
    search_url = f"https://duckduckgo.com/html/?q={query.replace(' ', '+')}"
    try:
        resp = requests.get(search_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.find_all('a', {'class': 'result__a'}, href=True)
        if results:
            url = results[0]['href']
            parsed = urllib.parse.urlparse(url)
            uddg = urllib.parse.parse_qs(parsed.query).get('uddg')
            if uddg:
                url = uddg[0]
            elif url.startswith('//'):
                url = 'https:' + url
            elif url.startswith('/'):
                url = 'https://duckduckgo.com' + url
            page = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            page_soup = BeautifulSoup(page.text, "html.parser")
            for p in page_soup.find_all('p'):
                text = p.get_text().strip()
                if len(text) > 40:
                    return text[:300]
    except Exception as e:
        return f"Error fetching web content: {e}"
    return "No relevant web data found."

def extract_answer(generated_text):
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:", 1)[-1].strip()
    elif "Assistant:" in generated_text:
        answer = generated_text.split("Assistant:", 1)[-1].strip()
    else:
        answer = generated_text.strip()
    stop_match = re.search(r"(User:|Assistant:)", answer[1:])
    if stop_match:
        answer = answer[:stop_match.start()+1].strip()
    return answer

def rag_answer(question):
    greetings = ["hi", "hello", "hey"]
    if any(greet in question.lower() for greet in greetings):
        prompt = f"User: {question}\nAssistant:"
        response = llm(prompt)
        generated_text = response[0]['generated_text']
    else:
        context = fetch_web_content(question)
        prompt = (
            f"Use the following context to answer the question briefly.\n\n"
            f"Context: {context}\n"
            f"Question: {question}\nAnswer:"
        )
        response = llm(prompt)
        generated_text = response[0]['generated_text']
    return extract_answer(generated_text)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"response": "Please provide a message."}), 400
    answer = rag_answer(user_input)
    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
