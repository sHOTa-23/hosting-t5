import requests

def generate_text(prompt: str):
    url = "http://localhost:8000/generate"
    response = requests.post(url, json={"prompt": prompt})
    return response.json()

if __name__ == "__main__":
    prompt = "Translate from English to French: Happy Birthday"
    result = generate_text(prompt)
    print(result)
