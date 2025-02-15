import requests

def generate_text(prompt: str):
    url = "http://localhost:8001/generateBatch"
    response = requests.post(url, json={"prompts": prompt})
    return response.json()

if __name__ == "__main__":
    prompts = ["Translate English to French: Hello?", "Summarize: The quick brown fox jumps over the lazy dog.", "Summarize: The quick brown fox jumps over the lazy dog.", "Summarize: The quick brown fox dsdasdas over the lazy dog.", "Summarize: dasdasd quick brown fox czxc over the lazy dog."] + [f"What is {i} times 2" for i in range(500)]
    result = generate_text(prompts)
    print(result)
