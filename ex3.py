import subprocess
import argparse
import requests
import base64
import time
import json
import sys

# Function to start Ollama server (optional)
def start_ollama_server():
    try:
        subprocess.Popen(["ollama", "run", "llava"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Starting Ollama server with LLaVa...")
        time.sleep(5)  # Wait a bit for the server to start
    except FileNotFoundError:
        print("Error: Ollama is not installed or not in the PATH.")
        sys.exit(1)

# Function to encode image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to analyze image with custom prompt
def analyze_image(image_path, tweet, stance, persuasiveness):
    url = "http://localhost:11434/api/generate"
    image_base64 = encode_image_to_base64(image_path)

    prompt = f"""
    Tweet: {tweet}

    Stance (labeled): {stance}
    Persuasiveness (labeled): {persuasiveness}

    Prompt: Does this tweet support or oppose gun control? Does the image improve the persuasiveness of the tweet?
    """

    payload = {
        "model": "llava",
        "prompt": prompt,
        "images": [image_base64]
    }

    response = requests.post(url, json=payload)

    try:
        # Extract the predicted stance and persuasiveness
        response_json = json.loads(response.text)
        predicted_stance = response_json["choices"][0]["text"].split(": ")[-1].strip()
        predicted_persuasiveness = response_json["choices"][0]["text"].split(": ")[-2].strip()
        return f"Predicted Stance: {predicted_stance}\nPredicted Persuasiveness: {predicted_persuasiveness}"
    except Exception as e:
        return f"Error: {e}"

# Function to parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='LLaVA Image Analysis for Gun Control')
    parser.add_argument('-i', '--image', required=True, help='Path to the image file')
    parser.add_argument('-t', '--tweet', required=True, help='The tweet text')
    parser.add_argument('-s', '--stance', required=True, choices=["Support", "Oppose"], help='Labeled stance of the tweet (Support or Oppose)')
    parser.add_argument('-p', '--persuasiveness', required=True, choices=["Low", "High"], help='Labeled persuasiveness of the tweet (Low or High)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Start Ollama server (optional, uncomment if needed)
    # start_ollama_server()

    result = analyze_image(args.image, args.tweet, args.stance, args.persuasiveness)
    print(result)