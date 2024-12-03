import subprocess
import argparse
import requests
import base64
import time
import json
import sys

# Note: This is assuming you have a server running that handles multimodal inputs, 
# combining both text and image for analysis.
# The 'start_ollama_server' function is used here just for illustration; 
# ideally, you should be running the server separately.

def start_ollama_server():
    try:
        subprocess.Popen(["ollama", "run", "llava"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Starting Ollama server with LLaVa...")
        time.sleep(5)  # Wait a bit for the server to start
    except FileNotFoundError:
        print("Error: Ollama is not installed or not in the PATH.")
        sys.exit(1)

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_image_and_tweet(image_path, tweet_text, stance_prompt, persuasion_prompt):
    url = "http://localhost:11434/api/generate"
    
    # Encode image to base64
    image_base64 = encode_image_to_base64(image_path)

    # Construct the full prompt including both stance and persuasiveness
    full_prompt = f"Image: {image_base64}\nTweet: {tweet_text}\nStance: {stance_prompt}\nPersuasiveness: {persuasion_prompt}"

    payload = {
        "model": "llava",
        "prompt": full_prompt,
        "images": [image_base64]
    }

    response = requests.post(url, json=payload)

    try:
        # Split the response text into separate lines
        response_lines = response.text.strip().split('\n')

        # Extract and concatenate the 'response' part from each line
        full_response = ''.join(json.loads(line)['response'] for line in response_lines if 'response' in json.loads(line))

        # Here, we expect the response to include both the stance and persuasiveness result
        return full_response
    except Exception as e:
        return f"Error: {e}"

def parse_arguments():
    parser = argparse.ArgumentParser(description='LLaVA Image and Tweet Analysis')
    parser.add_argument('-i', '--image', required=True, help='Path to the image file')
    parser.add_argument('-t', '--tweet', required=True, help='Text of the tweet')
    parser.add_argument('-s', '--stance_prompt', default="Does the tweet support or oppose gun control?", help='Prompt for stance classification')
    parser.add_argument('-p', '--persuasion_prompt', default="Does the image improve the persuasiveness of the tweet?", help='Prompt for persuasiveness classification')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    start_ollama_server()  # Start the server (can be omitted if already running)
    
    # Analyze both the image and tweet together
    result = analyze_image_and_tweet(args.image, args.tweet, args.stance_prompt, args.persuasion_prompt)

    print("Response:", result)