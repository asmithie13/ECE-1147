import subprocess
import argparse
import requests
import base64
import time
import json
import sys
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

def prompt_with_abortion_data():
    examples = []

    df = pd.read_csv('data\\abortion_train.csv')
    r=0
    c=0
    for r in range(891):
        newtuple = []
        for c in range(2):
            if c == 0:
                str1 = str(df.iloc[r,c]) + ".jpg"
                newtuple.append(str1)
            else:
                newtuple.append(df.iloc[r,c])
        examples.append(newtuple)
    return examples

# data does not include persusiven
def get_example_abortion_data():
    examples = []

    df = pd.read_csv('data\\abortion_train.csv')

    r=0
    c=0
    for r in range(891):
        newtuple = []
        for c in range(4):
            if c == 0:
                str1 = str(df.iloc[r,c]) + ".jpg"
                newtuple.append(str1)
            else:
                newtuple.append(df.iloc[r,c])
        examples.append(newtuple)
    return examples

# generates touples for the gun control training data
def get_example_gun_data():
    examples = []

    df = pd.read_csv('data\\gun_control_train.csv')

    r=0
    c=0
    for r in range(923):
        newtuple = []
        for c in range(5):
            if c==1:
                pass
            elif c == 0:
                str1 = str(df.iloc[r,c]) + ".jpg"
                newtuple.append(str1)
            else:
                newtuple.append(df.iloc[r,c])
        examples.append(newtuple)
    return examples


# note: including the start server code in this script for demo purposes.
# You might want to seperately start the server so that you're not starting the server 
# every time you make the call. 
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
    
def prepare_training_data_from_tuples(dataset):
    """
    Prepares training data from a tuple of tuples.
    Each inner tuple is unpacked into:
    - Image URL
    - Tweet
    - Stance
    - Persuasiveness
    Returns:
    - List of dictionaries with `prompt` and `image` keys.
    """
    examples = []
    for image_url, tweet, stance, persuasiveness in dataset:
        # Format the prompt to ask the model for stance and persuasiveness
        prompt = (
            f"<|image|> {image_url} \n"
            f"Tweet: {tweet} \n"
            "Please determine the following: \n"
            f"Stance: {stance} \n"
            f"Persuasiveness: {persuasiveness} \n"
            "</|image|>"
        )
        
        examples.append({"prompt": prompt, "image": image_url})
    
    return examples

def prompt_with_examples(prompt, examples=[]):
    # Start with the initial part of the prompt
    full_prompt = "<s>[INST]\n"

    # Add each example to the prompt
    for example_prompt, example_response in examples:
        full_prompt += f"{example_prompt} [/INST] {example_response} </s><s>[INST]"

    # Add the main prompt and close the template
    full_prompt += f"{prompt} [/INST]"

    return full_prompt


def analyze_image(image_path, custom_prompt):
    url = "http://localhost:11434/api/generate"
    image_base64 = encode_image_to_base64(image_path)

    payload = {
        "model": "llava",
        "prompt": custom_prompt,
        "images": [image_base64]
    }
    print("Payload:", payload)
    print("image 1")
   # print("Payload sent to server:", json.dumps(payload, indent=2))
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        print("Response:", response.text)
    except requests.exceptions.Timeout:
        print("Request timed out.")
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    

    try:
        print("image 2")
        # Split the response text into separate lines
        response_lines = response.text.strip().split('\n')
        print("image 3")
        # Extract and concatenate the 'response' part from each line
        full_response = ''.join(json.loads(line)['response'] for line in response_lines if 'response' in json.loads(line))
        print("image 4")
        return full_response
        
    except Exception as e:
        return f"Error: {e}"

def parse_arguments():
    parser = argparse.ArgumentParser(description='LLaVA Image Analysis')
    parser.add_argument('-i', '--image', required=True, help='Path to the image file')
    parser.add_argument('-p', '--prompt', default='Describe this image in detail', help='Custom prompt for image analysis')
    return parser.parse_args()

# python ex.py -i Comp2.png -p "is this a hot dog?"  
if __name__ == "__main__":
    gun_examples = get_example_gun_data()
    print(gun_examples)

    #args = parse_arguments()
    prepare_training_data_from_tuples(gun_examples)
    prompt = "What is the stance of this tweet and does the image make the tweet more persuasive?"
    image = "1324087921641721856.jpg"
    tweet = "Congratulations @ForHD65 on your victory! Weâ€™re proud to endorse a strong advocate who believes in our movement for gun safety, in the efficacy of universal background checks, and keeping the families of Texasâ€™ #HD65 safe from gun violence. https://t.co/LfIzR6iPA3"
    test_prompt = f"<|image|>{image} [/INST] Tweet: {tweet} {prompt} [/INST]"

    #start_ollama_server()
    print("main here 1")
    full_prompt = prompt_with_examples(test_prompt,gun_examples)
    print("main here 2")
    result = analyze_image("data/images/gun_control/"+image, full_prompt)
    print("main here 3")
    print(" Response:", result)


    #data/images/gun_control/1375923983225290753.jpg