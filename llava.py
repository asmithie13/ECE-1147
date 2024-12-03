import subprocess
import argparse
import requests
import base64
import time
import json
import sys
import pandas as pd

# note: including the start server code in this script for demo purposes. You might want to seperately start the server so that you're not starting the server every time you make the call. 
def get_gun_data():
    examples = []
    print(examples)

    df = pd.read_csv('data\\gun_control_dev.csv')

    print(df.iloc[0,0])
    examples = []
    r=0
    c=0
    for r in range(100):
        newtuple = []
        for c in range(5):
            if c==3:
                pass
            elif c == 1:
                pass
            elif c == 0:
                str1 = str(df.iloc[r,c]) + ".jpg"
                newtuple.append(str1)
            else:
                newtuple.append(df.iloc[r,c])
        examples.append(newtuple)
    return examples

def get_example_gun_data():
    examples = []
    print(examples)

    df = pd.read_csv('data\\gun_control_train.csv')

    print(df.iloc[0,0])
    examples = []
    r=0
    c=0
    for r in range(891):
        newtuple = []
        for c in range(5):
            if c==3:
                pass
            elif c == 1:
                pass
            elif c == 0:
                str1 = str(df.iloc[r,c]) + ".jpg"
                newtuple.append(str1)
            else:
                newtuple.append(df.iloc[r,c])
        examples.append(newtuple)
    return examples

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
    
def prompt_with_examples(prompt, examples=[]):
    """
    Parameters:
    - prompt (str): The main prompt to be processed by the language model.
    - examples (list of tuples): A list where each tuple contains a pair of strings 
      (example_prompt, example_response). Default is an empty list.

    Example usage:
    ```
    main_prompt = "Translate the following sentence into French:"
    example_pairs = [("Hello, how are you?", "Bonjour, comment ça va?"),
                     ("Thank you very much!", "Merci beaucoup!")]
    formatted_prompt = prompt_with_examples(main_prompt, example_pairs)
    print(formatted_prompt)
    ```
    """
    # Start with the initial part of the prompt
    full_prompt = "<s>[INST]\n"

    # Add each example to the prompt
    for image, tweet, persuasiveness in examples:
        full_prompt += f"""
        **Image:** {image}
        **Tweet:** {tweet}
        **Persuasiveness:** {persuasiveness}
        [/INST]
        """

    # Add the main prompt and close the template
    full_prompt += f"{prompt} [/INST]"

    return full_prompt

def analyze_image(image_path, tweet, custom_prompt=None):
    url = "http://localhost:11434/api/generate"
    image_base64 = encode_image_to_base64(image_path)

    prompt = f"""
    You are analyzing whether an image improves the persuasiveness of a tweet by providing context or propaganda.

    Tweet: "{tweet}"
    Image: [Attached Image]

    Question: Does the image improve the persuasiveness of the tweet? 
    Respond strictly with: **Yes** or **No** """

    payload = {
        "model": "llava",
        "prompt": custom_prompt,
        "images": [image_base64]
    }

    response = requests.post(url, json=payload)
    #print("Raw Response:", response.text)

    try:
        response_lines = response.text.strip().split('\n')
        full_response = ''.join(json.loads(line)['response'] for line in response_lines if 'response' in json.loads(line))

        return full_response
    except Exception as e:
        return f"Error: {e}"

def parse_arguments():
    parser = argparse.ArgumentParser(description='LLaVA Image Analysis')
    parser.add_argument('-i', '--image', required=True, help='Path to the image file')
    parser.add_argument('-p', '--prompt', default='Does this image improve the persuasiveness of the tweet? Answer yes or no', help='Custom prompt for image analysis')
    return parser.parse_args()

if __name__ == "__main__":
    #args = parse_arguments()

    examples = get_example_gun_data()
    realdata = get_gun_data()
    #start_ollama_server()
  
    prompt = "Does the image improve the persuasiveness of the tweet? Answer 'yes' or 'no'."


    i=65
    for i in range(len(realdata)):
        prompt_with_examples(prompt,examples)
        print(realdata[i][1])
        print("data/images/gun_control/" + str(realdata[i][0]))
        result = analyze_image("data/images/gun_control/" + '1369651557285982214.jpg', realdata[i][1], "Does this image improve the persuasiveness of the tweet? Answer 'yes' or 'no'")
        print(" Response:", result)

