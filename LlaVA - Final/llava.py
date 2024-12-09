import logging
import subprocess
import argparse
import requests
import base64
import time
import json
import sys
import pandas as pd

# LlaVA 1.6


# TODO:
    # - run llava without examples
    # - prompt with 1-10 examples then run
    # - reprompt with examples every ten data pieces

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
        time.sleep(5) 
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

def analyze_image(image_path, tweet, prompt):
    
    url = "http://localhost:11434/api/generate"
    image_base64 = encode_image_to_base64(image_path)

    # prompt = f"""
    # You are analyzing whether an image improves the persuasiveness of a tweet.

    # Tweet: "{tweet}"
    # Image: [Attached Image]

    # Question: Does the image improve the persuasiveness of the tweet? 
    # Respond strictly with: **NO** or **YES** """


    payload = {
        "model": "llava",
        "prompt": prompt,
        "tweet": [tweet],
        "images": [image_base64]
    }

    # print("Raw Response:", response.text)
    logging.debug(f"Payload: {json.dumps(payload, indent=4)}")
    logging.basicConfig(level=logging.DEBUG)
    #     return f"Error: {e}"
    try:
        response = requests.post(url, json=payload, timeout=600)
        response.raise_for_status()
        # print("Raw Response: ", response.text)
        # with open("rawresults.txt", "w") as f:
        #     print(response.text, file=f)

        # Parse and accumulate fragments
        raw_lines = response.text.strip().split('\n')
        complete_response = ""

        for line in raw_lines:
            try:
                parsed_line = json.loads(line)
                complete_response += parsed_line.get("response", "")
                if parsed_line.get("done"):
                    break
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping invalid JSON line: {line} - Error: {e}")

        return complete_response.strip() if complete_response else "No valid response found."

    except requests.exceptions.RequestException as e:
        logging.error(f"Request error: {e}")
        return f"Error: {e}"

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return f"Error: {e}" 


if __name__ == "__main__":

    results = []
    examples = get_example_gun_data()
    realdata = get_gun_data()
    #start_ollama_server()
  
    prompt = f"""
    You are analyzing whether an image improves the persuasiveness of a tweet.

    Tweet: [Attached Tweet]
    Image: [Attached Image]

    Question: Does the image improve the persuasiveness of the tweet? 
    Respond strictly with: **NO** or **YES** """

    examples = [tuple(example) for example in examples]
    prompt = prompt_with_examples(prompt,examples[:10])
    print(examples[:5])
    
    i=0

    for i in range(len(realdata)):
        print(i)
        imageeeeee = ("data/images/gun_control/" + str(realdata[i][0]))
        result = analyze_image(imageeeeee, realdata[i][1], prompt)
        print(" Response:", result)
        results.append(result)
    print(results)

    with open("results.txt", "w") as f:
        for b in range(len(results)):
            print(results[b], file=f)


