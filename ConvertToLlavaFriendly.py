import csv
import pandas as pd

# # Open the CSV file
# with open('data\\abortion_dev.csv', mode='r', newline='', encoding='utf-8') as file:
#     csv_reader = csv.reader(file)
    
#     # Iterate over rows in the CSV file
#     for column in csv_reader:
#         for row in csv_reader:
#             print(column,row)  # Each row is a list of values from the CSV

# Read the CSV file into a DataFrame

# generates touples for the gun control training data
# does not include stance or persuasiveness
def get_example_abortion_data():
    examples = []
    print(examples)

    df = pd.read_csv('data\\abortion_train.csv')

    print(df.iloc[0,0])
    r=0
    c=0
    for r in range(891):
        newtuple = []
        for c in range(4):
            if c==1:
                pass
            if c == 0:
                str1 = str(df.iloc[r,c]) + ".jpg"
                newtuple.append(str1)
            else:
                newtuple.append(df.iloc[r,c])
        examples.append(newtuple)
    return examples

def prompt_with_abortion_data():
    examples = []
    print(examples)

    df = pd.read_csv('data\\abortion_train.csv')

    print(df.iloc[0,0])
    r=0
    c=0
    for r in range(891):
        newtuple = []
        for c in range(2):
            if c==1:
                pass
            if c == 0:
                str1 = str(df.iloc[r,c]) + ".jpg"
                newtuple.append(str1)
            else:
                newtuple.append(df.iloc[r,c])
        examples.append(newtuple)
    return examples

# generates touples for the gun control training data
# does not include stance or persuasiveness
def get_example_gun_data():
    examples = []
    print(examples)

    df = pd.read_csv('data\\gun_control_train.csv')

    print(df.iloc[0,0])
    r=0
    c=0
    for r in range(891):
        newtuple = []
        for c in range(5):
            if c==3:
                pass
            elif c==1:
                pass
            elif c == 0:
                str1 = str(df.iloc[r,c]) + ".jpg"
                newtuple.append(str1)
            else:
                newtuple.append(df.iloc[r,c])
        examples.append(newtuple)
    return examples

ex = (get_example_gun_data())

c = 0
for i in ex:
    print(ex[c][1])
    print(ex[c][2])
    print("data/images/gun_control/" + str(ex[c][0]))
    c+=1
print(c)


