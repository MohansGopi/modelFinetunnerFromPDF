import os
import pdfplumber
from generateQuestion import question
import re
import pandas as pd
from makDict import convertToDict
import json

folder_path = "/home/mohan/Documents/code/modeltrainer/rawdata"
for filename in os.listdir(folder_path):
    with pdfplumber.open(f"{folder_path}/{filename}") as pdf:
        setNo = 1
        for page_num, page in enumerate(pdf.pages):
            content = page.extract_text()
            questionWithAnswer = question(f"""Please generate a different questions and answers(not a one word answer) for this content : {content} in the following format:
                                       """+"""{Question: <Question Text>,
                                       answer: <answer Text>}
                                       Ensure that each question and answer pair is provided in the only exact json format above, with "Question:" followed by the question and "Answer:" followed by the answer. Provide at least 10 question-answer pairs on any topic.""") if content else 'null'
            if questionWithAnswer!='null':
                modified_response = str(questionWithAnswer).split("<|end_header_id|>")
                payload = str(modified_response[3]).replace('<|eot_id|>"]','')if '<|eot_id|>"]' in str(modified_response[3]) else str(modified_response[3]).replace("<|eot_id|>']",'')
                dataDict = convertToDict(payload)
                print(dataDict)
                #Create a folder for store a dataset
                base_dir = "datasets"
                train_dir = os.path.join(base_dir, "train")
                os.makedirs(train_dir, exist_ok=True)
                data_to_json = {'conversations': []}
                data_to_json['conversations'].append({'content': f"{str(dataDict['Question'])}", 'role': 'user'})
                data_to_json['conversations'].append({'content': f"{str(dataDict['answer'])}", 'role': 'assistant'})
                with open(f"{train_dir}/problem_{setNo}.json", 'w') as data_file:
                    json.dump(data_to_json, data_file)
                    setNo += 1
                print(f"File {filename} page {page_num + 1} processed and saved as JSON.")
            else:
                print(f"File {filename} page {page_num + 1} has no content.")
#         print(f"File {filename} processed and saved as JSON.")



