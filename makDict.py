import re
import json

def convertToDict(input_string):
    pattern = r'\{(.*?)\}'

    matches = re.findall(pattern, input_string, re.DOTALL)

    qa_pairs = [{"Question": match[0], "Answer": match[1]} for match in matches]
    json_s = []
    for match in matches:
        clean_match = match.replace('\\"', '"').strip()
        result = f'{{{clean_match}}}'if clean_match else ''
        json_s.append(result)
    return json_s

