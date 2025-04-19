# Import necessary libraries
import os
import pandas as pd
import json

class Upload:
    def __init__(self,dataset_path):
        self.dataset_path= dataset_path
    # Create directory for the dataset

    def dataset_up(self):
        base_dir = "datasets"
        train_dir = os.path.join(base_dir, "train")
        os.makedirs(train_dir, exist_ok=True)

        # Read data from CSV
        dat_from_csv = pd.read_csv(self.dataset_path)

        
        # Convert CSV to JSON format
        for i in range(len(dat_from_csv['prompt'])):
            data_to_json = {'conversations': []}
            data_to_json['conversations'].append({'content': f"{str(dat_from_csv['prompt'][i])}", 'role': 'user'})
            jk = """""".join(dat_from_csv['json'][i])
            data_to_json['conversations'].append({'content': f"{jk}", 'role': 'assistant'})
            with open(f"{train_dir}/problem_{i}.json", 'w') as data_file:
                json.dump(data_to_json, data_file)

        return "Dataset prepared and saved to the directory: prompt_json_dataset."
