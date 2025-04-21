from fine_tune_pcs import fine_tuning
from unsloth import FastLanguageModel
import os
import warnings
import re,json
import getDataFromPdf as p2d

# Suppress all warnings
warnings.filterwarnings("ignore")

#condition for already trained
#fine_tune pcs
modelName = str(input("Enter the model name (optional): "))
if modelName == "":
   modelName = "fine_tuned_llama_updated_ai_model"

if os.path.isdir(modelName):
   #if model already trained it will get the trained model
   model_path=modelName
else:
   #Generating the chat dataset from pdf
   p2d()
   #model training
   model_path = fine_tuning(modelName=modelName)

#inference
max_seq_length = 2048 #seq length
dtype = None
load_in_4bit = True #reduce memory usage
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
def Gen_Json(input_query):
    global model
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    messages = [
        {"role": "user", "content": input_query},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(input_ids = inputs, max_new_tokens = 2000, use_cache = True,
                        temperature = 1.5, min_p = 0.1)
    answer_text = tokenizer.batch_decode(outputs)
    return answer_text

while True:
  Query = str(input("Enter your query : "))
  if Query.lower() =='exit':break
  response = Gen_Json(Query)
  modified_response = str(response).split("<|end_header_id|>")
  print(modified_response[3].replace("<|eot_id|>']",''))

  