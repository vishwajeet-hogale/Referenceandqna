import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import time
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')
def generate_summary(text):
    text = text.strip().replace("\n","")
    parts = text.split()
    new_list = []
    prev = 0
    str1 = ""
    for i in parts:
        if(prev == 512):
            prev = 0
            # print(str1)
            new_list.append(str1)
            str1 = ""
        str1 += " "+i
        prev += 1
    if(str1 != ""):
        # print(str1)
        new_list.append(str1)
    answer = []
    print(len(new_list))
    for j in new_list:
        print(len(j.split()))
        t5_prepared_Text = "summarize: "+ j
        # print ("original text preprocessed: \n", j)

        tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


        # summmarize 
        summary_ids = model.generate(tokenized_text,
                                            num_beams=4,
                                            no_repeat_ngram_size=2,
                                            min_length=70,
                                            max_length= 100,
                                            early_stopping=True)

        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        answer.append(output)
        time.sleep(2)
    print(answer)
    return "".join(answer)