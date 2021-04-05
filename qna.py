# import os
# import torch
# import time
# from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# from transformers import (
#     AlbertConfig,
#     AlbertForQuestionAnswering,
#     AlbertTokenizer,
#     squad_convert_examples_to_features
# )

# from transformers.data.processors.squad import SquadResult, SquadV2Processor, SquadExample

# from transformers.data.metrics.squad_metrics import compute_predictions_logits

# # READER NOTE: Set this flag to use own model, or use pretrained model in the Hugging Face repository
# use_own_model = False

# if use_own_model:
#   model_name_or_path = "/content/model_output"
# else:
#   model_name_or_path = "ktrapeznikov/albert-xlarge-v2-squad-v2"

# output_dir = ""

# # Config
# n_best_size = 1
# max_answer_length = 30
# do_lower_case = True
# null_score_diff_threshold = 0.0

# def to_list(tensor):
#     return tensor.detach().cpu().tolist()
#     # print(tensor)
#     # return tensor.tolist()

# # Setup model
# # config_class, model_class, tokenizer_class = (
# #     AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer)
# # config = config_class.from_pretrained(model_name_or_path)
# # tokenizer = tokenizer_class.from_pretrained(
# #     model_name_or_path, do_lower_case=True)
# # model = model_class.from_pretrained(model_name_or_path, config=config)

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # model.to(device)

# # processor = SquadV2Processor()

# def run_prediction(question_texts, context_text):
#     """Setup function to compute predictions"""
#     examples = []

#     for i, question_text in enumerate(question_texts):
#         example = SquadExample(
#             qas_id=str(i),
#             question_text=question_text,
#             context_text=context_text,
#             answer_text=None,
#             start_position_character=None,
#             title="Predict",
#             is_impossible=False,
#             answers=None,
#         )

#         examples.append(example)

#     features, dataset = squad_convert_examples_to_features(
#         examples=examples,
#         tokenizer=tokenizer,
#         max_seq_length=384,
#         doc_stride=128,
#         max_query_length=64,
#         is_training=False,
#         return_dataset="pt",
#         threads=1,
#     )

#     eval_sampler = SequentialSampler(dataset)
#     eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=10)

#     all_results = []

#     for batch in eval_dataloader:
#         model.eval()
#         batch = tuple(t.to(device) for t in batch)

#         with torch.no_grad():
#             inputs = {
#                 "input_ids": batch[0],
#                 "attention_mask": batch[1],
#                 "token_type_ids": batch[2],
#             }

#             example_indices = batch[3]
            
#             outputs1 = model(**inputs)
#             print(len([outputs1]))
#             for i, example_index in enumerate(example_indices):
#                 eval_feature = features[example_index.item()]
#                 unique_id = int(eval_feature.unique_id)
#                 # print([type(outputs1[i]) for output in outputs1])
#                 print([to_list(ot) for ot in outputs1])
#                 output1 = [to_list(ot[i]) for ot in [outputs1]]
#                 print(len(output1[0]))
#                 start_logits, end_logits = output1[0]
#                 result = SquadResult(unique_id, start_logits, end_logits)
#                 all_results.append(result)
#     print(all_results)
#     output_prediction_file = "predictions.json"
#     output_nbest_file = "nbest_predictions.json"
#     output_null_log_odds_file = "null_predictions.json"

#     predictions = compute_predictions_logits(
#         examples,
#         features,
#         all_results,
#         n_best_size,
#         max_answer_length,
#         do_lower_case,
#         output_prediction_file,
#         output_nbest_file,
#         output_null_log_odds_file,
#         False,  # verbose_logging
#         True,  # version_2_with_negative
#         null_score_diff_threshold,
#         tokenizer,
#     )

#     return predictions
# if __name__ == "__main__":
#     config_class, model_class, tokenizer_class = (
#     AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer)
#     config = config_class.from_pretrained(model_name_or_path)
#     tokenizer = tokenizer_class.from_pretrained(
#         model_name_or_path, do_lower_case=True)
#     model = model_class.from_pretrained(model_name_or_path, config=config)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model.to(device)
#     print(model)
#     processor = SquadV2Processor()
context = "Incorporated in 1999, MTAR Technologies is a leading national player in the precision engineering industry. The company is primarily engaged in the manufacturing of mission-critical precision components with close tolerance and in critical assemblies through its precision machining, assembly, specialized fabrication, testing, and quality control processes. Since its inception, MTAR Technologies has significantly expanded its product portfolio including critical assemblies i.e. Liquid propulsion engines to GSLV Mark III, Base Shroud Assembly & Airframes for Agni Programs, Actuators for LCA, power units for fuel cells, Fuel machining head, Bridge & Column, Drive Mechanisms, Thimble Package, etc. A wide range of complex product portfolios meets the varied requirements of the Indian nuclear, Defence, and Space sector. ISRO, NPCIL, DRDO, Bloom Energy, Rafael, Elbit, etc. are some of the esteem clients. Currently, the firm has 7 state-of-the-art manufacturing facilities in Hyderabad, Telangana that undertake precision machining, assembly, specialized fabrication, brazing and heat treatment, testing and quality control, and other specialized processes."
questions = ["which company is going to be listed?","which company is going to be IPO?","which company is going to be public?","which company is going to be listed on stock exchange?","which company is about to launch initial public offering?","which company is about to launch IPO?","which company is it talking about?","where is the company located?","in which country is it?"]

#     # Run method
#     predictions = run_prediction(questions, context)

#     # Print results
#     # Print results
#     import statistics as st
#     prelist = [i for i in predictions.values() if i != '']
#     try:
#         print(st.mode(prelist))
#     except:
#         print(prelist)
import requests 
req_body = {
  "question_texts": [
    "which company is going to be listed?","which company is going to be IPO?","which company is going to be public?","which company is going to be listed on stock exchange?","which company is about to launch initial public offering?","which company is about to launch IPO?","which company is it talking about?","where is the company located?","in which country is it?"
  ],
  "context_text": "Incorporated in 1999, MTAR Technologies is a leading national player in the precision engineering industry. The company is primarily engaged in the manufacturing of mission-critical precision components with close tolerance and in critical assemblies through its precision machining, assembly, specialized fabrication, testing, and quality control processes. Since its inception, MTAR Technologies has significantly expanded its product portfolio including critical assemblies i.e. Liquid propulsion engines to GSLV Mark III, Base Shroud Assembly & Airframes for Agni Programs, Actuators for LCA, power units for fuel cells, Fuel machining head, Bridge & Column, Drive Mechanisms, Thimble Package, etc. A wide range of complex product portfolios meets the varied requirements of the Indian nuclear, Defence, and Space sector. ISRO, NPCIL, DRDO, Bloom Energy, Rafael, Elbit, etc. are some of the esteem clients. Currently, the firm has 7 state-of-the-art manufacturing facilities in Hyderabad, Telangana that undertake precision machining, assembly, specialized fabrication, brazing and heat treatment, testing and quality control, and other specialized processes."
}
host = "http://8c56d61df2ba.ngrok.io"
url = f"{host}/api"
response = requests.post(url,json=req_body)
print(response.text)