



import requests 


def get_answer(text,question):
  req_body={}
  req_body["question_texts"]=question.split('","')
  req_body["context_text"]=text
  host = "http://a89df97acfc3.ngrok.io"
  url = f"{host}/api"
  response = requests.post(url,json=req_body)
  print(response.text)
  return(response.text)