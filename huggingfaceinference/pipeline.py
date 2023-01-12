from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def gramcorrector(text, gtokenizer=None, gmodel=None, n=None):
  
  if n == None:
    n = 1
  
  if gtokenizer == None:
    gtokenizer = AutoTokenizer.from_pretrained("vishnun/tinygram")
  if gmodel == None:
    gmodel = AutoModelForSeq2SeqLM.from_pretrained("vishnun/tinygram")
  
  input_ids = gtokenizer.encode(text, return_tensors='pt')
  outputs = gmodel.generate(
    input_ids,
    do_sample=True, 
    max_length=50,
    top_p=0.999,
    top_k=45,
    num_return_sequences=n
  )
  
  out_text = []
  for y in outputs:
    out_text.append(gtokenizer.decode(y, skip_special_tokens=True))
  
  return out_text
