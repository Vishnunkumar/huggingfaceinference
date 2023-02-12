import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DistilBertForTokenClassification

class TinyGram:
  
  def __init__(self, gtokenizer=None, gmodel=None):
    self.gtokenizer = gtokenizer
    self.gmodel = gmodel
    
    if self.gtokenizer == None:
      self.gtokenizer = AutoTokenizer.from_pretrained("vishnun/tinygram")
    if self.gmodel == None:
      self.gmodel = AutoModelForSeq2SeqLM.from_pretrained("vishnun/tinygram")

  def gramcorrector(self, text, gtokenizer=None, gmodel=None, n=None):
    
    self.text = text
    self.n = n

    if self.n == None:
      self.n = 1
    
    input_ids = self.gtokenizer.encode(self.text, return_tensors='pt')
    outputs = self.gmodel.generate(
      input_ids,
      do_sample=True, 
      max_length=50,
      top_p=0.999,
      top_k=45,
      num_return_sequences=n
    )
    
    out_text = []
    for y in outputs:
      out_text.append(self.gtokenizer.decode(y, skip_special_tokens=True))
    
    return out_text

 
class KnowledgeGraph:
  
  def __init__(self, gtokenizer=None, gmodel=None):
    
    self.gtokenizer = gtokenizer
    self.gmodel = gmodel
    
    if self.gtokenizer == None:
      self.gtokenizer = AutoTokenizer.from_pretrained("vishnun/kg_model")
    if self.gmodel == None:
      self.gmodel = DistilBertForTokenClassification.from_pretrained("vishnun/kg_model")
  
  def get_graph(self, text, gtokenizer=None, gmodel=None):
    
    self.text = text
    
    inputs = self.gtokenizer(self.text, return_tensors="pt")
    tokens = self.gtokenizer.tokenize(self.text)
    
    with torch.no_grad():
      logits = self.gmodel(**inputs).logits
    
    predictions = torch.argmax(logits, dim=2)
    predicted_token_class = [self.gmodel.config.id2label[t.item()] for t in predictions[0][1:-1]]
    
    entities = []
    for label, text in zip(predicted_token_class, tokens):
      js_dict = {}
      js_dict[text] = label
      entities.append(js_dict)
    
    return entities
