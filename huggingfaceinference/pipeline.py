from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
