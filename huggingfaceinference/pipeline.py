from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

gtokenizer = AutoTokenizer.from_pretrained("vishnun/tinygram")
gmodel = AutoModelForSeq2SeqLM.from_pretrained("vishnun/tinygram")

def gramcorrector(gtokenizer, gmodel):
