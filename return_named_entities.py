import spacy

def return_named_entities(model_name,text):
  '''
  prints the named entity with it's label and also returns the same object.
  '''
  nlp2 = spacy.load(model_name)
  doc2 = nlp2(text)
  out = []
  for ent in doc2.ents:
    print(ent.label_, ent.text)
    out.append((ent.label_, ent.text))
  return out

# return_named_entities('NLP Training','a four star restaurant with a bar')