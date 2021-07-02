import spacy
import random
from spacy.util import minibatch, compounding
import pickle

def train_model_from_scratch(model_name,train_data_path,label_path,n_iter=30):
  with open(train_data_path,'rb') as f:
    train_data = pickle.load(f)

  with open(label_path,'rb') as f:
    label = pickle.load(f)

  nlp = spacy.blank('en')
  ner = nlp.create_pipe('ner')
  nlp.add_pipe(ner)
  for i in label:
    ner.add_label(i)
  optimizer = nlp.begin_training()
  other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
  with nlp.disable_pipes(*other_pipes):  # only train NER
      for iteration in range(n_iter):
          random.shuffle(train_data)
          losses = {}
          batches = minibatch(train_data, size=compounding(4., 32., 1.001))
          for batch in batches:
              texts, annotations = zip(*batch)
              nlp.update(texts, annotations, sgd=optimizer, drop=0.35,
                          losses=losses)
          print('Loss:', losses)
  print('Saving Model to Disk')
  nlp.to_disk(model_name)

# train_model_from_scratch('NLP Training','train_data.pkl','label.pkl',30)