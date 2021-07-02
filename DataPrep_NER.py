import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import pickle

##############Read and Save Data###############
url = "https://groups.csail.mit.edu/sls/downloads/restaurant/restauranttrain.bio"    
html = requests.get(url)
html_page = html.content

text = BeautifulSoup(html_page, 'html.parser')
with open('data.txt','w') as f:
  f.write(text.text)

############Load Data and create sentences#############
'''
If a line is empty i.e., it has only '\n' in it 
then it is the seperator between 2 sentences.
'''
with open('data.txt','r') as f:
  sentences = []
  sentence = []
  for line in f:
    if line != '\n':
      sentence.append(line)
    else:
      sentences.append(''.join(sentence))
      sentence = []

#################Transform data into the format that can be used by spacy############ 
train_data = []
label = []
for sentence in sentences:
  res = pd.read_csv(StringIO(sentence),delimiter='\t',header=None)
  res.columns = ['Tag','Word']
  res['Word_Length'] = res['Word'].apply(lambda x: len(str(x)))
  res['Word_Start'] = ((res['Word_Length']+1).cumsum()).shift(1,fill_value=0)
  res['Word_End'] = res['Word_Start'] + res['Word_Length']
  words = res['Word'].values.tolist()
  if len(words)==1:
    text = str(words[0])
  else:
    text = ' '.join(words)
  list_entities = []
  for i in range(res.shape[0]):
    tag = res.loc[i,'Tag']
    start =  res.loc[i,'Word_Start']
    end =  res.loc[i,'Word_End']
    if tag not in label:
      label.append(tag)
    if tag != 'O':
      list_entities.append((start,end,tag))
  train_data.append((text,{'entities':list_entities}))

#Save train_data and labels
with open("train_data.pkl", "wb") as output_file:
  pickle.dump(train_data, output_file)

with open("label.pkl", "wb") as output_file:
  pickle.dump(label, output_file)