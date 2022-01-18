from __future__ import unicode_literals
from matplotlib import colors

from nltk.util import pr

# """ # commented for local testing
# For Legal Pythia

# @author: SURYA L RAMESH

# First Created on Thu May 27 17:28:25 2021

# """

# ''''''''''''''''''''' SECTION 1. IMPORT ALL REQUIREMENTS''''''''''''''''''''' # commented for local testing

# from __future__ import unicode_literals # moved to the top

import streamlit as st 
import pandas as pd
import os
import torch
import time 
import nltk
import spacy
import pdfplumber
import docx2txt
import matplotlib.pyplot as plt
import base64

from annotated_text import annotated_text

from torch.utils.data import Dataset, TensorDataset, DataLoader #SequentialSampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import AlbertTokenizer
from transformers import AlbertForSequenceClassification, AdamW


nlp = spacy.load('en_core_web_sm') # large needed for word vectors 

path = os.path.abspath(os.getcwd())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ("\n >>> device used: ",device)
print ("\n")


device_type = "cuda:0" if torch.cuda.is_available() else "cpu"



# ''''''''''''''''''''''''''' SECTION 2. THE MODEL ''''''''''''''''''''''''''''' # commented for local testing


model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=3)
model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

#  contains all of the hyperparemeter information for training loop 
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, correct_bias=False)

epoch_to_resume = 4
path_to_model_saved = 'model_epoch{}.pt'.format( epoch_to_resume)
if os.path.isfile(path_to_model_saved):
    print("\n >>> loading checkpoint '{}'".format(path_to_model_saved))

    checkpoint = torch.load(path_to_model_saved,map_location=torch.device('cpu'))
    savd_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("\n >>> loaded checkpoint '{}' (epoch {})"
          .format(path_to_model_saved, checkpoint['epoch']))
else:
    print("\n >>> no checkpoint found at '{}'".format(path_to_model_saved))
    
    
    
# ''''''''''''''''''''''' SECTION 3. THE FUNCTIONS ''''''''''''''''''''''''''''' # commented for local testing
       
def calculate_similarity_percentage(file1, file2):
    
    # spaCy has support for word vectors whereas NLTK does not
    
    text1 = nlp(file1)
    text2 = nlp(file2)   
             
    sim = text1.similarity(text2)
    return sim

class SNLIDataAlbertPredictor(Dataset):

  def __init__(self,input_df):
    self.label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    self.input_df = input_df

    self.base_path = '/content/'
    self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True)
    self.input_data = None
  
    self.init_data()

  def init_data(self):
    
    self.input_data = self.load_data(self.input_df)

  def load_data(self, df):
    token_ids = []
    mask_ids = []
    seg_ids = []

    premise_list = df['premise'].to_list()
    hypothesis_list = df['hypothesis'].to_list()
  

    for (premise, hypothesis) in zip(premise_list, hypothesis_list):
      premise_id = self.tokenizer.encode(premise, add_special_tokens = False)
      hypothesis_id = self.tokenizer.encode(hypothesis, add_special_tokens = False)
      pair_token_ids = [self.tokenizer.cls_token_id] + premise_id + [self.tokenizer.sep_token_id] + hypothesis_id + [self.tokenizer.sep_token_id]
      premise_len = len(premise_id)
      hypothesis_len = len(hypothesis_id)

      segment_ids = torch.tensor([0] * (premise_len + 2) + [1] * (hypothesis_len + 1))  # sentence 0 and sentence 1
      attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values

      token_ids.append(torch.tensor(pair_token_ids))
      seg_ids.append(segment_ids)
      mask_ids.append(attention_mask_ids)
    
    
    token_ids = pad_sequence(token_ids, batch_first=True)
    mask_ids = pad_sequence(mask_ids, batch_first=True)
    seg_ids = pad_sequence(seg_ids, batch_first=True)
   
    dataset = TensorDataset(token_ids, mask_ids, seg_ids)
    #print(len(dataset))
    return dataset

  def get_data_loaders(self, batch_size=32, shuffle=True):
    input_loader = DataLoader(
      self.input_data,
      shuffle=shuffle,
      batch_size=batch_size
    )
    return input_loader

# code for checking similarity and contradiction
     
def check_similarity_contradiction(sentence1, sentence2):
           
       data_input = {'premise':[sentence1], 'hypothesis':[sentence2]}
       df_input = pd.DataFrame(data_input, columns = ['premise','hypothesis'])
       
       input_dataset = SNLIDataAlbertPredictor(df_input)
       input_loader = input_dataset.get_data_loaders(batch_size=1)
       
       (pair_token_ids, mask_ids, seg_ids) = next(iter(input_loader))
       pair_token_ids = pair_token_ids.to(device)
       mask_ids = mask_ids.to(device)
       seg_ids = seg_ids.to(device)
       result = model(pair_token_ids, 
                                     token_type_ids=seg_ids, 
                                     attention_mask=mask_ids)
       prediction = result.logits #Predition in tensor Form
       softmax =torch.log_softmax(prediction, dim=1)
       pred =softmax.argmax(dim=1)
       
       target_map = {0: 'entailment',1:'contradiction',2:'neutral'}

       
       if device_type == "cpu":
          outcome = target_map[pred.data.cpu().numpy()[0]]
       else:
           outcome = target_map[pred[0]]  # modified to get value from tensor
       return  outcome 

def styler(col):
    # apply style to prediction column only 
    if col.name != 'prediction':
        return [''] * len(col)
    bg_map = []
    
    for x in col:
        
        if x[0] == 'contradiction' :
                bg_map.append ('background-color:LightCoral')
        elif x[0] == 'entailment' :
                bg_map.append( 'background-color:LightGreen')
        else:
                bg_map.append('')
    
        
    #print (bg_map)
    return bg_map

def pdf_to_text(file):
    pdf = pdfplumber.open(file)
    page = pdf.pages[0]
    text = page.extract_text()
    pdf.close()
    return text.encode('utf8')
                
def doc_to_text(file):
    text = docx2txt.process(file)
    text = text.replace('\n\n',' ')
    text = text.replace('  ',' ')
    return text.encode('utf8')

def visualise_ner(text):
    tokens = []
    doc=nlp(text)
    for token in doc:
        if (token.ent_type_ == "PERSON"):
            tokens.append((token.text, "PERSON", "#faa"))
        elif (token.ent_type_ == "LOC"):
            tokens.append((token.text, "LOC", "#fda"))
        elif (token.ent_type_ == "GPE"):
            tokens.append((token.text, "GPE", "#be2"))
        elif (token.ent_type_ == "ORG"):
            tokens.append((token.text, "ORG", "#0cf"))
        elif (token.ent_type_ == "DATE"):
            tokens.append((token.text, "DATE", "#fd1"))
        elif (token.ent_type_ == "MONEY"):
            tokens.append((token.text, "MONEY", "#f1d"))    
        else:
            tokens.append(" " + token.text + " ")

    return tokens

# def visualise_ner(file1, file2):
#     text1 = nlp(file1)
#     text2 = nlp(file2)

#     annotated_text(text1)
#     annotated_text(text2)

    




# ''''''''''''''''''''''''''' SECTION 4. THE MAIN APP CODE ''''''''''''''''''''' # commented for local testing
       
def main():
    header = st.container() # updated st.beta_container() to st.container()
    steps = st.container() # updated st.beta_container() to st.container()
    userinputfiles = st.container() # updated st.beta_container() to st.container()
    userchoice = st.container() # updated st.beta_container() to st.container()
 
    # -- Default selector list
    selector_list = ['Similarity %','Similarity and Contradition detection', 'Visualise Entities']

    with header:
        # st.image('/Users/gayanin/RA-Work/Legal Pythia/LegalPythia-V2/codes/res/legalpythiaheader.jpg')
        st.image('codes/res/legalpythiaheader.jpg')
        st.title(' Welcome to the Live Demo!')
        st.text(' Here you get to upload two text files and check for similarity or contradiction')

    # with steps:
    #     st.subheader('The Three Step Process:')
        
    #     st.markdown ('* ** Step 1:** Load document 1' )    
    #     st.markdown ('* ** Step 2:** Load document 2' )
    #     st.markdown ('* ** Step 3:** Choose Similarity %  or Similarity and Contradiction Detection or Visualisation')
        
    
    # with userchoice:
        # userchoice = st.radio("Choose your comparison method",('Similarity % ','Similarity and Contradition detection', 'Visualise Entities'))
        # if userchoice == 'Similarity % ':
        #         st.write('You have selected Similarity.')
        # if userchoice == 'Similarity and Contradition detection':
        #     st.write('You have selected Similarity and Contradition detection.')
        # if userchoice == 'Visualise Entities':
        #         st.write('\n You have selected Visualisation.')
        

        # selector = st.sidebar.selectbox('Selector', selector_list)

    with userinputfiles and userchoice:
       
        # sel_col, disp_col = st.sidebar.columns(2) # updated st.beta_columns() to st.columns()
        file1 = st.sidebar.file_uploader("Upload first document", type = ['txt','pdf','docx'])
        file2 = st.sidebar.file_uploader("Upload second document", type = ['txt','pdf','docx'])

        print("Document1...................................",file1)
        print("Document2...............................",file2)

        userchoice = st.sidebar.selectbox('Setect the feature function', selector_list)
        # st.write('You selected:', userchoice)

        # if file1 is not None:
        #     premise_text = file1.read()       
        #     premises = nltk.sent_tokenize(premise_text.decode('utf8')) # bytes to string

        # if file2 is not None:
        #     hypothesis_text = file2.read()           
        #     hypotheses = nltk.sent_tokenize(hypothesis_text.decode('utf8')) # bytes to string

        if file1 is not None:
            if 'pdf' in file1.name:
               premise_text = pdf_to_text(file1) 
            elif 'doc' in file1.name:
               premise_text = doc_to_text(file1)  
            else:
                premise_text = file1.read()       
            premises = nltk.sent_tokenize(premise_text.decode('utf8')) # bytes to string
            
        if file2 is not None:
            if 'pdf' in file2.name:
               hypothesis_text = pdf_to_text(file2) 
            elif '.doc' in file2.name:
               hypothesis_text = doc_to_text(file2) 
            else:
                hypothesis_text = file2.read()             
            hypotheses = nltk.sent_tokenize(hypothesis_text.decode('utf8')) # bytes to string

        if(file1 is not None) and  (file2 is not None) and userchoice == 'Similarity %':                       
           sim = calculate_similarity_percentage(premise_text.decode('utf8'),hypothesis_text.decode('utf8'))
        #    st.write(sim)
           sim_percent = "{:.0%}".format(sim)
           st.write ("\n The similarity of two documents is ", sim_percent)
           sim_p = 1 - sim
           #draw a pie chart
           plot_labels = 'Similarity %', 'Contradiction %'
           plot_sizes = [sim, sim_p]
        #    explode = (0.1, 0) 
           colours = ['#81ef7d','#ea696d']
    
           fig1, ax1 = plt.subplots()
           ax1.pie(plot_sizes, colors=colours, labels=plot_labels, autopct='%1.1f%%',
           shadow=True, startangle=90)
           ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

           st.pyplot(fig1, transparent=True)
           
        if(file1 is not None) and  (file2 is not None) and userchoice == 'Similarity and Contradition detection':         
            st.text('File upload successful!.')
            st.text('Checking for Similarity and Contradictions...')
            my_bar = st.progress(0)

            for percent_complete in range(100):
                time.sleep(0.05)
            my_bar.progress(percent_complete + 1)
            df_output = pd.DataFrame(columns = ['premise', 'hypothesis', 'prediction'])
                             
            row_count = 0
            
            # Add a placeholder for progress bar
            checking_text = st.empty()
            bar = st.progress(0)
                       
                      
            totalCount = len(premises)  * len(hypotheses)
            for premise in premises:
                for hypothesis in hypotheses:
                    outcome = check_similarity_contradiction(premise, hypothesis) 
                    row = {'premise':premise, 'hypothesis':hypothesis,'prediction':[outcome]}
                    row_count = row_count + 1
                    df_output =  df_output.append( row , ignore_index=True)
                    print("Row = ", row)
                    
                     # Update the progress bar 
                    checking_text.text(f'Processing Similarity and Contradiction Detection...  {row_count} of {totalCount}')
                    bar.progress((row_count/totalCount))
                    time.sleep(0.1)
               
            streamlit_df = pd.DataFrame(df_output)
            df_output.to_csv('predictions.csv')    
            st.dataframe(streamlit_df.style.apply(styler))
            
            @st.cache
            def convert_df_to_csv(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')


            st.download_button(
                label="Download data as CSV",
                data=convert_df_to_csv(streamlit_df),
                file_name='document_comparison.csv',
                mime='text/csv',
            )

        if(file1 is not None) and  (file2 is not None) and userchoice == 'Visualise Entities':         
            st.write('Document 1: \n')
            premise_tokens = visualise_ner(premise_text.decode('utf8'))
            annotated_text(*premise_tokens)
            st.write('\n')
            st.write('Document 2: \n')
            hypothesis_tokens = visualise_ner(hypothesis_text.decode('utf8'))
            annotated_text(*hypothesis_tokens)
            st.write('\n')
    
    if "load_state" not in st.session_state:
       st.session_state.load_state = False

timestr = time.strftime("%Y%m%d-%H%M%S")

if __name__ == "__main__":
    main()