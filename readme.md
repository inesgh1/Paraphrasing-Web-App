# What is Paraphrasing?
Paraphrasing is the act of writing some text using different words, phrases, and sentence structures without changing its meaning. Paraphrasing is an important technique utilized in writing
Artificial intelligence is used to automate the process of paraphrasing. There exist many paraphrase tools that can paraphrase text automatically. However, programmers can also directly create a Python program to paraphrase text.
In this project we are going to check out both methods of paraphrasing, a) by using Python, and b) by using paraphrasing tools.
# How To Paraphrase Your Text Using Python?
There are are plenty of different ways to create a paraphrasing program using Python. However, currently one of the most popular methods is to use transformers.

**Transformers** are artificial neural networks that are able to learn the context of a text. This means they can understand its meaning. This makes them ideal to use for paraphrasing as you can count on them to not butcher the meaning of the text.

In this project we will be using the Pegasus model. It is a transformer that uses an encode-decoder model for sequence-to-sequence learning. You can learn more about the Pegasus model by reading its documentation.

Pegasus stands for “Pre-training with Extracted Gap-Sentences for Abstractive Summarization”. Don’t be confused by this. Abstractive summaries are basically paraphrasing as well. An abstract summary summarizes a text by rewriting all of its main points in a concise form using different wording. 

Now that we have covered the basics, let’s see how we can use Pegasus for paraphrasing.
# Import dependencies
``` 
!pip install sentence-splitter
!pip install transformers
!pip install SentencePiece 
```
# Setting Up Pegasus 
The next thing we need to do is to import the Pegasus model and set it up because we need it to do the paraphrasing. Without it, things will become more difficult.
To  set up Pegasus, we need to install Pytorch first. Pytorch is a Python package that provides high-level features of tensor computation and deep neural networks. It is the underlying framework that powers Pegasus.
``` 
  import torch
  from typing import List
  from transformers import PegasusForConditionalGeneration, PegasusTokenizer 
  model_name = 'tuner007/pegasus_paraphrase' 
  torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
  tokenizer = PegasusTokenizer.from_pretrained(model_name)
  model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
``` 
Running this code should download a bunch of files that look like this. The Pytorch model bin is a pretty large file, so don’t worry if it takes a bit of time to download it.

![image](https://github.com/inesgh1/Paraphrasing-Web-App/blob/main/set_peagasus.png)

# Access the model
``` 
def get_response(input_text,num_return_sequences):
  batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text
``` 
# Test the model
```
#test the model
context = "Which course should I take to get started in data science?"
num_return_sequences = 10
num_beams = 10
get_response(context,num_beams)
```
#### we'll get a result like that :
![image](https://github.com/inesgh1/Paraphrasing-Web-App/blob/main/outputof%20test.png)
# Break the Text into Individual Sentences
We got ten different paraphrased sentences by the model because we set the number of responses to 10. Paraphrase a paragraph: The model works efficiently on a single sentence. Hence, we have to break a paragraph into single sentences. The code below takes the input paragraph and splits it into a list of sentences. Then we apply a loop operation and paraphrase each sentence in the iteration.

The next step in paraphrasing is to break the provided sample text into sentences. This is because it is easier to paraphrase one sentence rather than an entire paragraph.
```
from sentence_splitter import SentenceSplitter, split_text_into_sentences
splitter = SentenceSplitter(language='en')

context = "I will be showing you how to build a web application in Python using the SweetViz and its dependent library. Data science combines multiple fields, including statistics, scientific methods, artificial intelligence (AI), and data analysis, to extract value from data. Those who practice data science are called data scientists, and they combine a range of skills to analyze data collected from the web, smartphones, customers, sensors, and other sources to derive actionable insights."
sentence_list = splitter.split(context)
num_return_sequences = 5
get_response(context, num_return_sequences)
paraphrase = [] 
for i in sentence_list:
	a = get_response(i,1)
	paraphrase.append(a)
```
##### Here is what we get as a result :

![result image](https://github.com/inesgh1/Paraphrasing-Web-App/blob/main/result%20list.png)

Combine the separated lists into a paragraph:
first we need to create a second split using the following code.
```
paraphrase2 = [''.join(x) for x in paraphrase]

paraphrase2
```
This will have a similar output as the first time. Then finally we will combine these split lists into a paragraph. The code for doing that is:

```
paraphrase3 = [''.join(x for x in paraphrase2) ]

paraphrased_text =str(paraphrase3).strip('[ ]').strip("'")

paraphrased_text
```











