# DeBerta-NLP-
Cross Domain Sentiment Analysis a NLP approach by DeBerta model is being deployed over here . Transfer Learning is simply a transfer of training from data base to anather data base . The model here is deployed on Review data set of Amzaon where the data problem simply define if a model is trained on books data set then can it give a Rating or review out put on other review data sets like DVD , TV etc . The concepts of FAST AI -Transformers and Bi Directional Containzers is being used here . De Berta is choosen over other models Like XLNET , BERT ,RoBERTa,AlBERTa,SVM and Deep Neural approaches for its promised best fit of accuracy . All the work references and Researches are added in the doc for the better understanding of the concepts.
Our Basic approach mainly works with first Pretraining the DeBerta model on the large text corpus then simply fine tuning and clustering on the data bse finally passing the trained architecutre to a transformer model - DeBerta-GPT2 for cross domain classfication and output .Other Details are mentioned Below .

# Requirements & Installations :
Since we are using Sciket Learn and Torch -Pytorch libraries for the deployment of DeBERTa , hence we need to install the following :

 ### For Google Colab & Python Server Deployments : 
     !pip install deberta // to install all the deberta model requirements 
     !pip install transformers 
     !pip install torch 
     !pip install pytorch 1.3.0
     !curl 
     !docker(optional if you run the data in docker)
     !Ubuntu 18.04 LTS e.g Linux server deployments
     !nvdia-docker-2 (optional)
     !bash -shell (optional)
     !pip install sciket-learn
     !install nltk (NLP Library)
  
  This Installations should be done before to run this DeBERTa script for Amazon review data set for Cross-Domain Sentiment calssification. 
  ### About the Data : 
The Data is being collected from Amazon Review data site : [Data](http://jmcauley.ucsd.edu/data/amazon/). The Data of kitchen ,Books ,DVD ,Video is used for the cross domain classification . First the model is trained and finetuned on a single data then tested on the other Data . Here Data preprocessing is done by using Tf-df vectorizer concept , for masking the vectors and pass it to the DeBERTa model for complete processing . More details about [tf-idf-vectorizer](https://www.google.com/search?q=tf-idf+vectorizer&oq=tf-df+&aqs=chrome.3.69i57j0l7.3633j0j7&sourceid=chrome&ie=UTF-8) can be found over here . This approach is used with some supporting python Libraries like NLTK for extracing the features out of it in the form of a key word extracion then passed for a vectorization for better preprocessing approach , This follows a hybrid architecture . This kind of approach being previously used by python paclage [Spacy](https://spacy.io/) in this [paper](https://github.com/deepopinion/domain-adapted-atsc). We followe the approach with a different pacakge for better accuracy of pre processing . Though both approaches can be used for expecting accuracy of >95%.
The Data is more than size of 1GB hence for handling this Big Datas we followed of the approach of saving the storing the data into XML format inside the file [dataset](https://github.com/Anustup900/DeBerta-NLP-/tree/master/Dataset).

### About DeBERTa Apprpoach : 

DeBERTa (Decoding-enhanced BERT with disentangled attention) improves the BERT and RoBERTa models using two novel techniques. The first is the disentangled attention mechanism, where each word is represented using two vectors that encode its content and position, respectively, and the attention weights among words are computed using disentangled matrices on their contents and relative positions. Second, an enhanced mask decoder is used to replace the output softmax layer to predict the masked tokens for model pretraining. We show that these two techniques significantly improve the efficiency of model pre-training and performance of downstream tasks.

Since DeBERTa is a very new algorithms it have some pretrained model as well a self deployable code for implementing the algorithm as an NLP approach in a dataset. So we did the second one as per the DeBERTa proposed code base we used DeBERTa on the above mentioned dataset to find the best fit accuracy over BERT as well other BERT models like RoBERTa, AlBERTa etc. 

DeBERTa is still unsupported in tensorflow estimators , Sciket Learn Library for a redy made call that will only deal with mathematical computation and Pytorch ,The deployed torch module of the deberta that calls its installaed library after installation in the form of [torch.nn.module](https://deberta.readthedocs.io/en/latest/modules/deberta.html#deberta-model) is : 
```
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("DeBERTa/deberta-base")
model = AutoModel.from_pretrained("DeBERTa/deberta-base")
```
This is the deployed pytorch API used for DeBERTa Deployment in transformer approach , this approach we used to deploy the transformers in our Code.The [documentation](https://huggingface.co/DeBERTa/deberta-base) regarding this speaks of it is deployed in the[ Hugging face transformers ](https://github.com/huggingface/transformers) .This hugging face transformers are pytorch supported transformer library which stores all the transformer approaches in FAST AI . It deals with GPT2, DeBERTa BASE ,Lm Transformers . Hence it is used in this approach . 

More about Deberta can be found [here](https://deberta.readthedocs.io/en/latest/modules/deberta.html#). 
     
    


     
  
