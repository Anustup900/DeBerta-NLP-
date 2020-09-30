# DeBerta-NLP-
Cross Domain Sentiment Analysis a NLP approach by DeBerta model is being deployed over here . Transfer Learning is simply a transfer of training from data base to anather data base . The model here is deployed on Review data set of Amzaon where the data problem simply define if a model is trained on books data set then can it give a Rating or review out put on other review data sets like DVD , TV etc . The concepts of FAST AI -Transformers and Bi Directional Containzers is being used here . De Berta is choosen over other models Like XLNET , BERT ,RoBERTa,AlBERTa,SVM and Deep Neural approaches for its promised best fit of accuracy . All the work references and Researches are added in the doc for the better understanding of the concepts.
Our Basic approach mainly works with first Pretraining the DeBerta model on the large text corpus then simply fine tuning and clustering on the data bse finally passing the trained architecutre to a transformer model - DeBerta-GPT2 for cross domain classfication and output .Other Details are mentioned Below .

# Requirements & Installations :
Since we are using Sciket Learn and Torch -Pytorch libraries for the deployment of DeBERTa , hence we need to install the following :
 
 
     #For Google Colab & Python Server Deployments : 
     !pip install deberta
     !pip install transformers
     
      
      




