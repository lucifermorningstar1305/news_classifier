# NEWS CLASSIFIER

![](https://img.shields.io/badge/Code-Python-informational?style=plastic&logo=python&logoColor=yellow) ![](https://img.shields.io/badge/Framework-PyTorch-red?style=plastic&logo=pytorch&logoColor=red) ![](https://img.shields.io/badge/mainted-yes-green?style=plastic) 



### Description

This project is about the application of RNN with LSTM to classify pieces of news headline text into their respective categories i.e. whether it is a Science/Tech News or a Lifestyle News or a Political News etc.

The **objective** is simple, the user enters a piece of text related to a news article headline, and the model will predict what type of news it is.



### Categories

The model can classify the news headline in one of the following categories:

```text
GENERAL
SPORTS AND ENTERTAINMENT
WORLDNEWS
POLITICS
EMPOWERED VOICES
BUSINESS & MONEY
TRAVEL-TOURISM & ART-CULTURE
MISC
SCIENCE & TECH
PARENTING AND EDUCATION
LIFESTYLE AND WELLNESS
NATURE
```



### MODEL METRICS

Before anyone judges the model prediction, it’s better to highlight it’s metrics i.e. what is it’s `accuracy`, `f1-score` etc.

```json
{
	"validation-Loss" :  1.2118033170700073,
	"validation-accuracy" : 0.6360628498134329,
	"validation-f1-score" : 0.6272426502239625
}
```



### RUN Local Dev

To run the project in your local setup follow these steps:

```text
									==================================
											To Train the model
									==================================
									
	1. Clone the repo using git clone https://github.com/lucifermorningstar1305/news_classifier.git
	2. cd src
	3. python3 init.py --action train --epochs <number of epochs> --train_batch_sz <training-batch size>
	4. Wait for the model to train
	
	
								============================================
											To Test the model
								=============================================
								
	1. python3 app.py 
	2. Choose GET REQUEST
	3. Paste this URL : http://localhost:5000/predict/
	4. In the Query param enter a field with name : 'text' and value corresponding to the news article headline
	5. Done!
```





### Use Model as API

I already have an instance of this model running at https://newsclass.herokuapp.com/ . In-case one desires to use my API please follow these steps:

```
Choose a GET REQUEST
Paste the following URL : https://newsclass.herokuapp.com/predict
In the Query param enter a field with name : 'text' and value corresponding to the news article headline
Done!
```



### Acknowledgements

Building this project wouldn’t have been successful without help from these websites:

1. https://www.udemy.com/course/pytorch-deep-learning/learn/lecture/18315932#overview
2. https://www.kaggle.com/derinrobert/newsclassification-using-lstm
3. https://medium.com/@speedforcerun/heroku-deploy-no-web-process-running-6f6b4059765d