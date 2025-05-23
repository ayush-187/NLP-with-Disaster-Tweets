# NLP-with-Disaster-Tweets
Disaster Tweet Classification Using DistilBERT

This project focuses on building a Natural Language Processing (NLP) model to classify tweets as either related to real disasters (1) or not (0). It uses the Kaggle Disaster Tweets Dataset, consisting of 7,613 training tweets and 3,263 test tweets.

Objective

To compare traditional machine learning approaches with modern transformer-based methods for binary text classification, and highlight the superior performance of fine-tuned transformers on contextual NLP tasks.

Approaches
	1.	Approach 1: TF-IDF + Ensemble (Gradient Boosting, Logistic Regression, Voting Classifier)
	•	Applied text cleaning (lowercasing, URL/punctuation removal, stopword filtering)
	•	Converted text to numerical features using TF-IDF
	•	Trained ensemble models with class weighting
	•	Limitations: lacks contextual understanding, prone to overfitting
	2.	Approach 2: Naïve Bayes + TF-IDF
	•	Used TF-IDF bigrams with Multinomial Naïve Bayes
	•	Fast and interpretable, but weak on context and nuance
	3.	Approach 3: BERT Embeddings + Logistic Regression
	•	Used CLS token embeddings from pre-trained BERT
	•	Lightweight and interpretable, but less adaptive than full fine-tuning
	4.	Final Approach: Fine-tuned DistilBERT
	•	Tokenized tweets using DistilBERT tokenizer
	•	Fine-tuned DistilBERT with a classification head
	•	Achieved higher accuracy and generalization by capturing tweet context
	•	Trained with AdamW optimizer, cross-entropy loss, and mixed precision

Key Highlights
	•	Preprocessing included cleaning text, removing special characters, and tokenizing inputs.
	•	Exploratory Data Analysis (EDA) involved word clouds, sentiment analysis, and token length distribution.
	•	Fine-tuning DistilBERT outperformed all other approaches, leveraging its deep contextual understanding and transfer learning capabilities.
