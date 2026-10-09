# Sentiment Analysis

This project predicts whether an English movie review is positive or negative. I use TF-IDF to convert text into numerical features and compare logistic regression with a small PyTorch neural network.

## Dataset

I use the IMDb movie-review dataset:
https://huggingface.co/datasets/stanfordnlp/imdb

The dataset contains review text and sentiment labels:
- 0: Negative
- 1: Positive

I split the cleaned training data into 80% training and 20% validation. The provided test set is kept separate for evaluation.

## Project Steps

1. Load and explore the dataset.
2. Remove HTML tags, empty reviews, duplicates and conflicting labels.
3. Remove test reviews that also appear in the training data.
4. Create training and validation sets.
5. Convert reviews into TF-IDF features.
6. Train logistic regression and a PyTorch neural network.
7. Evaluate both models and inspect incorrect predictions.
8. Test new reviews.
9. Save and reload the models.

## Models

### Logistic Regression

I use logistic regression as a baseline to compare with the neural network.

### PyTorch Neural Network

The network includes:
- An input layer matching the TF-IDF features
- One hidden layer with 64 neurons
- ReLU activation
- Dropout
- One output for binary classification

I use BCEWithLogitsLoss and the Adam optimizer. Training uses mini-batches, and I keep the weights from the epoch with the best validation F1 score.

## Evaluation

I compare the models using:
- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrices

The notebook also includes training graphs and examples of incorrect predictions. The final model is selected using validation F1.

The test results use a cleaned subset of IMDb. I had already checked the original test results in an earlier attempt, so the test set is not completely unseen.

## Tools Used

- Python
- pandas and NumPy
- scikit-learn
- PyTorch
- Matplotlib
- Hugging Face Datasets
- joblib
- Google Colab

## How to Run

1. Open the notebook in Google Colab.
2. Run the cells in order.
3. Change the example reviews to test your own text.
4. Download the saved model files before closing the session.

The first cell installs the required libraries. A GPU is optional.

## Saving the Models

The notebook saves the TF-IDF vectorizer, logistic-regression model, neural-network weights, model settings and evaluation results.

It also reloads the saved models and checks that their predictions match, so they can be used without retraining.

## Limitations

The model only predicts positive or negative sentiment. It does not have a neutral class.

It may make mistakes on sarcasm or reviews with mixed opinions. Since it is trained on English movie reviews, it may not work as well on other types of text.

## Author

Aitezaz Alam
