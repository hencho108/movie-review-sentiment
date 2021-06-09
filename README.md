# Machine Learning Deployment: Movie Review Sentiment Classifier

In this project my colleage Kelly Raas and I trained and deployed a machine learning model into an interactive web app. The web app allows a user to review random movies. As the user is typing, the app classifies the sentiment of the review in either positive or negative along with a probability in real-time. If the proposed sentiment classification is not correct, the user can submit feedback so that the model can learn. You can think of this as an app that suggests an appropriate score for a review with a sentiment analysis model that learns with more reviews.

View a live demo of the app [here](https://movie-review-ai.herokuapp.com/).

<p align="center">
<img src="https://github.com/hencho108/hencho108.github.io/blob/main/img/movie-demo.gif" width="327" height="633">
</p>

We built the application following these steps:
- Collecting training data from a [dataset](https://ai.stanford.edu/~amaas/data/sentiment/) of 50,000 highly polar movie reviews
- Training a SVM classifier using Sklearn (approx. 90% accuracy, precision and recall)
- Scraping movie titles from IMDB using Beautiful Soup
- Building an interactive web app using Dash (Flask based)
- Deploying the app to Heroku

Potential next steps for further improvement:
- Training a more sophisticated model (deep learning)
- Trying different pre-processing techniques, e.g. n-grams and lemmatization since the SVM model performs poorly in some cases such as differatiating between "good" and "not good"
- Setting up a SQL database to store user feedback
