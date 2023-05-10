# Netflix-CineMatch
Fundamentals of Machine Learning Capstone Project

### Introduction
Preamble: In 2006, Netflix released over 100 million ratings from over 400,000 users on over 17,000 films. Whoever improved the RMSE of Netflix’s “Cinematch” algorithm by 10% within 5 years received $1,000,000. Here, we revisit this notorious contest by using a subset of this dataset, with a different test set than the original contest.

```
Data: The file dataSet.zip contains two files: data.txt and movieTitles.csv
The data.txt file contains the ratings from the ~400k users on 5k movies, in the following format: Each row represents either a movie id, or a movie rating. If a row contains only a number followed by a colon, e.g.
758:
it represents that the following rows represent the ratings for the 758th movie in the dataset, until encountering the row that contains only
759:
the following rows represent the ratings for the 759th movie in the dataset, until the next number/column only row. Rows between those contain a first integer, a second integer and a date, from left to right, separated by commas. The first integer represents the id of the user (from 1 to ~400k) that gave the rating, the second integer represents the rating given by that user to the movie with the last encountered movie id (the integer before the colon). This second integer will always be a number between 1 and 5, representing a rating of 1 to 5 stars. There are no half-stars. The date is the date on which the user rated the movie, in YYYY-MM-DD format.
The movieTitles.csv file contains information as to which movie is represented by the movie id in the data.txt file. For instance, the movies “758” and “759” mentioned above are “Mean Girls” and “Fifteen Minutes”, respectively. Each row in this file represents a movie, the first column is the movie id (corresponding to the number in front of the colon of the rows in the data.txt file), the 2nd column contains the release date and the 3rd column the movie title. Strictly speaking, this file is provided for your edification only – it probably won’t have much impact on your model performance per se. However, this could be a good source of material for the extra credit observations. For instance, how does release date impact rating, how about the interval between release data and rating date, how about the linguistic features of the title? Have titles gotten shorter over time, etc.?
```

```
Prompt: Use *all* ratings for a given movie in the training set, except for *one* randomly picked rating. That one rating (per movie) constitutes the test set. So the test set will be 5000x1 randomly picked ratings (one per movie). Use all the other ratings in the training set. *Use this test set to determine the RMSE of your model. Try to get the RMSE as low as possible without having any leakage between training and test set.
```

### Analysis

Challenge: The ratings data is natively provided in a large text file. With the ratings of each movie just glued one after each other. You need to find a way to parse this file, extract the relevant information and put it into some kind of usable format (likely an array or dataframe).

> Response: To parse the ratings data, I first read the entire content of data.txt file into a list of strings, where each string represents a line of text in the file. I iterated over the list and stripped new line characters. Then, I defined a regular expression pattern that matches lines that consist of a movie ID followed by a colon. Iterating over the stripped data, if the line matched the regex pattern, I extracted the movie ID, customer ID, rating, and rating date and appended them to a flat data list. Lastly, I created a Pandas DataFrame from the flat data list.

Challenge: There is lots of missing data. In fact, most of the data will be missing. Most users have not rated most of the movies. You will need to find a way to handle that, either by imputation or in some other way. There might even be information in the missing data (as this might reflect a choice by a user NOT to see a movie that they anticipated not to enjoy)

> Response: The ratings data has roughly 27M ratings. There are 5K unique movies and ~470K unique customers. If each customer rated each movie, there would be ~2.4B total ratings, so there are 2B+ missing ratings. The missing ratings could be filled in by mean imputation where the mean rating of each movie ID is calculated and used to impute missing ratings for each movie. In my case, I found it more appropriate to omit the missing ratings data due to the additional computational complexity, memory limitations, risk of not capturing the nuances of the underlying data, and potential for overfitting. The movie titles data had two missing release date values. I used the mean of all release dates to impute those two missing values. Another technique I considered was Googling the movie and replacing the missing value with its reported release date.

Challenge: There is lots of data. This is genuinely a “big” dataset, so you need a model/infrastructure that can handle that. Don’t build a model that will take months to train, as you don’t have that much time. 

> Response: One advantage of RNNs is that they process input data in batches, which can speed up the training process by improving computational efficiency since vectorized operations can utilize specialized hardware like GPUs. RNNs are also able to take advantage of mini-batch gradient descent, which updates the model parameters after each batch of data. This can reduce the computational burden of processing large datasets and mitigate risks of overfitting. My model also employs dropout regularization, which randomly drops a proportion of the neurons in the network during training. This prevents overfitting, improves generalization performance, and can reduce the iterations to achieve good performance. Finally, early stopping can be used to evaluate the model’s performance on the validation dataset during training and stop the training when the validation RMSE stops improving.

Challenge: The movie ratings themselves (per movie) are unlikely to be normally distributed. 

> Response: RNNs can be good models to use when the target variable is not normally distributed because they are capable of modeling complex, nonlinear relationships between inputs and outputs, while capturing patterns and dependencies over time. I also used a TanH output activation function to add additional nonlinearity to the outputs, which marginally improved my model’s performance.

Challenge: There might be temporal or seasonal effects in movie ratings. You do not have to include that information in your model (as to when a movie was rated), but you can expect the model to improve if you do so. However, doing so will be challenging (you would have to model some kind of temporal or seasonal kernel). So including this kind of information is optional (but could be interesting, if you like to be challenged).

> Response: An RNN is good at capturing temporal or seasonal effects in movie ratings because it is designed to process sequential data in a way that captures the dependencies in the data over time. In the context of movie ratings, temporal effects refers to the fact that movie ratings may be influenced by the release date of the movie and the length of time since the movie’s release. For example, a movie that receives high ratings immediately after its release may indicate substantial hype surrounding the movie, while a movie that receives high ratings months after its release may indicate sustained popularity. An RNN is well-suited for capturing these types of effects since it models the dependence between inputs at different time steps, allowing it to learn patterns in the data over time.

<img width="589" alt="ratings" src="https://github.com/richardwzhu/netflix-cinematch/assets/30671520/57920eee-bd26-46fa-95dc-f1d1191fe718">

#### Figure 1 

<img width="433" alt="releases" src="https://github.com/richardwzhu/netflix-cinematch/assets/30671520/b34edbf7-db8f-4167-aabd-548a215a74bf">

#### Figure 2

During data exploration, I created the above plots. Figure 1 illustrates the relationship between the release date and the number of movies released. There is a notable exponential growth in movies released as the year approaches 2000. In the situation where release date is a significant predictor of ratings, it may be appropriate to balance the movies so that release dates are more evenly distributed. Figure 2 illustrates the proportion of each category of rating, the total number of unique movies (5,000), the total number of unique customers (472,542), and the total number of ratings (27,010,225). The ratings tend to be relatively positive, with most of the ratings being greater than three. This may be due to the fact that customers who disliked a movie may tend to not leave ratings. Therefore, low reviews may indicate a customer's extreme negative sentiment.

### Result
#### Final validation RMSE: 0.365
