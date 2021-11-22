# Human-activity-recognition-with-smartphones

In this notebook, I will predict the human activities of daily living using Keras Sequential model.

The Human Activity Recognition database was built from the recordings of 30 study participants performing activities of daily living (ADL) while carrying a waist-mounted smartphone with embedded inertial sensors. The objective is to classify activities into one of the six activities performed.

Description of experiment

The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope were captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz. The experiments have been video-recorded to label the data manually. The obtained dataset has been randomly partitioned into two sets, where 70% of the volunteers were selected for generating the training data and 30% for the test data.

The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low-frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. From each window, a vector of features was obtained by calculating variables from the time and frequency domain.

This dataset contains 563 columns, which include also categorical features.
The target for this data set is called "Activity". We have 6 classes, so this is a classification problem. Because the dataset contain 563 columns, I decided to try keras Sequential model with Dense layers.
I train the model with our dataset and evaluate the model with test labels and test features.
The score I ve obtained is 0.8384, a good one!

Conclusions:

The largest entry is the predicted class—the class with the highest probability.

Because I have many classes, this problem is an instance of multiclass classification; and because each data point should be classified into only one category, the problem is more specifically an instance of single-label, multiclass classification.

The best loss function to use in this case is categorical_crossentropy. Loos function measures the distance between two probability distributions: here, between the probability distribution output by the network and the true distribution of the labels. By minimizing the distance between these two distributions, you train the network to output something as close as possible to the true labels.

Evaluation metrics to be used for classification accuracy.

In a multiclass classification problem (many output classes).The rmsprop optimizer is generally a good enough choice, whatever the problem.

