# Google Machine Learning Crash Course

This [Crash Course](https://developers.google.com/machine-learning/crash-course/ml-intro) introduces some basic machine learning topics and provide hands-on exercises using TensorFlow. This course does not provide very detailed concepts. Instead, it throws out common topics and few examples, and can be a good complement to the [coursera machine learning course by Andrew Ng](https://www.coursera.org/learn/machine-learning/home/welcome). You can find a detailed learning topics I summarized on [github](https://github.com/YuZhangIsCoding/ML_coursera). Here, I will just list some of the bullet points I learned from this Crash Course.

* L2 Loss (Least Squared Error) 
    * Mean Square Error (MSE) is the average squared loss per example.

    * Although MSE is commonly used in machine learning, it's neither the only practical loss function nor the best loss function for all circumstances

* Stochastic Gradient Descent and Mini-Batch Gradient Descent
    
    * A large dataset with randomly sampled examples probably contains redundant data.

    * Redundancy becomes more likely as the batch size grows

* TensorFlow API Hierarchy

    Contains differnt levels of API

        * Highest level of abstrction to solve problems, but may be less flexible
        * If need additional flexibility, move one level lower

* Workflow To Use TensorFlow
    
    1. Select features and targets from data
    2. Use tensorflow to build feature columns and optimizer with hyperparameters such as learning rate
    3. Build an estimator based on step 2
    4. Train the data with steps specified
    5. Calculate the training and validation loss
    6. Tune learning rate, steps and batch size to reduce loss
    7. Try different features
    8. Test new features from synthesis of old features
    9. Cap outliers

* Batch Size

    * Steps are the total number of training iterations. One step calculate the loss from one batch and use the value to modify the model's weights once
    * Batch size is the number of examples (randomly) selected for each single step
    * Total number of trained examples = batch size &times; steps
    * Periods control the granularity of reporting
    * Number of training examples in each period = (batch size &times; steps)/periods

* Overfitting

    * Split the data into training set, validation set and test set

* Representations

    * String can be represented as a string vector using one-hot encoding
        * A binary vector that only has one element of 1 and all others 0
        * 1 means the feature belongs to some category

* Properties of Good Features

    * Appear with non-zero value more than a small handful of times in the dataset: *avoid rarely used data*
    * Clear and obvious meaning: *sanity check*
    * Shouldn't take on magic values: *split into 2 separate features, one representing whether the feature exists and second indicate the values*
    * The definition of a feature should not change over time
    * Should not have crazy outlier values

* Good Habits: *know your data*

    * Visualize
    * Debug
    * Monitor

* Feature Scaling

    * Scale the features to [-1, 1] or scaling with Z score to (mostly) [-3, 3]
    * Logarithmic scaling
    * Clip feature
    * The binning trick: *create several boolean bins, each mapping to a new feature and allow model to fit a different value for each bin*
        
        E.g. there's no linear relationship between latitude and the house price, but the individual latitudes may be a good indicator of house values. Binning by location or by quantile (ensure each bin has same examples).

* Scrubbing Data
    * Many exmaples are not reliable because:
        * omitted values
        * Duplicate values
        * Bad label
        * Bad feature values
    * Detect bad data by
        * Histogram
        * Min and max
        * Mean and median
        * Standard deviation

* Pearson Correlation Coefficient
    * Learn the linear correlation between targets and feature and between feature and feature

* Feature crosses
    * Combine several features together (polynomial terms), and incorporate nonlinear learning into linear learner
    * For boolean features, feature crosses may be very sparse
    * Combine feature crosses and massive data is one efficient strategy for learning highly complex systems
    * Bucketized column

* Regularization: Avoid Model Complexity When Possible
    * Minimize lost(data|model) + complexity(model)
        * Smaller weights: complexity as the sum of the squares of the weights
        * Smaller number of features with nonzero weights
    * Performing L2 regularization has following effect:
        * Encourage weight values toward 0
        * Encourage the mean of the weights toward 0, with a normal(Gaussian) distribution
    * Increasing the lambda value strenghthens the regularization effect

* Logistic Regression
    * Used as:
        * Probabilities(expectation)
        * Classifications
    * Regularization is very important for logistic regression:
        * L2 regularization
        * Early stopping

* Classification
    * Evaluation metrics:
        * Accuracy, but breaks down when only contains extremely low positives or negatives
        * True positives, false positive, true negatives, false negatives
    * ROC curve:
        * Each point is the TP and FP rate at one decision threshold
    * AUC: area under the ROC curve
        * Gives an aggregate measures of performance aggregate across all possible classification thresholds
    * Prediction bias:
        * Should have average of prediction == average of observatioins, otherwise biased
    * Bias is a canary:
        * Zero bias alone does not mean everything in your system is perfect
        * But it's a good sanity check
    * If having bias:
        * Imcomplete feature set
        * Buggy pipeline
        * Biased training set
    * Not suggested to fix bias with a calibration layer, fix the model instead

* L1 Regularization
    * Feature crosses: sparser feature crosses may significantly increase feature space
        * Model size (RAM) become huge
        * Noise coefficients (overfitting)
    * Penalize the sum of abs(weights)
    * Convex problem (L0 regularization is non-convex, thus hard to optimize)
    * Encourage sparsity unlike L2

Neural Networks:
    * Nonlinearity::
        * Relu law: rectify linear unit (max(0, x)), usually a little better than sigmoid
        * Sigmoid
    * Backpropagation
        * Gradients can vanish
            * The gradients for lower layer can become very small and thus the layers may be trained very slow
            * Relu activation function may help prevent vanishing gradients
        * Gradients can explode
            * The weights in the network are very huge, they may get too large to converge
            * Large batch size and slower learning rate may help
        * ReLu layers can die
            * If the weighted sum is smaller than 0, the ReLu unit can get stuck, and the gradient cannot flow through during the backpropagation
            * Lowing learning rate may keep ReLu unit from dying
    * Normalizing feature values
        * Have reasonable scales
            * Roughly 0-centered, [-1, 1] range often helps
            * Help gradient descent work, avoid NaN trap
            * Avoid outlier values also help
        * Methods
            * Linear scaling
            * Hard cap to max, min
            * Log scaling
    * Dropout regularization
        * Work by randomly drop out units in the network for a single gradient descent
        * The more you drop out, the stronger the regularization
            * 0.0 -> no dropout regularization
            * 1.0 -> drop everything out and learn nothing
            * Intermidiate values are helpful

* Multi-Class Neural Networks
    * One-vs-all Multi-Class
        * Create one unique output for each possible class
        * Train that on a signal of "my class" vs "all other classes"
        * Can do in a deep network, or with separate model
        * Reasonable when the total number of classes is small, but becomes increasingly inefficient as the number of classes rises
    * SoftMax Multi-Class
        * Add additional contraints: require output of one-vs-all nodes to sum up to 1
        * Helps the training converge quickly
        * Outputs can be interpret as probabilities
    * What to use and when
        * Multi-class, single-label classification
            * An example be only a member of one class
            * Constraint that classes are mutually exclusive is helpful structure
            * Useful to encode this in the loss
            * Use one SoftMax loss for all possible classes
        * If an example may be a member of more than one classes, use logistic regression instead
    * SoftMax Options
        * Full SoftMax
            * Brute force, calculate for all classes
            * Fairly cheap when the number of classes is small, but prohibitively expensive when the number fo classes climbs
        * Candidate sampling
            * Calculate for all positive labels, but only for a random sample of negatives
            * Improve the efficiency in the problems that have a large number fo classes

* Confusion matrix

* Embedding
    * E.g. Use 0 and 1 to represent if a movie is watched is not very efficient. Instead build a dictionary mapping each feature to an integer (movie no.)
    * Higher dimensional embedding can more accurately represent the relationship between input values
    * More dimensions increases the chance of overfitting and lowering training efficiency
    * Empirical rule: Dimensions = (possible values)^(1/4)

