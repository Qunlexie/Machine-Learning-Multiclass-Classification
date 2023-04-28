 # INTRODUCTION
Given three data sets representing training, testing, and label data with the goal of creating a multi-class classifier. Multiclass classification is a machine learning task with more than two classes; e.g., classify a set of images of fruits which may be oranges, apples, or pears. Multi-class classification makes the assumption that each sample is assigned to one and only one label: a fruit can be either an apple or a pear but not both at the same time. 
This work aims to classify the hypothetical data given into three classes: 0, 1, or 2 using categorical and numerical features. Our goal here is to maximize the accuracy of the predictions on the training set while generalizing well on the testing data. 

# METHOD
## XGBoost Classifier
XGBoost is a popular and efficient open-source implementation of the gradient-boosted trees algorithm. Gradient boosting is a supervised learning algorithm, which attempts to accurately predict a target variable or class by combining the estimates of a set of simpler, weaker models. 

In this work, I have used the XGBoost Classifier algorithm for the multiclass classification. I experimented with three variations of this model

## Model Training
First, the training features were divided into a set of training and validation sets. After running the preprocessing pipeline transformer, the model was fit on the training set and tested on the preprocessed validating set. The pre-tuned model was trained with default parameters in the \textit{XGBoostClassifier} module. A tuned XGB model was trained using the best parameters obtained from a grid search over a defined search space. The last model was built to account for class imbalance with weights built into the XGB model. 

## Metrics and Evaluation
The evaluation for all the models was done using accuracy as the main scoring criterion in accordance with the instruction for the assignment. Accuracy is the fraction of predictions our model got right. Formally, accuracy has the following definition: Accuracy = Number of correct predictions/ Total number of predictions. I also examined the F1-score though it was closely correlated with the accuracy in all experiments.  F1 score is a machine learning evaluation metric that measures a model's accuracy. It combines the precision and recall scores of a model. 

# RESULT 
 In this work, the result shows that the pre-tuned model obtained the highest accuracy of  91\% on the validation set and a macro average F1-score of 88\%. The pre-tuned model outperformed the other two: a Grid-search tuned XGB model and the XGB weighted model built to account for class imbalance.  

# CONCLUSION
In this project, I have explored the development of a multiclass classifier for the hypothetical data set with the goal of classifying instances into classes [0,1,2]. I used an XGBoost classifier which is one of the best Machine Learning algorithms for tabular data. I explored three variations of these models and used the best model for training the entire training data and finally generation prediction on the test data. Here I will go over the limitations and possible areas of improvement for this project. 


## Data Limitation
For this project, the dataset provided has 903 features but only 140 training instances with the task of generalizing on 560 instances in the test data. More data will greatly improve this project and provide a better signal for the prediction exercise. 

## More Model experiments
Given more time, I would experiment with deeper models e.g Deep Learning Neural Networks. Deeper models may or may not outperform XGBoost for this dataset but more experiments will provide us with more signals. 


## Model Accuracy vs Fairness and Interpretability tradeoff
There is no traditional definition for machine learning interpretability except that it should be a prerequisite for some
indispensable standards - trust, causality, informativeness, model transferability, and fair and ethical decision-making. In
supervised machine learning, machine learning stakeholders are mostly concerned about evaluation metrics that show
how well the model performs. Generally, interpretability approaches are classified into two groups. The first group
focuses on individual interpretation (local interpretability) while the second summarizes the entire model behavior
on a population level (global interpretability). Model-Specific interpretability is limited to specific models which by
definition are interpretations intrinsic to interpretable models. Model-Agnostic interpretability on the other hand is
used on any machine learning model and applied after the model has been trained (post-hoc). The xGBoost model is
the SOTA (state-of-the-art) for tabular data but it lacks interpretability. One approach for explaining the prediction is to use a model-agnostic tool like the Shapley additive explanations (SHAP) tool for both local and global instance explanations.

