# Kaggle: Don't Get Kicked
## Programming Language
* Python

## Tools
* Numpy
* sklean
* pandas

## Approach
1. Analyze data with pandas dataframe.  
	In this dataset, we have 72983 data points in total. 64007 points are class 0 and 8976 are class 1 (kicked car). Each data point has 34 features, including 19 numeric features and 15 categorical features. Due to the time limitation, in this assignment I just used the numeric data.
	
2. Feature engineering  
	1. Manually select few features at the first.

3. Data preprocessing  
	1. Imputation: fill the missing values with the most frequent numbers (imputation by the most frequent numbers outperforms by mean)
	2. Standadization: for each feature column we:  
		a). calculate mean b). subtract by mean c). divide by variance.  
		By doing so, the data would be in normal distribution with 0 mean and has unit variance.

3. Train/validation split  
	We split 72983 data points into training and validation set with the ratio 7:3. The corresponding amounts are 58386 and 14597.
	
4. Imbalanced classes  
	Downsample class 0 data to balanced the amount of 2 classes.
	
5. Training  
	1. Use logistic regression for classification and set class weight to 'balanced'
	2. Hyperparamter tuning: find best lambda based on validation mean f-score


6. Evaluation  
	We use precision, recall and F-score for evaluation because the class 0 (negative) are majority.

## Result

Best lambda = 0.00001   
Training set  


|					|Class 0|Class 1|
|---------------	| -------|------|
|Precision  |0.768	|0.447|  
|Recall     |0.608	|0.632|  
|F-score    |0.679	|0.524| 

Averaged F-score: 0.601  
Accuracy:   0.616 

Validation set  

|					|Class 0|Class 1|
|---------------	| -------|------|
|Precision  			|0.921	|0.182  |
|Recall     			|0.604	|0.629  |
|F-score    			|0.730	|0.282  |

Averaged F-score: 0.506	  
Accuracy: 0.607


## Discussion
1. Use categorical features to improve model performance.
2. Not much difference with different lamdbas.
3. Not much difference with downsampling.
4. Try different classifiers such as SVM or random forest.