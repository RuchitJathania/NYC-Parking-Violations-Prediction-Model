# NYC-Parking-Violations-Prediction-Model
Neural Network was created, trained, and evaluated for the prediction of Violation Code and for the prediction of fine amount for input data from subset NYC Parking Violations for fiscal year of 2024 for end goal of estimating the potential revenue from fines

Introduction/Abstract:
A tensorflow/keras Neural Network was created, trained, and evaluated for the prediction of Violation Code and for the prediction of fine amount for input data from subset NYC Parking Violations for fiscal year of 2024 for end goal of estimating the potential revenue from fines in question. The prediction on violation code had 58.43% test accuracy, which is a 31.1% increase in accuracy over model taken inspiration from Downey [1]. The other model used to predict fine amount was 75.14% accurate for test split, and only produces a 0.42% error for revenue estimation. 
Problem:
In fiscal year 2023, NYC issued $1.16 billion in fines for parking and camera violations [3]. Revenue generated from fines accounts for a significant portion of the city’s general revenue. However, NYPD and other bodies of regulation must spend crucial resources, time, and efforts to decide where to concentrate policing and issuing of fines due to limited resources. Hence, it is worthwhile to anticipate the revenue generated given some time, location, type of vehicle, license, etc. Accurately predicting the revenue from issuing fine(sum of fines issued) constitutes an important problem to solve for optimizing the use of resources to maximize generated revenue for NYC. 
Dataset:
Data was taken from NYC OpenData, the “Parking Violations Issued – Fiscal Year 2024” from [4]. Violations are taken from September 1st to February 28th to provide enough training data for the machine learning model eventually finalized and implemented. As from visual below, most features or columns of dataset are categorical data and there are very few numerical features available to use. 
 
Figure 1. Jupyter Lab ‘df.head(10)’ output for original Data Frame used.
Other Models/Projects:
Prior machine learning projects have been completed using the same dataset of “Parking Violations Issued” but for previous years. The big data/machine learning project from Ramanathan’s GitHub uses Parking Violations Issued from a prior year and they utilized a XGBoostClassifier and RandomForestClassifier to predict the location of the violation(meaning predicting which borough of NYC, Manhattan, Bronx, Kings, Queens) [2]. However, while these models produced 99.9% and 95% accuracy, they have utilized as input features which directly correspond to location like ‘Violation County Index’, ‘Issuer Precinct’, ‘Street Code’, etc. This means they have used data features as inputs which are directly linearly dependent to output. The idea of predicting the precise location of a violation given likely input data is innovative; however, this implementation only performs so accurately since location itself is basically used as an input to predict which borough of NY the violation occurs in.
A more considered and rational model using similar data from fiscal years 2013 to 2017, made by Taylor Downey, utilizes a ‘keras’ deep feedforward neural network to perform multiclass classification to predict the specific violation code of the fine [1]. Downey utilized features that were independent from violation code and features that were more descriptive of the data which could potentially lead to relation or pattern with violation code(or type of violation committed). This model uses only 12 features from the available 43. The current model described in this report also utilizes many of the same features but also some changes. There is also a K-Nearest Neighbor and Decision Tree Classifier used, but the Neural Network performed the best with accuracy of 45.40% on 1.4 million samples for training [1]. 

Feature Selection/Preprocessing:
For the eventual model chosen, a specific series of feature processing was performed. The finals features used as input include: ‘Vehicle Body Type’, ‘Vehicle Make’, ‘Vehicle Color’, ‘Violation County’, ‘Street Code1’, ‘Street Code2’, ‘Street Code3’, ‘Issuing Agency’, ‘Plate Type’, ‘Violation Precinct’, ‘Registration State’, ‘Day of Week’, and hour of day in cyclic form.
 
Figure 2. Feature Selection, encoding, and normalization.
Since one of the models uses the fine amount as target variable, a new feature column was made mapping the violation code to the fine amount based on fine from external dataset. Next, the vehicle color categorical data had 349 unique values which were reduced to 14 color categories to improve model training. Samples with NaN or missing values were dropped instead of choosing to use filling in with the mode of the feature to prevent overfitting and skewing of data to more popular types of violations. 
Differing form Downey’s feature selection, this model uses the time during the day as a cyclic input instead of only using day of the week for temporal numeric data. Additionally, vehicle color category was reduced significantly more than what model from [1] implemented. Instead of using ‘Street Name’ like [1], which produces around 14,000 one-hot encoded parameters, ‘Street Code’ was utilized since each only produces around 5000 parameters. Also, Issuer precinct was ignored since it was common with Violation precinct and would be redundant linearly dependent data. 
Model(s):
After testing a few model types from LightGBM classifier, to SVM, to logistic regression classifier, the use of a Neural Network was finalized with one hidden layer of 128 units, another hidden layer with 64 units, and a final output layer using SoftMax. The hidden layers use ReLU activation and there are dropout layers in between with 30% dropout rate. Initially, seeing as predicting location was not too useful, and the end goal was to estimate revenue for given input data, the preliminary model was chosen to predict the Violation Code directly. There are around 95 different violation codes, meaning each sample produces 95 different probabilities for the prediction output. After testing, this model seemed to improve on [1]’s accuracy, yet the relation between input data and violation code was not strong enough. After predicting the violation code, the codes are then converted to fine amounts and summed to estimate revenue for the  input data.
Hence, the same model was modified to predict the fine amount instead, in categorical form as an index of fine amount to improve model accuracy for predicting fine amount revenue. Then, the fine amount index is converted to fine amount and revenue generated is computed from prediction. Adam optimizer was used, loss function of categorical cross entropy loss was used, and accuracy was used to compile the tensorflow keras model. 
 
Figure 3. Neural Network Model code.
Training Model/Results:
The initial model produces a 58.43% test accuracy for prediction on violation code, which is a 31.1% increase in accuracy over model from Downey [1]. This was trained over 20 epochs with batch size of 128. Both models discussed use 2,149,799 samples and use 19,181 feature columns after feature normalization and processing. 
 
Figure 4. Prediction on Violation Code Accuracy.

 
Figure 5. Prediction on Violation Code Loss.

This model calculates the predicted revenue of $34,944,600 and an actual revenue of $34,036,230.
The next model, predicting the fine amount as categorical output, results in 75.14% accuracy and predicted a revenue of $33,892,030 and the same true revenue of before of $34,036,230 for the given input data. 
 
Figure 6. Prediction on Fine Amount Accuracy.

 
Figure 7. Prediction on Fine Amount Loss.

Conclusion:
While both models implemented result in a revenue prediction of within 5% error overall for all samples in test split, the model that predicts the fine amount directly more accurately finds patterns between the input features and resulting fine amount. The model predicting on Violation Code does not have as strong of pattern or relation between input feature and one of the 95 types of violations. Furthermore, not a fault of the models implemented, but the high degree of categorical data that may not contribute or reasonably infer which type of violation performed is more of an issue with the dataset and available features to use. If more useful and telling information was collected about resulting violation, like demographic information, socioeconomic information of area of violation, relational data to holidays or major events, etc., then maybe those new features could provide a better prediction on violation code. Overall, work on this project highlights the importance of feature selection, proper feature processing, scaling and normalizing, and emphasizes that more complex models will not always produce better results if your data is not optimal or processed in a way the model can take advantage of. 






Citations
[1] Taylor Downey (2022) ticket_analysis [Source Code and README].
https://github.com/tadowney/ticket_analysis

[2] Vishwesh Ramanathan, Ashwin Nair, Raghav Moar (2020) Real-time-prediction [Project Report]. https://github.com/Vishwesh4/Real-time-prediction

[3] Division of Treasury & Payment Services NYC. (2023). ANNUAL REPORT OF NEW YORK CITY PARKING TICKETS AND CAMERA VIOLATIONS. NYC Department of Finance. https://www.nyc.gov/assets/finance/downloads/pdf/23pdf/2023-local-law-6-report.pdf

[4] (DOF), Department of Finance. “Parking Violations Issued - Fiscal Year 2024: NYC Open Data.” Parking Violations Issued - Fiscal Year 2024 | NYC Open Data, 16 Dec. 2024, data.cityofnewyork.us/City-Government/Parking-Violations-Issued-Fiscal-Year-2024/pvqr-7yc4/about_data. 









