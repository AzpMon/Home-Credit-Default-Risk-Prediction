# Home Credit Default Risk Prediction

Alejandro Monroy Azpeitia, July 10th 2024.



### Project Overview
This project aims to predict the repayment ability of clients using alternative data, broadening financial inclusion for the unbanked population. The dataset used in this project is sourced from Kaggle and provided by Home Credit Group. The project involves extensive exploratory data analysis (EDA) using PySpark with SQL-like queries to efficiently manipulate and explore data. A significant challenge addressed in this project is the imbalance in the dataset, where the number of positive and negative examples in the target variable ('TARGET') differs significantly



### Dataset Description
The dataset includes several CSV files with information about loan applications, previous credits, and repayment histories.  Here are the main files used:

* application.csv: Main table with loan applications data.
* bureau.csv: Records of clients' previous credits from other financial institutions.
* bureau_balance.csv: Monthly balances of previous credits reported to Credit Bureau.
* POS_CASH_balance.csv: Monthly balance snapshots of previous POS and cash loans.
* credit_card_balance.csv: Monthly balance snapshots of previous credit cards.
* previous_application.csv: Records of all previous applications for Home Credit loans.
* installments_payments.csv: Repayment history for previously disbursed credits.
* HomeCredit_columns_description.csv: Descriptions of the columns in the various data files.



### Exploratory Data Analysis (EDA) with PySpark and SQL

In the EDA phase, extensive data analysis was conducted using PySpark, leveraging SQL-like queries for efficient data manipulation and exploration. A specific python class were created tom make easier the analysis. The following strategies were employed:

1. Read and Load Data: Each CSV file was read and loaded into a PySpark DataFrame.
2. SQL-Like Queries: Utilized PySpark's SQL capabilities to perform complex data transformations and aggregations.
3. Incorporate the Target Column: Merged the target column into each DataFrame, handling non-corresponding IDs appropriately.
4. Data Imbalance: Addressed the challenge of imbalanced data, where the number of positive and negative examples in the target variable ('TARGET') differs significantly.
5. Automatic Data Type Correction: Corrected data types of each column for consistency and accuracy.
6. Outlier Analysis: Identified outliers using statistical methods and visualizations.
7. Handling Null Values: Analyzed and filled null values based on column types and domain knowledge.
8. Correlation Analysis: Used correlation matrices to identify relationships between variables and eliminate redundant features.
9. Feature Engineering: Created new features based on domain knowledge to improve model performance.



### Model Training
A separate Jupyter notebook was used to train different models:

1. XGBoost with Hyperparameter Optimization (Optuna)
Used Optuna for hyperparameter optimization with cross-validation.
Evaluated using AUC-ROC curves for training and testing data.
Plotted ROC curves and visualized recall, precision, and F1-Score against different probability thresholds.
Selected the best model and displayed the confusion matrix.
2. XGBoost without Hyperparameter Optimization
Trained a basic XGBoost model without using Optuna for hyperparameter optimization.
Evaluated using the same metrics as the optimized model.
3. Logistic Regression with Data Balancing
Oversampling: Used to address data imbalance by increasing the minority class instances.
Grid Search for Hyperparameters: Conducted using grid search to find the best parameters.
Evaluation: Assessed models using ROC curves, recall, precision, and F1-Score.
Logistic regression was implemented in two different ways:

    *  With PCA
Reduced dimensionality to 2 principal components, explaining ~95% of the variance.
Implementation: Used sklearn to implement logistic regression due to reduced dataset dimensionality.
    *  Without PCA
Implementation: Used PySpark for logistic regression and hyperparameter search, suitable for the large and high-dimensional dataset.




### Repository Structure
* Data/: Contains the dataset files.
* EDA/: Contains Jupyter notebooks for EDA.
*  Model Training/: Contains Jupyter notebooks for model training.
* .gitattributes: Attributes for Git.
* README.md: Project overview and instructions.


### Results and Discussion
The results of the models, including evaluation metrics and visualizations, are discussed in the model training notebook. Handling data imbalance was crucial in achieving accurate predictions, especially highlighted in the logistic regression model. The best model was an Extreme Gradient Boosting (XGBoost) model optimized using Optuna, achieving an AUC for the ROC curve of 0.9803 and 0.9965 for the training and test data, respectively. A deeper analysis of the results can be read in the PDF document "Final Report: Home Credit Default Risk Prediction.



### Conclusion
This project showcases the application of advanced data analysis techniques and machine learning models using Grid Search with Scikit-Learn and PySpark, along with hyperparameter optimization with Optuna, to predict loan repayment abilities using alternative data sources. The use of PySpark with SQL-like queries facilitated efficient data handling, while addressing data imbalance was crucial for improving model performance.


### Acknowledgements
The dataset is provided by Home Credit Group via a Kaggle competition.
Thanks to the creators of PySpark, XGBoost, Optuna, and other libraries used in this project.







### Contact
For any questions or feedback, please contact me at:

   * amonroy.azpeitia@gmail.com

   * https://www.linkedin.com/in/alejandromonroyazpeitia/

