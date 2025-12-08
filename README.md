Project Overview:

This project aims to develop a robust machine learning pipeline to predict company bankruptcy based on a dataset of financial ratios and attributes.
The primary objective is to classify companies as either "Financially Stable" or "Financially Unstable" by analyzing complex patterns in their economic data.
A significant challenge of this analysis is the highly imbalanced nature of the dataset, where actual bankruptcy cases are rare compared to stable companies.

Methodology and Workflow :

Our approach follows a structured data science lifecycle, designed to handle high-dimensional numerical data and class imbalance:

Data Preprocessing and Sanity Check:
- Cleaning: Standardization of column names and removal of constant features (zero variance).
- Outlier Management: We performed a deep statistical analysis (histograms and boxplots) to identify features with extreme skewness.
  We applied Log Transformations (log(1+x)) to specific non-fractional features where the tail distribution was too heavy, ensuring better model stability.

Feature Engineering:
- Scaling: Implementation of StandardScaler to normalize all features to a zero mean and unit variance.
- Feature Selection: Usage of Mutual Information Classification to filter out noise and retain only the variables that share a significant dependency with the target variable.

Handling Class Imbalance (SMOTE):
To address the scarcity of positive bankruptcy cases, we utilized SMOTE (Synthetic Minority Oversampling Technique).
This technique generates synthetic examples in the feature space for the minority class, ensuring the models are trained on a balanced distribution (50/50) rather than being biased toward the majority class.

Modeling Strategy: We implemented and compared a variety of classifiers to evaluate linear vs. non-linear performance:
- Logistic Regression (with ElasticNet regularization) as a baseline.
- Support Vector Classifier (SVC) for high-dimensional margin separation.
- Ensemble Methods: Gradient Boosting and AdaBoost to leverage decision trees for capturing complex non-linear relationships.

Requirements:
To reproduce this analysis, the following Python libraries are required:
- pandas and numpy (Data manipulation)
- matplotlib and seaborn (Visualization)
- scikit-learn (Modeling and preprocessing)
- imbalanced-learn (SMOTE implementation)

Bash:
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

Usage:
- Clone the repository.
- Ensure the dataset path is correctly set in the notebook.
- Run "Bankruptcy_Detection.ipynb" to execute the preprocessing pipeline and train the models.
