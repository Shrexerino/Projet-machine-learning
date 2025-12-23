This project aims to develop a robust Machine Learning model capable of predicting corporate bankruptcy based on financial indicators. The objective is to assist financial institutions in anticipating default risks by analyzing quantitative data.

Context and Challenge

The dataset contains financial information on approximately 7000 companies. The primary challenge lies in the severe class imbalance, as stable companies represent about 96.7% of the data, while companies in bankruptcy account for only 3.3%.

Given this distribution, a simple accuracy metric would be misleading. We therefore focus our performance analysis on the AUC-ROC score and Recall. This approach ensures we minimize false negatives, which corresponds to the critical error of predicting a company is stable when it is actually heading toward bankruptcy.


Methodology


Data Preprocessing 

The data cleaning process involves standardizing column names and removing constant variables. We applied logarithmic transformations (np.log1p) to skewed financial features to normalize their distributions. Finally, we used StandardScaler to center and reduce the data for optimal model performance.

Handling Imbalance 

To prevent the models from biasing towards the majority class, we implemented specific strategies. These include using SMOTE (Synthetic Minority Over-sampling Technique) within training pipelines, assigning balanced class weights for tree-based models, and using stratified sampling during the train/test split.

Modeling 

We trained and compared several algorithms using Imbalanced Pipelines and optimized hyperparameters via GridSearchCV. The models evaluated include Logistic Regression (Baseline), Support Vector Machine (SVM), Random Forest, Gradient Boosting, Voting Classifier, and Stacking Classifier.

Results and Performance

The Stacking Classifier emerged as the top-performing model on the test set, achieving an AUC-ROC of 0.956. By combining the predictive strengths of Random Forest and Gradient Boosting, it offers the most reliable predictions. The Gradient Boosting model followed closely with an AUC of 0.951, while the Voting Classifier achieved 0.947.

Feature Importance and Interpretability

While the Stacking Classifier provides the best predictive performance, it operates as a "black box." To understand the critical financial factors driving bankruptcy, we rely on the Random Forest model, which offers excellent intrinsic interpretability.

Our analysis identifies the following indicators as the most determinant:

Debt Ratio % This measures the overall level of indebtedness and is the strongest predictor of distress.

Current Liability to Assets This ratio highlights the proportion of short-term debts relative to total assets.

Borrowing Dependency This metric indicates the company's reliance on external financing to sustain operations.

Net Income to Total Assets Also known as Return on Assets (ROA), this measures the profitability and efficiency of the company.

These findings confirm that solvency and liquidity are the primary levers of financial health within this dataset.


Installation and Usage

To reproduce this analysis, ensure you have the necessary dependencies installed.

First, clone the repository to your local machine.

Next, install the required Python libraries using pip: pandas, numpy, matplotlib, seaborn, scikit-learn, and imbalanced-learn.

Finally, open the file Bankruptcy_Detection.ipynb using Jupyter Notebook or Jupyter Lab. Please ensure the data.csv file is present in the same directory as the notebook before running the cells.


Business Conclusion

The final model successfully detects approximately 95% of potential bankruptcies based on the AUC score. For a banking institution, this model serves as an effective Early Warning System, allowing risk managers to intervene before a client's financial situation becomes critical.
