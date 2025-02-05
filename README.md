# Auto-Loan-Credit-Decisioning-Model
Background:
An auto loan is a type of secured credit that allows consumers to borrow money to purchase a vehicle, with said vehicle used as collateral on the loan. Prospective borrowers may apply for an auto loan individually or jointly. Joint borrowers are typically spouses, or a child and a parent. Borrowers repay the loan in fixed installments over a set period, with interest charged on the outstanding balance amount. Defaulting on the loan could cause considerable damage to a person's credit score and impact their future creditworthiness.
Setting:
Assuming that we are working in the consumer lending modeling team of a hypothetical financial institution (i.e., XYZ Bank) and are assigned a task to enhance the current application decisioning process with a focus on providing equal credit lending opportunity to all applicants. We want to build two credit decisioning models based on the Auto Loan applicants' credit quality information. The model will aim to identify the applicants with good credit quality and unlikely to default.
Information Provided:
We are given Auto Loan account data containing one binary response called 'bad_flag' in the datasets and a set of potential predictor variables from those Auto Loan accounts. Each record represents a unique account. There are two datasets: 1. Training data with around 21,000 records 2. Testing data with around 5,400 accounts
Objectives:
- Conducting an exploratory analysis to provide data summaries and necessary pre-processing for modeling.
- Developing and assessing machine learning models such as Logistic Regression, Decision Trees, Random Forest, Gradient Boosting (XGBoost, LightGBM).
- Comparing the results from these models to recommend the most effective approach for approving loan applications.
- Addressing critical business questions related to model transparency, gender bias, and potential racial discrimination.