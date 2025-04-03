import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from data_loading import load_data

def cap_outliers(df, cols, percentile=0.99):
    #Caps outliers in the given numeric columns at the specified percentile.
    for col in cols:
        if df[col].dtype.kind in 'biufc':  # Ensure the column is numeric
            upper_limit = df[col].quantile(percentile)
            df.loc[:, col] = df[col].clip(upper=upper_limit)
    return df

def remove_correlated_variables(X_train, X_test, vif_thresh=5, corr_thresh=0.7):
    #Removes highly correlated and high-VIF variables.
    numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    important_features = ['fico', 'amtfinanced_1req', 'ltv_1req', 'pti_1req']
    features_to_drop = []
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = numerical_features
    vif_data["VIF"] = [variance_inflation_factor(X_train[numerical_features].values, i)
                        for i in range(len(numerical_features))]
    
    high_vif_features = vif_data[vif_data["VIF"] > vif_thresh]["feature"].tolist()
    high_vif_features = [feat for feat in high_vif_features if feat not in important_features]
    features_to_drop.extend(high_vif_features)
    
    corr_matrix = X_train.drop(columns=features_to_drop).corr().abs()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > corr_thresh:
                colname_i = corr_matrix.columns[i]
                colname_j = corr_matrix.columns[j]
                if colname_i in important_features:
                    features_to_drop.append(colname_j)
                elif colname_j in important_features:
                    features_to_drop.append(colname_i)
                else:
                    mean_corr_i = corr_matrix[colname_i].mean()
                    mean_corr_j = corr_matrix[colname_j].mean()
                    features_to_drop.append(colname_i if mean_corr_i > mean_corr_j else colname_j)
    
    features_to_drop = list(set(features_to_drop))
    X_train_reduced = X_train.drop(columns=features_to_drop)
    X_test_reduced = X_test.drop(columns=features_to_drop)
    
    return X_train_reduced, X_test_reduced

def preprocess_data(train_df, test_df):
    #Preprocesses the data including handling missing values, outlier capping, and feature transformations.
    missing_percent_train = ((train_df.isnull().sum() / len(train_df)) * 100).round(2)
    high_missing_columns_train = missing_percent_train[missing_percent_train > 80].index.tolist()
    train_df.drop(columns=high_missing_columns_train, inplace=True)
    test_df.drop(columns=high_missing_columns_train, inplace=True)

    # drop race and gender since they reduce fair lending and bad_flag since it might result in data leakage
    columns_to_drop = ['Race', 'Gender']
    train_df.drop(columns=columns_to_drop, errors='ignore', inplace=True)
    test_df.drop(columns=columns_to_drop, errors='ignore', inplace=True)
    
    binary_categorical_cols = ['collateral_dlrinput_newused_1req']
    numerical_cols = [col for col in train_df.columns if col not in binary_categorical_cols + ['aprv_flag']]

    # capping outliers
    train_df = cap_outliers(train_df, numerical_cols, percentile=0.99)
    test_df = cap_outliers(test_df, numerical_cols, percentile=0.99)
    
    # transforms variables that are highly skewed 
    columns_to_apply_log = [col for col in numerical_cols if (train_df[col] > 0).all() and train_df[col].skew() > 1.5]
    for col in columns_to_apply_log:
        train_df[col] = np.log1p(train_df[col])
        test_df[col] = np.log1p(test_df[col])
        
    # scaling and imputer 
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    binary_categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('labelencoder', OneHotEncoder(sparse_output=False, drop='if_binary'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numerical_cols),
        ('binary_cat', binary_categorical_transformer, binary_categorical_cols)
    ])
    
    X_train = train_df.drop(columns=['aprv_flag'])
    y_train = train_df['aprv_flag']
    X_test = test_df.drop(columns=['aprv_flag'])
    y_test = test_df['aprv_flag']
    
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    feature_names = preprocessor.get_feature_names_out()
    X_train = pd.DataFrame(X_train, columns=preprocessor.get_feature_names_out())
    X_test = pd.DataFrame(X_test, columns=preprocessor.get_feature_names_out())

    X_train_reduced, X_test_reduced = remove_correlated_variables(X_train, X_test)
    
    return X_train_reduced, y_train, X_test_reduced, y_test


if __name__ == "__main__":
    train_data_path = "/Users/heba/Desktop/Erdos/Training Dataset A_R-384891_Candidate Attach #1_PresSE_SRF #1142.csv"
    test_data_path = "/Users/heba/Desktop/Erdos/Evaluation Dataset B_R-384891_Candidate Attach #2_PresSE_SRF #1142.csv"
    
    df_train, df_test = load_data(train_data_path, test_data_path)
    
    X_train, y_train, X_test, y_test = preprocess_data(df_train, df_test)
    X_train_final, X_test_final = remove_correlated_variables(X_train, X_test)
    
    print("Final Preprocessed Data Ready for Modeling")
