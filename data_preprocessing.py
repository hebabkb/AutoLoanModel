import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from data_loading import load_data

def preprocess_data(train_df, test_df):
    
    #Dropping columns that have more than 80% of missing data in train
    missing_percent_train = ((train_df.isnull().sum()/len(train_df))*100).round(2)
    high_missing_columns_train = missing_percent_train[missing_percent_train > 80].index.tolist()

    train_df.drop(columns=high_missing_columns_train, inplace=True)
    test_df.drop(columns=high_missing_columns_train, inplace=True)
    # Identify categorical and numerical columns
    binary_categorical_cols = ['collateral_dlrinput_newused_1req']  # Binary categorical variables
    multi_categorical_cols = ['Gender','Race']  # Multi-class categorical variables
    numerical_cols = [col for col in train_df.columns if col not in binary_categorical_cols\
                                                                               +multi_categorical_cols+['aprv_flag']]

    # Preprocessing pipeline
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Using KNN Imputer for numerical data
        ('scaler', RobustScaler())
    ])

    multi_categorical_transformer = Pipeline([
         ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(sparse_output=False))
    ])
    binary_categorical_transformer = Pipeline([
         ('imputer', SimpleImputer(strategy='most_frequent')),
        ('lablecoder', OneHotEncoder(sparse_output=False,drop='if_binary'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numerical_cols),
        ('multi_cat', multi_categorical_transformer, multi_categorical_cols),
        ('binary_cat', binary_categorical_transformer, binary_categorical_cols)
    ])


    # Define features and target
    X_train = train_df.drop(columns=['aprv_flag'])
    y_train = train_df['aprv_flag']
    X_test = test_df.drop(columns=['aprv_flag'])
    y_test = test_df['aprv_flag']

    # Apply preprocessing pipeline
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # build a dataframe after preprocessing data
    col_names=preprocessor.get_feature_names_out()
    X_train_processed=pd.DataFrame(X_train_processed,columns=col_names)
    X_test_processed=pd.DataFrame(X_test_processed,columns=col_names)
    
    return X_train_processed, y_train, X_test_processed, y_test

if __name__ == "__main__":
    from data_loading import load_data
    train_data_path="D:/Pycharm/AutoLoanModel/Training Dataset.csv"
    test_data_path="D:/Pycharm/AutoLoanModel/Testing Dataset.csv"
    df_train, df_test = load_data(train_data_path,test_data_path)
    X_train_processed, y_train, X_test_processed, y_test=preprocess_data(df_train, df_test)
    print("Data Preprocessed Successfully")
