from sklearn.preprocessing import StandardScaler, LabelEncoder
def preprocess_data(df):
    #split data in to features and target
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    
    # Encode target (M = 1, B = 0)
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Standardize features (VERY important for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler