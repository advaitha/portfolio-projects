# Exploring the data
df.dtypes   #checking the dtypes
df['column'] = df['column'].astype('float') #change data type to float
df.isnull().sum()   #Get the number of null values in the dataframe
df[['column1', 'column2', 'column3']].isnull().sum()  #check for null in specific columns
# Keep only rows where column1, column2, and column3 are not null
df_no_missing = df[df['column1'].notnull() & 
          df['column2'].notnull() & 
          df['column3'].notnull()]




# Dropping multiple columns
# Create a list of redundant column names to drop
to_drop = ["category_desc", "locality", "region", "vol_requests", "created_date"]
# Drop those columns from the dataset
volunteer_subset = volunteer.drop(to_drop, axis =1)



# Standardizing the numerical features
# To check variance of all columns in a dataframe
df.var()

# Log normalization is used when there is high variance. Transformation onto a scale
# that approximates normality
import numpy as np
df['new_column_name'] = np.log(df['old_column_name'])

# Feature scaling when variables are on different scales. Useful when we want to model
# with linear characteristics. Mean = 0, variance = 1
# The output of fit_transform is a numpy array. Here we are converting it back into
# a dataframe
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)

# A common method in case of numerical features is to take average or some
# other derived column from the set of numerical features to reduce dimensionality
numeric_columns = ["column1", "column2", "column3"]
df['mean_column'] = df.apply(lambda row: row[numeric_columns].mean(), axis = 1)



# Extracting features from dates
# convert the date to pandas datetime object
df["date_converted"] = pd.to_datetime(df["date"])
df['month'] = df['date_converted'].apply(lambda row:row.month)
df['day'] = df['date_converted'].apply(lambda row:row.day)
df['year'] = df['date_converted'].apply(lambda row:row.year)



# Encoding categorical variables
# Encoding binary variables
# using pandas
df['encoded_column'] = df['binary_categorical_column'].apply(lambda val:
                                                      1 if val == 'y' else 0)
# using sklearn will be useful if we are creating pipelines and to use
# the same functionality on unseen or test data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['encoded_column'] = le.fit_transform(df['binary_categorical_column'])

# one-hot encoding - More than two levels
pd.get_dummies(df['column_to_encode'])




# Extracting features from text
# Extracting digits from string
import re
my_string = "temperature:75.6 F"
pattern = re.compile("\d+\.\d+") # \d means we want to extract digits, + means we want to grab as many as possible
temp = re.match(pattern, my_string)
print(float(temp.group(0))  #Extract using group

# Vectorizing text using tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer()
text_tfidf = tfidf_vec.fit_transform(df['name_of_text_column'])
X_train, X_test, y_train, y_test = train_test_split(text_tfidf.toarray(), targets, stratify = targets) #to_arry is required to get the data in the required format for sklearn



# Feature selection
# what is variance threshold? univariate statistical tests?
# Reduce the noise by removing the unnecessary features
# iterative process
# Remove correlated features for linear models
df.corr() 
correlated_features_to_drop = ['column1', 'column2', 'column3']
df_without_correlated_features = df.drop(correlated_features_to_drop, axis =1) # Removing correlated features is an iterative process.



#combining two dataframes on columns (equal rows)
df_combined = pd.concat([df1,df2], axis=1)


# Removing text vectors based on weightage
def return_weights(vocab, vector, vector_index):
    zipped = dict(zip(vector[vector_index].indices, vector[vector_index].data))
    return {vocab[i]:zipped[i] for i in vector[vector_index].indices}
print(return_weights(vocab, text_tfidf,3))

# Add in the rest of the parameters
def return_weights(vocab, original_vocab, vector, vector_index, top_n):
    zipped = dict(zip(vector[vector_index].indices, vector[vector_index].data))
    
    # Let's transform that zipped dict into a series
    zipped_series = pd.Series({vocab[i]:zipped[i] for i in vector[vector_index].indices})
    
    # Let's sort the series to pull out the top n weighted words
    zipped_index = zipped_series.sort_values(ascending=False)[:top_n].index
    return [original_vocab[i] for i in zipped_index]

# Print out the weighted words
print(return_weights(vocab, tfidf_vec.vocabulary_, text_tfidf,8, 3))


def words_to_filter(vocab, original_vocab, vector, top_n):
    filter_list = []
    for i in range(0, vector.shape[0]):
    
        # Here we'll call the function from the previous exercise, and extend the list we're creating
        filtered = return_weights(vocab, original_vocab, vector, i, top_n)
        filter_list.extend(filtered)
    # Return the list in a set, so we don't get duplicate word indices
    return set(filter_list)

# Call the function to get the list of word indices
filtered_words = words_to_filter(vocab, tfidf_vec.vocabulary_, text_tfidf, 3)

# By converting filtered_words back to a list, we can use it to filter the columns in the text vector
filtered_text = text_tfidf[:, list(filtered_words)]

# Split the dataset according to the class distribution of category_desc, using the filtered_text vector
train_X, test_X, train_y, test_y = train_test_split(filtered_text.toarray(), y, stratify=y)

# Fit the model to the training data
nb.fit(train_X,train_y)

# Print out the model's accuracy
print(nb.score(test_X,test_y))



# Dimensionality reduction
# PCA - Linear transformation to uncorrelated space
# Captures as much variance as possible in each component
# we can use this when traditional dimension reduction techniques are not reducing the features
from sklearn.decomposition import PCA
pca = PCA()
df_pca = pca.fit_transform(df)
print(pca.explained_variance_ratio_)



# Train Test split
# use stratify in case of imbalance in class labels
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, stratify = y,
                                                    test_size = 0.2, random_state = 123)


#KNN is in linear space?
# Most models have a score function which can used to evaluate the performance
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn.score(X_test, y_test)



# Write a pattern to extract numbers and decimals
def return_mileage(length):
    pattern = re.compile(r"\d+\.\d+")
    
    # Search the text for matches
    mile = re.match(pattern, length)
    
    # If a value is returned, use group(0) to return the found value
    if mile is not None:
        return float(mile.group(0))
        
# Apply the function to the Length column and take a look at both columns
hiking["Length_num"] = hiking["Length"].apply(lambda row: return_mileage(row))
print(hiking[["Length", "Length_num"]].head())



_______________________________________________________________________________________


# UFO Sightings

# Check how many values are missing in the length_of_time, state, and type columns
print(ufo[['length_of_time', 'state', 'type']].isna().sum())

# Keep only rows where length_of_time, state, and type are not null
ufo_no_missing = ufo[ufo['length_of_time'].notnull() & 
          ufo['state'].notnull() & 
          ufo['type'].notnull()]

# Print out the shape of the new dataset
print(ufo_no_missing.shape)






