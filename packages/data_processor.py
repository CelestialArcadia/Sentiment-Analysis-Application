from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import pandas as pd

def prepare_data(path_to_data, encoding = "latin-1"):
    """
        @params:
            - path_to_data: the path to the data
            - encoding: the encoding format to be used
            
        @return:
            - dictionary with tfollowing keys:
                - text: the actual text message
                - label: the label associated to that text message
    """
    # Read data from path
    data = pd.read_csv(path_to_data, encoding = encoding)
    
    # Encode labels
    data['label'] = data['v1'].map({'ham': 0, 'spam': })
    
    x = data['v2']
    y = data['label']
    
    return {'text': x, 'label': y}

def create_train_test_data(x, y, test_size, random_state):
    """
        @params:
            - x: the feature for training
            - y: the labels
            - test_size: the percentage of testing size
            - random_state: the random state
        @return:
            - a dictionary containing training data, testing data and their corresponding labels
    """
    count_Vector = CountVectorizer()
    x = count_Vector.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = random_state)
    
    return {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}, cv