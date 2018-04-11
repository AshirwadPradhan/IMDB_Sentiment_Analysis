import re

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as bs
from nltk.corpus import stopwords
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
import pickle


class SentAna():
    '''
    Sentiment Analysis using NLTK library and BOW model
    '''
    _train_data = None
    _test_data = None


    def __init__(self):
        pass

    def import_train_dataset(self):
        '''
        imports the dataset from the path specified
        INPUT : tsv file with reviews having the header as 'review'
        '''
        # importing the dataset
        try:
            _train_data = pd.read_csv(
                'data/train.tsv', delimiter='\t',  quoting=3)
            # quoting = 3 tells python to ignore doubled quotes
            if _train_data is not None:
                return _train_data
            else:
                print('Cannot import dataset. Train Dataset not found')
        except FileNotFoundError:
            print('Dataset not found. Please check the path')
        except Exception as e:
            print('Unexpected Error')
            print(e.__traceback__)

    def import_test_dataset(self):
        '''
        imports the dataset from the path specified
        INPUT : tsv file with reviews having the header as 'review'
        '''
        # importing the dataset
        try:
            _test_data = pd.read_csv(
                'data/test.tsv', delimiter='\t',  quoting=3)
            # quoting = 3 tells python to ignore doubled quotes
            if _test_data is not None:
                return _test_data
            else:
                print('Cannot import dataset. Test Dataset not found')
        except FileNotFoundError:
            print('Dataset not found. Please check the path')
        except Exception as e:
            print('Unexpected Error')
            print(e.__traceback__)


    def text_preprocess(self, data):
        '''
        Text preprocessing function 
        1. Cleans HTML tags if any
        2. Removes punctuation and numbers in the text
        3. Tokenization
        4. Removing Stopwords

        '''
        #train_data = self.import_dataset()
        # review_text = train_data.iloc[:, 2:2].values
        # this doesn't work in case of text as it returns narray but we
        # need str for bs4

        review_text = data['review']
        for i in range(len(review_text)):

            if(i+1) % 1000 == 0:
                print('Completed pre-processing of {} out of {}\n'.format(
                    i+1, len(review_text)))
            # cleaning the HTML tags in the dataset
            review_text[i] = bs(str(review_text[i]), 'lxml').get_text()

            # cleaning the numbers and punctuations in the text
            review_text[i] = re.sub('[^a-zA-z]', ' ', review_text[i])

            # converting the text in to lowercase
            # coverting the text into tokens
            review_text[i] = review_text[i].lower()
            review_text[i] = review_text[i].split()

            # removing the stopwords from the text
            # converting stopwords into sets for efficiency
            stopwords_sets = set(stopwords.words('english'))
            review_text[i] = [w for w in review_text[i]
                              if not w in stopwords_sets]

            # covert the list of words back into sentences
            review_text[i] = " ".join(review_text[i])
        return review_text

    def count_vectorizer(self, review_text, max_features=5000):
        '''
        Bag of words model features creation
        '''
        # creating vocabulary of 5000 most occuring features
        most_words_dict = OrderedDict()
        for i in range(len(review_text)):
            if(i+1) % 3 == 0:
                print('Completed making dictionary of {} out of {}\n'.format(
                    i+1, len(review_text)))

            text_list = review_text[i].split()
            for word in text_list:
                if word not in most_words_dict.keys():
                    most_words_dict[word] = 1
                else:
                    most_words_dict[word] = most_words_dict[word] + 1
        # sorting the dict based on frequency
        most_words_dict = OrderedDict(
            sorted(most_words_dict.items(), key=lambda t: t[1]))

        # Removing least occuring word
        # Shaping to max_features
        if len(most_words_dict) > max_features:
            extras = len(most_words_dict) - max_features
            for i in range(extras):
                most_words_dict.popitem(last=False)

        # getting the bag of words model
        bow_model = {}
        vec_words = []
        count = 0
        for n, keys in enumerate(most_words_dict.keys()):
            if (n+1) % 1000 == 0:
                print('Creating BOW model for keys {} out of {}\n'.format(
                    n+1, len(most_words_dict)))
            vec_words = []
            for i in range(len(review_text)):
                if (i+1) % 1000 == 0:
                    print('BOW processing for key {} review #{} out of {}\n'.format(
                       n+1, i+1, len(review_text)))
                text_list = review_text[i].split()
                for words in text_list:
                    if keys == words:
                        count += 1
                vec_words.append(count)
                count = 0
            bow_model[keys] = vec_words

        feature_vectors = []
        for val in bow_model.values():
            feature_vectors.append(val)
        feature_vectors = np.asarray(feature_vectors)
        feature_vectors = feature_vectors.T
        print(feature_vectors.shape)
        return feature_vectors

    def classifier_model(self, feature_vectors):
        '''
        Fits the training set to the model
        and after fitting pickles the classifier model
        '''
        train_data = self.import_train_dataset()
        classifier = RandomForestClassifier(n_estimators=100)

        print('Starting fit\n')
        classifier = classifier.fit(feature_vectors, train_data['sentiment'])
        print('Ending Fit')

        print('Start pickling process')
        pickle.dump(classifier, open('model1.sav','wb'))
        print('Done pickling the model')

    def classifier_predict(self):
        '''
        Predicts the sentiments of the reviews and 
        save it in tsv format
        '''
        print('Training Started...')
        test_data = self.import_test_dataset()
        cleaned_text = self.text_preprocess(test_data)
        feature_vectors = self.count_vectorizer(cleaned_text)

        print('Loading ML model...')
        try:
            classf_model = pickle.load(open('model_random_forest.sav', 'rb'))
        except Exception:
            print('Error Loading Model')

        print('Predicting....')
        predicted_sentiments = classf_model.predict(feature_vectors)
        print('Prediction Completed!')

        print('Saving the predictions')
        output = pd.DataFrame( data={"id":test_data["id"], "sentiment":predicted_sentiments})
        output.to_csv( "BOW_RF_pred.csv", index=False, quoting=3, escapechar='\\' )
        print('Testing Completed!')
        



if __name__ == '__main__':

    o_sentana = SentAna()

    #***********TRAINING CODE************************************
    # train_data = o_sentana.import_train_dataset()
    # cleaned_text = o_sentana.text_preprocess(train_data)
    # feature_vectors = o_sentana.count_vectorizer(cleaned_text)
    # o_sentana.classifier_model(feature_vectors)
    #*************************************************************
    
    #***********TESTING CODE*****************
    o_sentana.classifier_predict()


    #*************UNIT TEST DATA**************************************************
    # print(cleaned_text[0])
    # fv = o_sentana.count_vectorizer(
    #   ['This is a a a a triple triple word', 'Wow This is a a a a triple word'])
    # o_sentana.classifier_model(fv)
    #print(np.asarray([1, 2], [2, 3]))
    #*****************************************************************************