import packages.data_processor as data_proc
import streamlit as stream_l
import joblib

# Model loading
spam_clf = joblib.load(open('./models/spam_detector_model.pkl','rb'))

# Vectorize loading
vectorizer = joblib.load(open('./vectors/vectorizer.pickle','rb'))

### MAIN FUNCTION ###
def main(title = "Streamlit Text classification App".upper()):
    
    stream_l.markdown("<h1 style ='text-align: center; front-size: 65px; color: #468284;'>{}</h1>".format(title), unsafe_allow_html = True)
    stream_l.image("./images/message-image.jpeg")
    info = ''
    
    with stream_l.expander("1. Check if your text is a spam or ham"):
        text_message = stream_l.text_input("Please enter your text")
        
        if stream_l.button("Predict"):
            prediction = spam_clf.predict(vectorizer.transform([text_message]))
            
            if(prediction[0] == 0):
                info = 'Ham'
            else:
                info = 'Spam'
            stream_l.success('Prediction: {}'.format(info))
            
if __name__ == "__main__":
    main()