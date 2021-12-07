# Packages
import joblib

# Get customized functions from library
import packages.data_processor as data_proc
import packages.model_trainer as model_trainer

# Data path
path_to_data = './data/spam.csv'

# Data preparation
prepared_data = data_proc.prepare_data(path_to_data, encoding = "latin-1")

# Create train - test split
train_test_data, vectorizer = data_proc.create_train_test_data(prepare_data['text'], prepared_data['label'], 0.33, 2021)

# Run training
model = model_trainer.run_model_training(train_test_data['x_train'], train_test_data['x_test'], train_test_data['y_train'], train_test_data['y_test'])

# Save the trained model and vectorizer
joblib.dump(model, './models/spam_detector_model.pkl')
joblib.dump(vectorizer, open("./vectors/vectorizer.pickle","wb"))