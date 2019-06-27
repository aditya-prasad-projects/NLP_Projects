from Summary_evaluation.Classifiers import Classifiers
from Summary_evaluation.FluencyPrediction.Preprocessing_Fluency import Preprocess_Fluency



def run_fluency():
    preprocess_training_data = Preprocess_Fluency("Train_Data.csv - Sheet1.csv", remove_digit=True, remove_punctuation=True, extra_features=[])
    training_data = preprocess_training_data.preprocess()
    preprocess_testing_data = Preprocess_Fluency("Test_Data.csv - Sheet1.csv", remove_digit=True, remove_punctuation=True, extra_features=[])
    testing_data = preprocess_testing_data.preprocess()
    MLP_classifier = Classifiers(training_data=training_data, testing_data=testing_data, classifier=["mlp"])
    MLP_classifier.build_model()

run_fluency()