from Summary_evaluation.Classifiers import Classifiers
from Summary_evaluation.FluencyPrediction.Preprocessing_Fluency import Preprocess_Fluency
import numpy as np



def run_fluency_extra_feature():
    extra = np.array(["RIX", 'Kincaid', 'ARI', 'Coleman-Liau', 'GunningFogIndex','LIX', 'SMOGIndex', 'DaleChallIndex'])
    indices = [4,7]
    preprocess_training_data = Preprocess_Fluency("Train_Data.csv - Sheet1.csv", remove_digit=True, remove_punctuation=True, extra_features = extra[indices])
    training_data = preprocess_training_data.preprocess()
    preprocess_testing_data = Preprocess_Fluency("Test_Data.csv - Sheet1.csv", remove_digit=True, remove_punctuation=True, extra_features= extra[indices])
    testing_data = preprocess_testing_data.preprocess()
    MLP_classifier = Classifiers(training_data=training_data, testing_data=testing_data, classifier=["mlp"])
    training_error, testing_error, testing_pearson, training_pearson = MLP_classifier.build_model()
    with open("./4.2.2.txt", "w") as f:
        f.write("training_error = " + str(training_error) + "\n")
        f.write("testing_error = " + str(testing_error) + "\n")
        f.write("training_pearson = " + str(training_pearson) + "\n")
        f.write("testing_pearson = " + str(testing_pearson) + "\n")

run_fluency_extra_feature()