from Summary_evaluation.RedundancyPrediction.Preprocessing_Redundancy import Preprocess
from Summary_evaluation.Classifiers import Classifiers

def run_redundancy_extra_feature():
    extra = ["uni_count", "bi_count", "pos_tags", "tri_count", "word_count"]
    preprocess_training_data = Preprocess("../../../Train_Data.csv - Sheet1.csv", remove_digit=True, remove_punctuation=True,
                                          extra_feature=extra[0:2])
    training_data = preprocess_training_data.preprocess(1)
    preprocess_testing_data = Preprocess("../../../Test_Data.csv - Sheet1.csv", remove_digit=True, remove_punctuation=True,
                                         extra_feature=extra[0:2])
    testing_data = preprocess_testing_data.preprocess(1)
    MLP_classifier = Classifiers(training_data=training_data, testing_data=testing_data, classifier=["mlp"])
    training_error, testing_error, testing_pearson, training_pearson = MLP_classifier.build_model()


run_redundancy_extra_feature()