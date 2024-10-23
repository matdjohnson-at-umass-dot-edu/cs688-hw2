
from dataset_holder import DatasetHolder
from param_holder import ParamHolder
from hw2_runner import Homework2Runner


dataset_holder = DatasetHolder()
dataset_holder.read_file_contents()
param_holder = ParamHolder()
param_holder.load_params()
crf_runner = Homework2Runner(
    dataset_holder.get_train_text(),
    dataset_holder.get_train_images(),
    dataset_holder.get_train_words(),
    dataset_holder.get_test_text(),
    dataset_holder.get_test_images(),
    dataset_holder.get_test_words(),
    dataset_holder.get_letter_index(),
    param_holder.get_feature_params(),
    param_holder.get_feature_grad(),
    param_holder.get_transition_params(),
    param_holder.get_transition_grad()
)
crf_runner.run()

