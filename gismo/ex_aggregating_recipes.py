import pickle
import os#

GISMO_INPUTS_PATH = os.path.abspath(".\\gismo\\checkpoints\\")

def load_all_ground_truth_sets(gismo_inputs_path):
    val_comments_subs_path = os.path.join(gismo_inputs_path, "val_comments_subs.pkl")
    with open(val_comments_subs_path, "rb") as val_comments_file:
        val_comments = pickle.load(val_comments_file)

    test_comments_subs_path = os.path.join(gismo_inputs_path, "test_comments_subs.pkl")
    with open(test_comments_subs_path, "rb") as test_comments_file:
        test_comments = pickle.load(test_comments_file)

    train_comments_subs_path = os.path.join(gismo_inputs_path, "train_comments_subs.pkl")
    with open(train_comments_subs_path, "rb") as train_comments_file:
        train_comments = pickle.load(train_comments_file)

    return train_comments, test_comments, val_comments

def aggregate_datasets(test_set, train_set, val_set):
    set = test_set + train_set + val_set
    return set


gismo_inputs_path = os.path.abspath(GISMO_INPUTS_PATH)
train_set, test_set, val_set = load_all_ground_truth_sets(gismo_inputs_path)

set = aggregate_datasets(train_set, test_set, val_set)

print("DONE")

