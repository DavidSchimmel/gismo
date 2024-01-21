import os
import pickle
import torch

from data_loader import load_data

TRAIN_COMMENTS_PATH = os.path.abspath("./gismo/checkpoints/train_comments_subs.pkl")

print(TRAIN_COMMENTS_PATH)

if __name__ == "__main__":
    # with open (TRAIN_COMMENTS_PATH, "rb") as train_comments_file:
    #     train_comments = pickle.load(train_comments_file)

    # print("test")


    data_path = os.path.abspath("C:\\UM\\Master\\FoodRecommendations\\literature_models\\GISMo\\gismo\\gismo\\checkpoints")
    precalc_substitutailities_path = os.path.join(data_path, "precalculated_substitutabilities", "cos_similarities.pt")
    ingr_name_2_subst_col_path = os.path.join(data_path, "precalculated_substitutabilities", "ingr_2_col.pkl")
    sample_2_subst_row_path = os.path.join(data_path, "precalculated_substitutabilities", "sample_2_row.pkl")


    graph, train_dataset, val_dataset, test_dataset, ingrs, node_count2id, node_id2name, node_id2count, recipe_id2counter, filter_ingredient = load_data(
        400,
        43,
        False,
        "precalc",
        False,
        0.5,
        False,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        data_path = data_path,
    )

    samples = train_dataset.__getitem__(5)

    print("done")