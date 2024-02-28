import os
import json
import pickle
import csv
import yaml
from inv_cooking.datasets.vocabulary import Vocabulary

GISMO_INPUTS_PATH = os.path.abspath("./gismo/checkpoints/")


with open(os.path.abspath("./gismo/conf/config.yaml"), "r") as yamlfile:
    data = yaml.load(yamlfile, Loader=yaml.FullLoader)
    RESULTS_PATH = os.path.abspath(data['results_path'])
print(data)

# RESULTS_PATH = os.path.abspath("C:\\Users\\David\\project\\gismo\\out\\first_tests\\baseline results 500 gen\\")
NAMED_RESULTS_AND_GT_FILE_NAME = "named_ranked_recommendations.csv"
DO_SAVE_LIST_OF_LISTS_TO_CSV = True



def load_all_output_pickles(gismo_inputs_path):
    vocab_ingrs_path = os.path.join(gismo_inputs_path, "vocab_ingrs.pkl")
    with open(vocab_ingrs_path, "rb") as vocab_ingrs_file:
        vocab_ingrs = pickle.load(vocab_ingrs_file)

    # saving default dicts to use them elsewhere (as vocabs somehow needs a gimso module)
    with open(os.path.abspath("./idx_2_word.pkl"), "wb") as idx_2_word_file:
        pickle.dump(vocab_ingrs.idx2word, idx_2_word_file)
    with open(os.path.abspath("./word_2_idx.pkl"), "wb") as word_2_idx_file:
        pickle.dump(vocab_ingrs.word2idx, word_2_idx_file)

    with open(
            os.path.join(RESULTS_PATH, "val_ranks_out_modified.pkl"),
            "rb") as gismo_validation_output_file:
        gismo_output = pickle.load(gismo_validation_output_file)


    val_comments_subs_path = os.path.join(gismo_inputs_path, "val_comments_subs.pkl")
    with open(val_comments_subs_path, "rb") as val_comments_file:
        val_comments = pickle.load(val_comments_file)


    return gismo_output, val_comments, vocab_ingrs

def get_ranked_recommendations(ranked_recommendations_file_path):
    ranked_recommendations = []

    with open(ranked_recommendations_file_path, 'r') as ranked_recommendations_file:
        csv_reader = csv.reader(ranked_recommendations_file, delimiter=' ')
        for row in csv_reader:
            numbers = [int(num) for num in row]  # Convert each number to an integer
            ranked_recommendations.append(numbers)  # Append the array of numbers to the main array

    return ranked_recommendations

def get_recommendations_with_ground_truth(ranked_recommendations_w_GT_file_path):
    recommendations_w_GT = []

    with open(recommendations_w_GT_file_path, 'r') as recommendations_w_GT_file:
        csv_reader = csv.reader(recommendations_w_GT_file, delimiter=' ')
        for row in csv_reader:
            numbers = [int(num) for num in row]  # Convert each number to an integer
            recommendations_w_GT.append(numbers)  # Append the array of numbers to the main array

    return recommendations_w_GT

def zip_ground_truth_ranks_and_ranked_recommendations(ranked_recommendations, recommendations_w_GT):
    appended_ranked_recommendations = [
        [substitutions[0]] + ground_truths[1:3] + substitutions[1:]
        for substitutions, ground_truths
        in zip(ranked_recommendations, recommendations_w_GT)
    ]
    return list(appended_ranked_recommendations)

def load_vocab_to_ids(results_path):
    # load the translations for the ingredient indeces and labels
    id2name_path = os.path.join(results_path, "node_id2name.pkl")
    count2id_path = os.path.join(results_path, "node_count2id_path.pkl")

    with open(id2name_path, "rb") as id2name_file:
        id2name = pickle.load(id2name_file)


    with open(count2id_path, "rb") as count2id_file:
        count2id = pickle.load(count2id_file)

    return id2name, count2id

def replace_indeces_with_names(ingredient_lists_list):
    # translate the results file
    ranks = [result[2] for result in ingredient_lists_list]
    named_ranked_substitutions = [[id2name[count2id[ingredient_count]] for ingredient_count in ingredient_list] for ingredient_list in ingredient_lists_list]

    named_ranked_substitutions_w_ranks = [ingredients[:2] + [rank] + ingredients[3:] for ingredients, rank in zip(named_ranked_substitutions, ranks)]
    # named_ranked_substitutions = [[id2name[count2id[cnt]]] for ingredient_list in ingredient_lists_list]
    return named_ranked_substitutions_w_ranks

def save_list_of_lists_to_csv(file_path, data_list):
    with open(file_path, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(data_list)


# def convert_vocab_to_csv(vocab_path, converted_vocab_path_target):
#     """This function is not worked out"""
#     with open(vocab_path, "rb") as val_comments_file:
#         vocab = pickle.load(val_comments_file)

#     with open(converted_vocab_path_target, "w") as csv_file:
#         print("hello")

if __name__ == "__main__":
    gismo_output, val_comments, vocab_ingrs = load_all_output_pickles(GISMO_INPUTS_PATH)

    # convert_vocab_to_csv(os.path.join(GISMO_INPUTS_PATH, "vocab_ingrs.pkl"), os.path.join(GISMO_INPUTS_PATH, "vocab_ingrs.csv"))

    ranked_recommendations_file_path = os.path.join(RESULTS_PATH, "val_ranks_full.txt")
    ranked_recommendations = get_ranked_recommendations(ranked_recommendations_file_path)

    recommendations_w_GT_file_path = os.path.join(RESULTS_PATH, "val_ranks.txt")
    recommendations_w_GT = get_recommendations_with_ground_truth(recommendations_w_GT_file_path)

    ranked_recommendations_w_GTs = zip_ground_truth_ranks_and_ranked_recommendations(ranked_recommendations, recommendations_w_GT)

    # ranked_recommendations_w_GTs = sorted(ranked_recommendations_w_GTs, key=lambda x: x[2], reverse=True)

    id2name, count2id = load_vocab_to_ids(RESULTS_PATH)

    named_ranked_recommendations = replace_indeces_with_names(ranked_recommendations_w_GTs)

    named_ranked_recommendations_file_path = os.path.join(RESULTS_PATH, NAMED_RESULTS_AND_GT_FILE_NAME)

    if DO_SAVE_LIST_OF_LISTS_TO_CSV:
        save_list_of_lists_to_csv(named_ranked_recommendations_file_path, named_ranked_recommendations)

    print("DONE")

## to get the recommendations with values, the best place to start might actually be the get_model_output (I assume we want the sims)