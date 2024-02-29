# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import json
import os
import pickle
import random
import numpy as np
import dgl
import dgl.function as fn
import dgl.ops as ops
import torch
import torch.utils.data as data

from inv_cooking.datasets.vocabulary import Vocabulary


def load_edges(
    dir_, node_id2count, nnodes, add_self_loop, device, normalize=True,
):
    sources, destinations, weights, types = [], [], [], []

    with open(os.path.join(dir_, "edges_191120.csv"), "r") as edges_file:
        csv_reader = csv.DictReader(edges_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            node1, node2 = int(row["id_1"]), int(row["id_2"])
            node1_cnt, node2_cnt = node_id2count[node1], node_id2count[node2]

            edge_type = row["edge_type"]
            if "ingr-ingr" in row["edge_type"]:
                edge_type = 1
                score = float(row["score"])
            elif "rec-ingr" in row["edge_type"]:
                edge_type = 5
                score = 1
            elif "rec-act" in row["edge_type"]:
                edge_type = 6
                score = 1
            elif "ingr-act" in row["edge_type"]:
                edge_type = 7
                score = 1
            elif "step-step" in row["edge_type"]:
                edge_type = 8
                score = 1
            elif "step-act" in row["edge_type"]:
                edge_type = 9
                score = 1
            elif "step-rec" in row["edge_type"]:
                edge_type = 10
                score = 1
            elif "ingr-step" in row["edge_type"]:
                edge_type = 11
                score = 1
            elif "ingr-fcomp" in row["edge_type"]:
                edge_type = 2
                score = 1
            elif "ingr-dcomp" in row["edge_type"]:
                edge_type = 3
                score = 1

                # TODO add remaining edge types

            sources.append(node1_cnt)
            destinations.append(node2_cnt)
            weights.append(score)
            types.append(edge_type)

            # make it symmetric
            sources.append(node2_cnt)
            destinations.append(node1_cnt)
            weights.append(score)
            types.append(edge_type)

    # add self-loop
    if add_self_loop:
        for node in range(nnodes):
            sources.append(node+1)
            destinations.append(node+1)
            weights.append(1)
            types.append(4)
        print("self-loop is added to all nodes.")

    sources = torch.tensor(sources)
    destinations = torch.tensor(destinations)
    weights = torch.tensor(weights)
    types = torch.tensor(types)

    if torch.cuda.is_available():
        sources = sources.to(device)
        destinations = destinations.to(device)
        weights = weights.to(device)
        types = types.to(device)

    graph = dgl.graph((sources, destinations))
    graph.edata["w"] = weights
    graph.edata["t"] = types

    # symmetric normalization
    if normalize:
        print("noramlizing graph")
        in_degree = ops.copy_e_sum(graph, graph.edata["w"])
        in_norm = torch.pow(in_degree, -0.5)
        out_norm = torch.pow(in_degree, -0.5).unsqueeze(-1)
        graph.ndata["in_norm"] = in_norm
        graph.ndata["out_norm"] = out_norm
        graph.apply_edges(fn.u_mul_v("in_norm", "out_norm", "n"))
        graph.edata["w"] = graph.edata["w"] * graph.edata["n"].squeeze()
        print("noramlizing graph done")

    return graph


def load_nodes(dir_):
    """Load graph nodes and set up several dictionaries to quickly match node labels and indeces for processing in GISMo.

    Args:
        dir_ (os.path): Path to the directory where nodes and edges live. Nodes file should be named "nodes_191120.csv" as in flavorgraph.

    Returns:
        dict: collection of various mappings.
            ingredients_cnt:
            compounds_cnt:
    """
    node_id2name = {}
    node_name2id = {}
    node_id2type = {}
    ingredients_cnt = []
    recipe_cnt = []
    action_cnt = []
    step_cnt = []
    compounds_cnt = []
    node_id2count = {}
    node_count2id = {}
    counter = 1  # start with 1 to reserve 0 for padding
    dir_ = os.path.expanduser(dir_)
    with open(os.path.join(dir_, "nodes_191120.csv"), "r") as nodes_file:
        csv_reader = csv.DictReader(nodes_file)
        for row in csv_reader:
            node_id = int(row["node_id"])
            node_type = row["node_type"]
            node_id2name[node_id] = row["name"]
            node_name2id[row["name"]] = node_id
            node_id2type[node_id] = node_type
            # TODO add remaining node types (maybe)
            if "ingredient" in node_type:
                ingredients_cnt.append(counter)
            elif "recipe" in node_type:
                recipe_cnt.append(counter)
            elif "action" in node_type:
                action_cnt.append(counter)
            elif "step" in node_type:
                step_cnt.append(counter)
            else:
                compounds_cnt.append(counter)
            node_id2count[node_id] = counter
            node_count2id[counter] = node_id
            counter += 1
    nnodes = len(node_id2name)
    print("#nodes:", nnodes)
    print("#ingredient nodes:", len(ingredients_cnt))
    print(f"#recipe nodes: {len(recipe_cnt)}")
    print(f"#action nodes: {len(action_cnt)}")
    print(f"#step nodes: {len(step_cnt)}")
    print("#compound nodes:", len(compounds_cnt))
    return (
        node_id2count,
        node_count2id,
        node_id2name,
        node_name2id,
        ingredients_cnt,
        node_id2name,
        nnodes,
    )

def node_count2name(count, node_count2id, node_id2name):
    return node_id2name[node_count2id[count]]


def load_graph(add_self_loop, dir_, device):
    dir_ = os.path.expanduser(dir_)
    (
        node_id2count,
        node_count2id,
        node_id2name,
        node_name2id,
        ingredients_cnt,
        node_id2name,
        nnodes,
    ) = load_nodes(dir_)
    graph = load_edges(dir_, node_id2count, nnodes, add_self_loop, device)
    return (
        graph,
        node_name2id,
        node_id2count,
        ingredients_cnt,
        node_count2id,
        node_id2name,
        nnodes
    )

def load_data(
    nr,
    max_context,
    add_self_loop,
    neg_sampling,
    data_augmentation,
    p_augmentation,
    filter,
    device,
    data_path: str,
):
    data_path = os.path.expanduser(data_path)
    flavor_graph_dir = os.path.join(data_path, "graph")
    substitution_dir = data_path
    (
        graph,
        node_name2id,
        node_id2count,
        ingredients_cnt,
        node_count2id,
        node_id2name,
        nnodes
    ) = load_graph(add_self_loop, flavor_graph_dir, device)
    ingr_vocabs = pickle.load(
        open(
            f"{substitution_dir}/vocab_ingrs.pkl",
            "rb",
        )
    )

    recipe_counter = 0
    recipe_id2counter = {}
    filtered_ing = {}

    train_dataset = SubsData(
        substitution_dir,
        "train",
        node_name2id,
        node_id2count,
        ingredients_cnt,
        nr,
        ingr_vocabs,
        max_context,
        recipe_counter,
        recipe_id2counter,
        neg_sampling,
        data_augmentation,
        nnodes,
        filtered_ing,
        filter,
        p_augmentation,
    )

    val_dataset = SubsData(
        substitution_dir,
        "val",
        node_name2id,
        node_id2count,
        ingredients_cnt,
        nr,
        ingr_vocabs,
        max_context,
        train_dataset.recipe_counter,
        train_dataset.recipe_id2counter,
        neg_sampling,
        data_augmentation,
        nnodes,
        train_dataset.filtered_ing,
        filter,
        p_augmentation=0.0
    )

    test_dataset = SubsData(
        substitution_dir,
        "test",
        node_name2id,
        node_id2count,
        ingredients_cnt,
        nr,
        ingr_vocabs,
        max_context,
        val_dataset.recipe_counter,
        val_dataset.recipe_id2counter,
        neg_sampling,
        data_augmentation,
        nnodes,
        val_dataset.filtered_ing,
        filter,
        p_augmentation=0.0
    )

    return (
        graph,
        train_dataset,
        val_dataset,
        test_dataset,
        ingredients_cnt,
        node_count2id,
        node_id2name,
        node_id2count,
        test_dataset.recipe_id2counter,
        test_dataset.filtered_ing
    )


class SubsData(data.Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        node_name2id: dict,
        node_id2count: dict,
        ingredients_cnt: list,
        nr: int,
        vocab: Vocabulary,
        max_context: int,
        recipe_counter: int,
        recipe_id2counter: dict,
        neg_sampling: str,
        data_augmentation: bool,
        nnodes: int,
        filtered_ing: dict,
        filter: bool,
        p_augmentation: float,
    ):
        self.substitutions_dir = os.path.join(data_dir, split + "_comments_subs.pkl")
        self.split = split
        self.dataset = []
        self.nr = nr
        self.data_augmentation = data_augmentation
        self.p_augmentation = p_augmentation
        self.max_context = max_context
        self.neg_sampling = neg_sampling
        self.filter = filter
        self.filtered_ing = filtered_ing
        # load ingredient voc
        self.ingr_vocab = vocab
        self.node_name2id = node_name2id
        self.node_id2count = node_id2count
        self.ingredients_cnt = ingredients_cnt
        self.set_ingredients_cnt = set(ingredients_cnt)
        # load dataset
        self.recipe_counter = recipe_counter
        self.recipe_id2counter = recipe_id2counter
        self.dataset_list = pickle.load(open(self.substitutions_dir, "rb"))
        self.dataset = self.context_full_examples(
            self.dataset_list, self.ingr_vocab, self.max_context
        )

        if self.split == "train" and "precalc" in self.neg_sampling:
            self.load_precalculated_substitutability(data_dir)

        self.nnodes = nnodes
        if self.split == "train":
            self.lookup_table = self.create_lookup_table(self.dataset)
        print("Number of datapoints in", self.split, self.dataset.shape[0])

    def create_lookup_table(self, dataset):
        lookup_table = torch.zeros(self.nnodes, self.nnodes)
        for ind in range(dataset.shape[0]):
            ing1, ing2 = dataset[ind][0].item(), dataset[ind][1].item()
            lookup_table[ing1, ing2] += 1
        return lookup_table

    def load_precalculated_substitutability(self, data_dir):
        # TODO: Load dict mapping ingredient (idx?) to a probability distribution for each ingredient to weight negative sampling, e.g. based on substitutability
        # TODO: make modular like the other options

        if self.split == "train":
            print("loading precalculated similarities")
            precalc_substitutailities_path = os.path.join(data_dir, "precalculated_substitutabilities", "cos_similarities.pt")
            ingr_name_2_subst_col_path = os.path.join(data_dir, "precalculated_substitutabilities", "ingr_2_col.pkl")
            sample_2_subst_row_path = os.path.join(data_dir, "precalculated_substitutabilities", "sample_2_row.pkl")

            self.precalc_substitutabilities = torch.load(precalc_substitutailities_path)
            with open(ingr_name_2_subst_col_path, "rb") as ingr_2_subst_col_file:
                ingr_name_2_subst_col = pickle.load(ingr_2_subst_col_file)
                # directly map the ingredient labels to the node counts used in the dataset to simplify lookup during negative sampling
                self.ingr_cnt_2_subst_col = {self.node_id2count[self.node_name2id[ingredient]]: col for ingredient, col in list(ingr_name_2_subst_col.items())}
            with open(sample_2_subst_row_path, "rb") as sample_2_subst_row_file:
                sample_2_subst_row_ = pickle.load(sample_2_subst_row_file)
                # replace the labels for the row mapping with ids as well.
                self.sample_2_subst_row = {}
                recipe_not_found_cnt = 0
                source_not_found_cnt = 0
                target_not_found_cnt = 0

                for recipe_id, source_target_dict in list(sample_2_subst_row_.items()):
                    if recipe_id not in self.recipe_id2counter:
                        recipe_not_found_cnt += 1
                        continue
                    mapped_recipe_cnt = self.recipe_id2counter[recipe_id]
                    if mapped_recipe_cnt not in self.sample_2_subst_row:
                        self.sample_2_subst_row[mapped_recipe_cnt] = {}

                    for source_label, target_dict in list(source_target_dict.items()):
                        if source_label not in self.node_name2id:
                            source_not_found_cnt += 1
                            continue
                        source_cnt = self.node_id2count[self.node_name2id[source_label]]
                        if source_cnt not in self.sample_2_subst_row[mapped_recipe_cnt]:
                            self.sample_2_subst_row[mapped_recipe_cnt][source_cnt] = {}

                        for target_label, row_idx in list(target_dict.items()):
                            if target_label not in self.node_name2id:
                                target_not_found_cnt += 1
                                continue
                            target_cnt = self.node_id2count[self.node_name2id[target_label]]
                            if target_cnt not in self.sample_2_subst_row[mapped_recipe_cnt][source_cnt]:
                                self.sample_2_subst_row[mapped_recipe_cnt][source_cnt][target_cnt] = row_idx
            print("loading precalculated similarities done")


    def context_full_examples(self, examples, vocabs, max_context):
        """
        Maps example ids (aka recipe ids) to an incremental id counting up from 0
        Parameters:
            examples: are the samples taken from the comments (train/test/val)
            max_context: upper limit on  number of ingredients from the ingredient list to be included in the sample
        """
        output = torch.full((len(examples), max_context + 3), 0)

        for ind, example in enumerate(examples):
            subs = example["subs"]
            id_ = example["id"]

            if id_ in self.recipe_id2counter:
                id_counter = self.recipe_id2counter[id_]
            else:
                id_counter = self.recipe_counter
                self.recipe_id2counter[id_] = self.recipe_counter
                self.recipe_counter += 1
            context = example["ingredients"][:max_context]
            # example["text"]
            r_name1 = vocabs.idx2word[vocabs.word2idx[subs[0]]][0]
            r_name2 = vocabs.idx2word[vocabs.word2idx[subs[1]]][0]

            context_ids = torch.empty(len(context))
            for ind_, ing in enumerate(context):
                context_ids[ind_] = self.node_id2count[
                    self.node_name2id[vocabs.idx2word[vocabs.word2idx[ing[0]]][0]]
                ]

            subs = torch.tensor(
                [
                    self.node_id2count[self.node_name2id[r_name1]],
                    self.node_id2count[self.node_name2id[r_name2]],
                ]
            )
            output[ind, 0:2] = subs
            output[ind, 2] = id_counter
            # output[ind, 3:len(context) + 3] = context_ids

            # Excluding ing1 from the context
            context_ids = context_ids[context_ids != output[ind, 0]]
            output[ind, 3:len(context_ids) + 3] = context_ids

            # if self.filter:
            if (output[ind, 2].item(), output[ind, 0].item()) not in self.filtered_ing:
                self.filtered_ing[(output[ind, 2].item(), output[ind, 0].item())] = []
            self.filtered_ing[(output[ind, 2].item(), output[ind, 0].item())].append(output[ind, 1].item())
        return output

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        if self.split == "train":
            pos_example = self.dataset[index, :].view(1, -1)

            if self.data_augmentation and self.p_augmentation > 0.0:
                p = np.random.rand()
                if p <= self.p_augmentation:
                    pos_example[0, 0], pos_example[0, 1] = pos_example[0, 1].clone(), pos_example[0, 0].clone()

            neg_examples = self.neg_examples(pos_example, self.nr, self.set_ingredients_cnt)
            pos_labels = torch.zeros(1, 1)
            neg_labels = torch.ones(len(neg_examples), 1)
            pos_batch = torch.cat((pos_example, pos_labels), 1)
            neg_batch = torch.cat((neg_examples, neg_labels), 1)
            ret = torch.cat((pos_batch, neg_batch), 0)

            return ret.long()

        elif self.split == "val" or self.split == "test":
            # if not self.filter:
            #     pos_example = self.dataset[index, :]
            #     all_indices = self.ingredients_cnt.copy()
            #     all_indices.remove(pos_example[1])
            #     neg_examples_1 = torch.tensor(all_indices).view(-1, 1)
            #     neg_examples_0 = (
            #         self.dataset[index, :][0].repeat(len(all_indices)).view(-1, 1)
            #     )
            #     neg_examples = torch.cat(
            #         (
            #             neg_examples_0,
            #             neg_examples_1,
            #             pos_example[2:].repeat(len(all_indices), 1),
            #         ),
            #         1,
            #     )
            #     return torch.cat((pos_example.view(1, -1), neg_examples), 0)
            # elif self.filter:
                # do not remove itself from the list to avoid conflicts in ingredient id and index

            pos_example = self.dataset[index, :]
            all_indices = self.ingredients_cnt.copy()
            # all_indices.remove(pos_example[1])
            filter_mask = torch.zeros(len(all_indices) + 1)
            should_be_filtered = self.filtered_ing[(pos_example[2].item(), pos_example[0].item())]
            if len(should_be_filtered) == 0:
                print("***")
            filter_mask[should_be_filtered] = -100000

            neg_examples_1 = torch.tensor(all_indices).view(-1, 1)
            neg_examples_0 = (
                self.dataset[index, :][0].repeat(len(all_indices)).view(-1, 1)
            )

            neg_examples = torch.cat(
                (
                    neg_examples_0,
                    neg_examples_1,
                    pos_example[2:].repeat(len(all_indices), 1),
                ),
                1,
            )
            return torch.cat((pos_example.view(1, -1), neg_examples), 0), filter_mask

    @staticmethod
    def collate_fn(data):
        n_examples = data[0].shape[0]
        batch = torch.zeros(len(data) * data[0].shape[0], data[0].shape[1])
        for ind, sub_batch in enumerate(data):
            batch[ind * n_examples : (ind + 1) * n_examples, :] = sub_batch
        if torch.cuda.is_available():
            batch = batch.cuda()
        return batch.long()

    @staticmethod
    def collate_fn_val_test(data):
        # no filter
        # batch = torch.zeros(len(data) * data[0].shape[0], data[0].shape[1])
        # for ind, sub_batch in enumerate(data):
        #     batch[ind * data[0].shape[0] : (ind + 1) * data[0].shape[0], :] = sub_batch
        # if torch.cuda.is_available():
        #     batch = batch.cuda()

        # filter
        batch = torch.zeros(len(data) * data[0][0].shape[0], data[0][0].shape[1])
        mask = torch.zeros(len(data), data[0][0].shape[0])

        for ind, sub_batch in enumerate(data):
            batch[ind * data[0][0].shape[0] : (ind + 1) * data[0][0].shape[0], :] = sub_batch[0]
            mask[ind] = sub_batch[1]
        if torch.cuda.is_available():
            batch = batch.cuda()
            mask = mask.cuda()
        return batch.long(), mask.long()

    def neg_examples(self, example, nr, ingredients_cnt):
        neg_batch = torch.zeros(nr, 2)
        if self.filter:
            should_be_filtered = set(self.filtered_ing[(example[0, 2].item(), example[0, 0].item())])
            filtered_entities = self.set_ingredients_cnt - should_be_filtered
            random_entities = torch.tensor(random.sample(filtered_entities, nr))
        elif self.neg_sampling == "regular":
            random_entities = torch.tensor(random.sample(ingredients_cnt, nr))
        elif self.neg_sampling == "smart":
            context = set(example.view(-1).cpu().numpy())
            ingredients_cnt_wo_context = self.set_ingredients_cnt - context
            random_entities = torch.tensor(random.sample(ingredients_cnt_wo_context, nr))
        elif self.neg_sampling == "smart2" or (self.neg_sampling == "precalc" and self.split != "train"):
            context = example[0, 3:]
            context = context[context>0]
            context_set = set(context.cpu().numpy())
            ingredients_cnt_wo_context = self.set_ingredients_cnt - context_set
            random_entities = torch.tensor(random.sample(ingredients_cnt_wo_context, nr-len(context)))
            random_entities = torch.cat((context, random_entities))
        elif "precalc" in self.neg_sampling and self.split == "train":
            sub_source = example[0, 0].item()
            sub_target = example[0, 1].item()
            recipe_id = example[0, 2].item()
            # context = example[0, 3:]
            # context = context[context>0]
            # context_set = set(context.cpu().numpy())
            # ingredients_cnt_wo_context = self.set_ingredients_cnt - context_set



            row_index = None
            if recipe_id not in self.sample_2_subst_row:
                pass
            elif sub_source not in self.sample_2_subst_row[recipe_id]:
                pass
            elif sub_target not in self.sample_2_subst_row[recipe_id][sub_source]:
                pass
            else:
                row_index = self.sample_2_subst_row[recipe_id][sub_source][sub_target]
            if row_index is not None:
                # ingredients_cnt_wo_context_list = list(ingredients_cnt_wo_context)
                # subst_per_ingr = self.precalc_substitutabilities[row_index, [self.ingr_cnt_2_subst_col[ingr_cnt] if ingr_cnt in self.ingr_cnt_2_subst_col else 0 for ingr_cnt in ingredients_cnt_wo_context_list]]

                ingredients_list = list(ingredients_cnt)
                # subst_per_ingr = self.precalc_substitutabilities[row_index, [self.ingr_cnt_2_subst_col[ingr_cnt] if ingr_cnt in self.ingr_cnt_2_subst_col else 0 for ingr_cnt in ingredients_cnt_wo_context_list]]
                # the comment-out sections could be used if you want to simulate also the "smart" or "smart2" strategies in combination with strategic sampling
                subst_per_ingr = self.precalc_substitutabilities[row_index, [self.ingr_cnt_2_subst_col[ingr_cnt] if ingr_cnt in self.ingr_cnt_2_subst_col else 0 for ingr_cnt in ingredients_list]]
                subst_sum = torch.sum(subst_per_ingr)
                if subst_sum <= 0: # random sampling if the precalc scores sum to (no information for that sample)
                    random_entities = torch.tensor(random.sample(ingredients_cnt, nr))
                    # random_entities = torch.tensor(random.sample(ingredients_cnt_wo_context, nr-len(context))) # this is also used with "smart" or "smart2"
                    # random_entities = torch.cat((context, random_entities)) # this is also used with "smart" or "smart2"
                else:
                    if self.neg_sampling == "precalc_center":
                        print("precalc_center")
                        # * if the center of the distribution should be weighted more, redicstribute around that center
                        subst_per_ingr = 1 - torch.abs(subst_per_ingr - torch.mean(subst_per_ingr))
                        # * adding the epsilon should be done in both center and margin heavy casese to avoid excluding suggestions
                        subst_per_ingr = subst_per_ingr + subst_per_ingr[subst_per_ingr > 0].min().item()
                        # * if we want the distances to carry more weight - exponate
                        subst_per_ingr = subst_per_ingr ** 2
                        # * then get p as relative based on substitutability
                        p_per_ingr = subst_per_ingr / subst_sum
                        choice_idxs = torch.multinomial(p_per_ingr, nr, replacement=False)
                        # choice_idxs = torch.multinomial(p_per_ingr, nr-len(context), replacement=False)
                        random_entities = torch.tensor([ingredients_list[i] for i in choice_idxs])
                    elif self.neg_sampling == "precalc_fringes":
                        print("precalc_fringes")
                        # * inverting the centre weighted distribution to put emphasize on the tail ends (inverted centre)
                        subst_per_ingr = torch.abs(subst_per_ingr - torch.mean(subst_per_ingr))
                        # * adding the epsilon should be done in both center and margin heavy casese to avoid excluding suggestions
                        subst_per_ingr = subst_per_ingr + subst_per_ingr[subst_per_ingr > 0].min().item()
                        # * if we want the distances to carry more weight - exponate
                        subst_per_ingr = subst_per_ingr ** 2
                        # * then get p as relative based on substitutability
                        p_per_ingr = subst_per_ingr / subst_sum
                        choice_idxs = torch.multinomial(p_per_ingr, nr, replacement=False)
                        # choice_idxs = torch.multinomial(p_per_ingr, nr-len(context), replacement=False)
                        random_entities = torch.tensor([ingredients_list[i] for i in choice_idxs])
                    elif self.neg_sampling == "precalc_head":
                        print("precalc_head")
                        # * add some epsilon to avoid empty probabilities
                        subst_per_ingr = subst_per_ingr + (subst_per_ingr[subst_per_ingr > 0].min().item())
                        # * if we want the distances to carry more weight - exponate
                        subst_per_ingr = subst_per_ingr ** 2
                        p_per_ingr = subst_per_ingr / subst_sum
                        choice_idxs = torch.multinomial(p_per_ingr, nr, replacement=False)
                        # choice_idxs = torch.multinomial(p_per_ingr, nr-len(context), replacement=False)
                        random_entities = torch.tensor([ingredients_list[i] for i in choice_idxs])
                    elif self.neg_sampling == "precalc_tail":
                        print("precalc_tail")
                        # * if inverted, subtract each element value from one and add epsilon to give every sample a chance :)
                        subst_per_ingr = 1 - subst_per_ingr + (subst_per_ingr[subst_per_ingr > 0].min().item())
                        # * if we want the distances to carry more weight - exponate
                        subst_per_ingr = subst_per_ingr ** 2
                        p_per_ingr = subst_per_ingr / subst_sum
                        choice_idxs = torch.multinomial(p_per_ingr, nr, replacement=False)
                        # choice_idxs = torch.multinomial(p_per_ingr, nr-len(context), replacement=False)
                        random_entities = torch.tensor([ingredients_list[i] for i in choice_idxs])



                    # # * if the center of the distribution should be weighted more, redicstribute around that center
                    # subst_per_ingr = 1 - torch.abs(subst_per_ingr - torch.mean(subst_per_ingr))
                    # # * inverting the centre weighted distribution to put emphasize on the tail ends (inverted centre)
                    # # subst_per_ingr = torch.abs(subst_per_ingr - torch.mean(subst_per_ingr))
                    # # * adding the epsilon should be done in both center and margin heavy casese to avoid excluding suggestions
                    # subst_per_ingr = subst_per_ingr + subst_per_ingr[subst_per_ingr > 0].min().item()
                    # # * if we want the distances to carry more weight - exponate
                    # subst_per_ingr = subst_per_ingr ** 2
                    # # * add some epsilon to avoid empty probabilities
                    # # subst_per_ingr = subst_per_ingr + (subst_per_ingr[subst_per_ingr > 0].min().item())
                    # # * if inverted, subtract each element value from one and add epsilon to give every sample a chance :)
                    # # subst_per_ingr = 1 - subst_per_ingr + (subst_per_ingr[subst_per_ingr > 0].min().item())
                    # # * then get p as relative based on substitutability
                    # p_per_ingr = subst_per_ingr / subst_sum
                    # choice_idxs = torch.multinomial(p_per_ingr, nr, replacement=False)
                    # # choice_idxs = torch.multinomial(p_per_ingr, nr-len(context), replacement=False)
                    # random_entities = torch.tensor([ingredients_list[i] for i in choice_idxs])

                    # these are, again, options needed to make negative sampling strategies work with "smart" or "smart2"
                    # random_entities = torch.tensor([ingredients_cnt_wo_context_list[i] for i in choice_idxs])
                    # random_entities = torch.tensor([ingredients_list[i] for i in choice_idxs])
                    # random_entities = torch.cat((context, random_entities))
            else:
                random_entities = torch.tensor(random.sample(ingredients_cnt, nr))
                # random_entities = torch.tensor(random.sample(ingredients_cnt_wo_context, nr-len(context)))
                # random_entities = torch.cat((context, random_entities))

            # precalc_subst_cols = [self.ingr_name_2_subst_col[self.] for ingr_cnt in ingredients_cnt_wo_context_list]

        neg_batch = torch.cat(
            (example[0, 0].repeat(nr).view(nr, 1), random_entities.view(nr, 1)), 1
        )
        output = torch.cat((neg_batch, example[0, 2:].repeat(nr, 1)), 1)
        return output
