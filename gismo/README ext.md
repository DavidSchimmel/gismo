# Extension

## Setup

- set up the environment as described in [README.md](README.md)
- into `gismo\checkpoints\graph\`, add the `edges_191120.csv` and `nodes_191120.csv` files
  - for the baseline, follow the instructions in [README.md](README.md)
  - for other graphs such as those generated in [structuredRecipe1M](https://github.com/DavidSchimmel/structured_recipe1m), use the nodes and edges from there and name them accordingly
- into `gismo\checkpoints\`, add:
  - `vocab_ingrs.pkl` as described in [README.md](README.md)
  - `train_comments_subs.pkl`, `test_comments_subs.pkl`, and `val_comments_subs.pkl` either as described in [README.md](README.md) or if you want to use other samples, copy those splits here
- into `gismo\checkpoints\precalculated_substitutabilities\`, copy `cos_similarities.pt`, `ingr_2_col.pkl`, and `sample_2_row.pkl`, containing a vector with precalculated substitutabilities and the mappings from ingredients to columns and samples to rows
  - this is only required if you want to apply strategic negative sampling

## Running the experiments

- make sure you use the right nodes, edges, comments_subs
- adapt the `.\conf\config.yaml` file accordingly
  - for runs using the default FlavorGraph, the provided config.yaml is enough (although you should set the `neg_sampling` item accordingly if you want to run negative sampling)
  - for runs using an ArcSubs graph, the provided `config_arcsubs.yaml` include a suggested configuration

- in your conda console, activate the environment: `conda activate inverse_cooking_gismo`
- change directory to the gismo directory (in which this readme is located)
- starting training:
  - for FlavorGraph: `python train.py name=GIN_MLP setup=context-full max_context=43 lr=0.00005 w_decay=0.0001 hidden=300 emb_d=300 dropout=0.25 nr=400 nlayers=2 lambda_=0.0 i=1 init_emb=random with_titles=False with_set=True filter=False`
  - for ArcSubs: `python train.py name=GIN_MLP setup=context-full max_context=43 lr=0.00005 w_decay=0.0001 hidden=300 emb_d=300 dropout=0.25 nr=20 nlayers=2 lambda_=0.0 i=1 init_emb=random with_titles=False with_set=True filter=False`
- the model and outputs will be written to the directory specified in the `config.yaml` (default: `~/project/gismo/out/` )
- the files with the results include:
  - `val_ranks.txt` and `test_ranks.txt`
    - lines containing 4 values, in order:
      - source ingredient (that which should be substituted in a recipe)
      - ground truth substitution ingredient
      - predicted rank of the ground truth substittuion ingredient
      - top ranked predicted ingredient
  - `val_ranks_full.txt` and `test_ranks_full.txt`
    - lines containing k+2 values, where k is a magic number within the `get_loss_test` function
    - values in order:
      - source ingredient
      - top ranked predicted ingredient
      - k top ranked ingredients

- the evaluation metrics MRR, HIT@k for the various val sets and the test set, and training losses are displayed in the conda console
- the diversity metrics are calculated separately based on the test or val ranks files
