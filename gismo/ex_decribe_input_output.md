# Input and output models

- Reminder: Conda environment: `conda activate inverse_cooking_gismo`
- starting training (best performing model): `python train.py name=GIN_MLP setup=context-full max_context=43 lr=0.00005 w_decay=0.0001 hidden=300 emb_d=300 dropout=0.25 nr=400 nlayers=2 lambda_=0.0 i=1 init_emb=random with_titles=False with_set=True filter=False`

- for the arcelik dataset: `python train.py name=GIN_MLP setup=context-full max_context=43 lr=0.00005 w_decay=0.0001 hidden=300 emb_d=300 dropout=0.25 nr=20 nlayers=2 lambda_=0.0 i=1 init_emb=random with_titles=False with_set=True filter=False`

- test `C:\UM\Master\FoodRecommendations\literature_models\GISMo\gismo\gismo>python train.py name=GIN_MLP setup=context-full max_context=43 lr=0.00005 w_decay=0.0001 hidden=300 emb_d=400 dropout=0.25 nr=20 nlayers=2 lambda_=0.0 i=1 init_emb=random with_titles=False with_set=True filter=False`

## Usage:
- add the right input graph and samples
- add the config file with your configuration specifications
- then run one of the commends above

## Used data
- files `train_comments_subs`, `test_comments_subs`, `val_comments_subs` all contain of recipes with ingredient lists as well a substitution tuple and recipe id (but not instructions, so if we want to use them, we might have to make use of text mining). Structure:
  - array of recipes as dictionaries with `id`, `ingredients`, and `subs` fields
  - id is the recipe id as from Recipe1M
  - subs is a tuple (source ingredient, ground truth substitution)
  - ingredients is a list of lists; each sub-list contains of a number of ingredient labels for the same ingredient (e.g. ```ingredients[0] = ['margerine', 'margerine_spread', 'becel_margerine'...]```)

## Input

- `val_comments_subs.pkl`, `test_comments_subs.pkl`, `train_comments_subs.pkl` are the samples for the test, train, validations sets.
- `vocab_ingrs.pkl` are used to find the right nodes for the ingredients specified in the samples
- `nodes_191120.csv`, `edges_191120.csv` are the

## Output

- both val_ranks.txt as well as val_ranks_full.txt use the node numbers to indicate the resources and can be mapped with the help of the [vocab_ingrs.pkl](https://dl.fbaipublicfiles.com/gismo/vocab_ingrs.pkl) file (requires the inverse cooking vocabulary module `from inv_cooking.datasets.vocabulary import Vocabulary`)

### val_ranks.txt

- lines containing 4 values, in order:
  - source ingredient (that which should be substituted in a recipe)
  - ground truth substitution ingredient
  - predicted rank of the ground truth substittuion ingredient
  - top ranked predicted ingredient

### val_ranks_full.txt

- lines containing k+2 values, where k is a magic number within the `get_loss_test` function
- values in order:
  - source ingredient
  - something that corresponds to the label if the ground truth substitution rank were a ingredient (i.e. the name of the ingredient that has as id the rank of the ingredient)
  - predicted substitute ingredient
  - I altered this to give: translated source ingredient, translated ground truth ingredient, rank of ground truth ingredient, recommendation


## Negative Sampling for Contrastive Learning
- there are three methods of sampling negative samples for the contrastive learning:
  - random: considers all ingredients barring the source ingredient
  - smart1: considers all ingredients excepts it removes part of the context from the positive sample (what exactly?)
  - smart2: also considers all ingredients excepts it removes part of the context from the positive sample (what exactly?)