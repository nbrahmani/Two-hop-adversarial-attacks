import argparse
import textattack
from get_dataset_from_file import get_dataset


# parser = argparse.ArgumentParser()
# parser.add_argument('--path-to-file', type=str)
# filename = parser.parse_args().path_to_file

# Path to the dataset file. Format: jsonl
filename = ''

dataset = textattack.datasets.Dataset(get_dataset(filename),
                                      input_columns=("premise", "hypothesis"),
                                      label_names=("contradiction",
                                                   "entailment",
                                                   "neutral"))
