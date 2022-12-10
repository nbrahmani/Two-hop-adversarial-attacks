import json
import string
import contractions


# remove contractions
def decontract(text):
    return contractions.fix(text)


# remove punctutations
def remove_punctuation(text):
    text = "".join([i for i in text if i not in string.punctuation])
    return text


def get_dataset(filename):
    data = []
    dataset = []
    with open(filename) as fp:
        for line in fp:
            data.append(json.loads(line))

    labels_dict = {"entailment": 1, "neutral": 2, "contradiction": 0}
    for row in data:
        premise, hypothesis = (remove_punctuation(decontract(row["sentence1"])),
                               remove_punctuation(decontract(row["sentence2"])))
        if row["gold_label"] != "-":
            dataset.append(((premise, hypothesis),
                            labels_dict[row["gold_label"]]))
    return dataset
