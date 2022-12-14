import os
import json
import torch
from pathlib import Path
from torch.nn.functional import softmax
# from common_pyutil.monitor import Timer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import util

# timer = Timer()


# Get predicted label from model- convert to batch
def get_label_from_model(tokenizer, model, text, device):
    # text of type tuple(premise, hypothesis)

    inputs_dict = tokenizer([(list(text))], add_special_tokens=True,
                            padding="max_length", max_length=512,
                            truncation=True, return_tensors='pt')
    inputs_dict.to(device)

    with torch.no_grad():
        outputs = model(**inputs_dict)
        logits = outputs[0]

    scores = softmax(logits, dim=1)
    label = scores.argmax()
    label = int(label.cpu().numpy())
    return label


def get_label_from_model_batch(tokenizer, model, inp_texts, batch_size, device):
    # text of type tuple(premise, hypothesis)
    batch = [list(x) for x in inp_texts]
    label = []
    i = 0
    end_index = 0

    print(f'len of input batch: {len(inp_texts)}')
    print(f'len of batch: {batch_size}')
    while end_index != len(inp_texts):
        if (i+1) * batch_size > len(inp_texts):
            end_index = len(inp_texts)
        else:
            end_index = (i+1) * batch_size

        current_batch = [list(x) for x in batch[i*batch_size: end_index]]
        inputs_dict = tokenizer(current_batch,
                                add_special_tokens=True,
                                padding="max_length", max_length=512,
                                truncation=True, return_tensors='pt')
        inputs_dict.to(device)

        with torch.no_grad():
            outputs = model(**inputs_dict)
            logits = outputs[0]

        scores = softmax(logits, dim=1)
        label += [int(x.argmax().cpu().numpy()) for x in scores]
        i += 1

    return label


# Checks if the attack is skipped

def is_skip_attack(ground_truth, pred_label):
    if ground_truth != pred_label:
        return True
    return False


# creates (P', H), (P, H') and (P', H') using equivalent and
# monotonicity based replacement
# EP^H- Equivalent tranform of Premise, Hypothesis
# MEP^H- Monotonic transform(entailment) of Premise; Hypothesis
# MNP^H- Monotonic transform(neutral) of Premise; Hypothesis
# index- sentence index in datafile

def create_all_transformations(marker, text, index, polarized_file):
    print("Creating transformations")
    premise = text[0]
    hypothesis = text[1]
    retval = {'EP^H': [],
              'MEP^H': [],
              'MNP^H': [],
              'PEH^': [],
              'PMEH^': [],
              'PMNH^': [],
              'EP^EH^': [],
              'EP^MEH^': [],
              'EP^MNH^': [],
              'MEP^EH^': [],
              'MEP^MEH^': [],
              'MEP^MNH^': [],
              'MNP^EH^': [],
              'MNP^MEH^': [],
              'MNP^MNH^': []}

    for m1 in marker[0][:1]:
        p_eq = util.equi_transform(m1, premise, 'premise', index)
        p_me = util.mono_transform_entail(m1, premise,
                                          index, polarized_file, 'premise')
        p_mn = util.mono_transform_neutral(m1, premise,
                                           index, polarized_file, 'premise')
        for m2 in marker[1][:1]:
            h_eq = util.equi_transform(m2, hypothesis, 'hypothesis', index)
            h_me = util.mono_transform_entail(m2, hypothesis,
                                              index, polarized_file, 'hypothesis')
            h_mn = util.mono_transform_neutral(m2, hypothesis,
                                               index, polarized_file, 'hypothesis')

            # creating (P', H) pairs
            retval['EP^H'] += [(p, hypothesis) for p in p_eq]
            retval['MEP^H'] += [(p, hypothesis) for p in p_me]
            retval['MNP^H'] += [(p, hypothesis) for p in p_mn]

            # creating (P, H') pairs
            retval['PEH^'] += [(premise, h) for h in h_eq]
            retval['PMEH^'] += [(premise, h) for h in h_me]
            retval['PMNH^'] += [(premise, h) for h in h_mn]

            # creating (P', H') pairs
            retval['EP^EH^'] += [(p, h) for p in p_eq for h in h_eq]
            retval['EP^MEH^'] += [(p, h) for p in p_eq for h in h_me]
            retval['EP^MNH^'] += [(p, h) for p in p_eq for h in h_mn]
            retval['MEP^EH^'] += [(p, h) for p in p_me for h in h_eq]
            retval['MEP^MEH^'] += [(p, h) for p in p_me for h in h_me]
            retval['MEP^MNH^'] += [(p, h) for p in p_me for h in h_mn]
            retval['MNP^EH^'] += [(p, h) for p in p_mn for h in h_eq]
            retval['MNP^MEH^'] += [(p, h) for p in p_mn for h in h_me]
            retval['MNP^MNH^'] += [(p, h) for p in p_mn for h in h_mn]

    for key in retval:
        retval[key] = list(set(retval[key]))

    return retval


# Get labels for all transformations of an input pair
# Input: transformations from create_all_transformations

def get_labels_for_all_transformations(transformations, tokenizer, model,
                                       batch_size, device):
    labels = dict()
    for k, v in transformations.items():

        if v:
            labels[k] = get_label_from_model_batch(tokenizer, model, v,
                                                   batch_size, device)
        else:
            labels[k] = []

    return labels


# Store transformations as a jsonl file

def store_transformations_file(transformations, ground_truth, outfile):
    row = {}
    label_changes = {'EP^H': {0: [0], 1: [1], 2: []},
                     'MEP^H': {0: [2], 1: [2], 2: []},
                     'MNP^H': {0: [0], 1: [1], 2: []},
                     'PEH^': {0: [0], 1: [1], 2: [2]},
                     'PMEH^': {0: [], 1: [1], 2: [2]},
                     'PMNH^': {0: [0], 1: [], 2: []},
                     'EP^EH^': {0: [0], 1: [1], 2: []},
                     'EP^MEH^': {0: [], 1: [1], 2: []},
                     'EP^MNH^': {0: [0], 1: [], 2: []},
                     'MEP^EH^': {0: [2], 1: [2], 2: []},
                     'MEP^MEH^': {0: [], 1: [2], 2: []},
                     'MEP^MNH^': {0: [2], 1: [], 2: []},
                     'MNP^EH^': {0: [0], 1: [1], 2: []},
                     'MNP^MEH^': {0: [], 1: [1], 2: []},
                     'MNP^MNH^': {0: [0], 1: [], 2: []}}

    with open(outfile, 'a') as fp:
        for k, v in transformations.items():
            if label_changes[k][ground_truth]:
                for index, val in enumerate(v):
                    row['sentence1'] = val[0]
                    row['sentence2'] = val[1]
                    row['labels'] = label_changes[k][ground_truth][0]
                    fp.write(f"{json.dumps(row)}\n")


# Get attack status of all transformaations based on label_changes
# label_changes defines the rules of label shifts for diff transformations
# 0- Contradiction, 1- Entailment, 2- Neutral

def attack_status(ground_truth, pred_label, transformation_labels):
    status = dict()
    actual_label = dict()
    label_changes = {'EP^H': {0: [0], 1: [1], 2: []},
                     # 2
                     # 'MEP^H': {0: [0, 2], 1: [1, 2], 2: []},
                     'MEP^H': {0: [2], 1: [2], 2: []},
                     # 3
                     'MNP^H': {0: [0], 1: [1], 2: []},
                     # 4
                     # 'PEH^': {0: [0], 1: [1], 2: [1, 2]},
                     'PEH^': {0: [0], 1: [1], 2: [2]},
                     # 5
                     # 'PMEH^': {0: [], 1: [1], 2: [1, 2]},
                     'PMEH^': {0: [], 1: [1], 2: [2]},
                     # 6
                     'PMNH^': {0: [0], 1: [], 2: []},
                     # 7
                     'EP^EH^': {0: [0], 1: [1], 2: []},
                     # 8
                     'EP^MEH^': {0: [], 1: [1], 2: []},
                     # 9
                     'EP^MNH^': {0: [0], 1: [], 2: []},
                     # 0
                     # 'MEP^EH^': {0: [0, 2], 1: [1, 2], 2: []},
                     'MEP^EH^': {0: [2], 1: [2], 2: []},
                     # 1
                     # 'MEP^MEH^': {0: [], 1: [1, 2], 2: []},
                     'MEP^MEH^': {0: [], 1: [2], 2: []},
                     # 2
                     # 'MEP^MNH^': {0: [0, 2], 1: [], 2: []},
                     'MEP^MNH^': {0: [2], 1: [], 2: []},
                     # 3
                     'MNP^EH^': {0: [0], 1: [1], 2: []},
                     # 4
                     'MNP^MEH^': {0: [], 1: [1], 2: []},
                     # 5
                     'MNP^MNH^': {0: [0], 1: [], 2: []}}

    if is_skip_attack(ground_truth, pred_label):
        status = 'Skipped'
    else:
        for k, v in transformation_labels.items():
            status[k] = []
            if label_changes[k][pred_label]:
                actual_label[k] = label_changes[k][pred_label]
                for label in v:
                    if label in label_changes[k][pred_label]:
                        status[k].append('Failed')
                    else:
                        status[k].append('Success')
            else:
                status[k].append('No label shift rule')

    return status


# return top 5 markers from both premise and hypothesis

def get_markers_from_file(filename):
    markers = []
    with open(filename, 'r') as fp:
        for line in fp:
            markers.append(json.loads(line))
    return markers


def attack_status_counts_individual(status):
    retval = {'success': 0, 'fail': 0, 'no_trans': 0, 'no_rule': 0}

    if status != 'Skipped':
        for k, v in status.items():
            if v:
                for stat in v:
                    if stat == 'No label shift rule':
                        retval['no_rule'] += 1
                    elif stat == 'Failed':
                        retval['fail'] += 1
                    elif stat == 'Success':
                        retval['success'] += 1
            else:
                retval['no_trans'] += 1

    return retval


def write_results_to_file(filename, text, ground_truth, index, markers, transformations,
                          tlabels, pred_label, status, final_status):

    with open(filename, 'a') as fp:
        fp.write(f"Example No: {index}\n")
        fp.write(f"Premise: {text[0]}\n")
        fp.write(f"Hypothesis: {text[1]}\n")
        fp.write(f"Markers: Prem- {markers[0]}, Hyp- {markers[1]}\n")
        fp.write(f"Ground Truth: {ground_truth}\n")
        fp.write(f"Predicted label: {pred_label}\n")
        # fp.write(f"Grammar Score Prem: {util.check_grammaticality(text[0])[0][0][1]}\n")
        # fp.write(f"Grammar Score Prem: {util.check_grammaticality(text[1])[0][0][1]}\n")
        for k, v in transformations.items():
            fp.write(f"\tTransformations of type {k}\n")
            if status[k] == []:
                fp.write("\t\t\tNo transformations exist\n")
                continue
            elif status[k][0] == 'No label shift rule':
                fp.write("\t\t\tAttack Status: 'No  label shift defined\n")
                continue
            else:
                for tindex, t in enumerate(v):
                    fp.write(f"\t\tPremise: {t[0]}\n")
                    fp.write(f"\t\tHypothsis: {t[1]}\n")
                    fp.write(f"\t\t\tAttack status: {status[k][tindex]}\n")
                    fp.write(f"\t\t\tLabel: {tlabels[k][tindex]}\n")
                    # fp.write(f"\t\t\tGrammar score: {grammar_score[k][tindex]}\n")
        stat_count = attack_status_counts_individual(status)
        fp.write(f"\t\t\tStatus counts: {stat_count}\n\n")
        fp.write(f"Final Attack Status: {final_status}\n")


def get_grammar_score(transformations):
    grammar_score = {}

    for k, v in transformations.items():
        grammar_score[k] = []
        for _ in v:
            #  get prob pf index 1 that corresponds to true value
            score_p = util.check_grammaticality(_[0])[0][0][1]
            score_h = util.check_grammaticality(_[1])[0][0][1]
            grammar_score[k].append([score_p, score_h])

    return grammar_score


# Get final status for that example
# 0-success, 1-fail 2- skip

def final_attack_status(stat_count):
    if stat_count['success'] >= 1:
        return 'success'
    else:
        return 'fail'


def initialize_models(device):
    print("Initializing pipeline model")
    # with timer:
    model_path = str(Path(__file__).absolute().parent.parent.parent.
                     joinpath("resources/models/bert-base-uncased-MNLI"))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    # print(f"Initialized pipeline model in {timer.time} seconds")
    return model, tokenizer


def pipeline():
    device = 'cuda:0'
    batch_size = 128
    filename = "/home/gen/brahmani/code/resources/datasets/mnli/train_split.jsonl"
    polarized_file = "/home/gen/brahmani/code/ccg/mnli_5k.depccg.parsed.txt.polarized"
    # polarized_file = "/home/gen/brahmani/code/adversarial_analysis/polarized_files/sentences.depccg.parsed.txt.polarized"
    # markers_file = '/home/gen/brahmani/code/adversarial_analysis/transformations/markers_snli_sample.txt'
    markers_file = '/home/gen/brahmani/code/adversarial_analysis/transformations/markers/markers_mnli_5k.jsonl'
    outfile = 'results/bert_results_mnli_final_marker1_5k.txt'
    total_status = {'success': 0, 'fail': 0, 'no_trans': 0, 'no_rule': 0}
    final_status_counts = {'success': 0, 'fail': 0, 'skip': 0}
    storefile = 'bert_m1_mnli_5k.jsonl'

    model, tokenizer = initialize_models(device)

    util._init_grammar_model()

    if Path(outfile).exists():
        os.remove(outfile)

    print("Getting dataset and markers")
    data = util.get_dataset(filename)
    markers = get_markers_from_file(markers_file)

    if os.path.exists(storefile):
        os.remove(storefile)

    for index, row in enumerate(data):
        print(f'Index: {index}')
        pred_label = get_label_from_model(tokenizer, model, row[0], device)
        if not is_skip_attack(row[1], pred_label) and len(markers[index]) == 2:
            # with timer:
            transformations = create_all_transformations(markers[index],
                                                         row[0],
                                                         index,
                                                         polarized_file)
            # print(f'Time to create transformations: {timer.time}')

            store_transformations_file(transformations, row[1], storefile)
            # with timer:
            tlabels = get_labels_for_all_transformations(transformations, tokenizer,
                                                         model, batch_size, device)
            # print(f'Time to get labels of all trans: {timer.time}')
            # with timer:
            status = attack_status(row[1], pred_label, tlabels)
            stat_count = attack_status_counts_individual(status)
            for k in total_status.keys():
                total_status[k] += stat_count[k]
            print(total_status)
            final_status = final_attack_status(stat_count)
            final_status_counts[final_status] += 1
            # print(f'All status counts: {timer.time}')
            print('Writing to file')
            write_results_to_file(outfile, row[0], row[1], index, markers[index],
                                  transformations, tlabels, pred_label, status,
                                  final_status)
        else:
            status = 'Skipped'

        print(status)
        print(f"Final status: {final_status_counts}")

    with open(outfile, 'a') as fp:
        fp.write(f"Total status counts: {total_status}\n")
        fp.write(f"Example wise status counts: {final_status_counts}\n")


if __name__ == '__main__':
    pipeline()
