# Author: Brahmani Nutakki: Saturday 04 June 2022 01:56:45 PM IST

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch.nn.functional as nnf
from nltk.corpus import wordnet
from pathlib import Path
import contractions
import pyinflect
import string
import torch
import spacy
import json

# from common_pyutil.monitor import Timer

# timer = Timer()
grammar = None


nlp = spacy.load("en_core_web_sm")

sense_file_premise = 'mnli_premise_semantic_concordance_predictions.txt'
sense_file_hypothesis = 'mnli_hypothesis_semantic_concordance_predictions.txt'


near_synset_data_premise = []
near_synset_data_hypothesis = []
with open(sense_file_premise, 'r') as fp:
    for line in fp:
        near_synset_data_premise.append(line)
temp = [x.split() for x in near_synset_data_premise]
temp_dict = {}
for x in temp:
    if len(x) == 2:
        k, v = x
        k = k.split(".")[1]
        if k in temp_dict:
            temp_dict[k].append((v.split("%")[0], v))
        else:
            temp_dict[k] = [(v.split("%")[0], v)]
near_synset_data_premise = temp_dict.copy()
for k in near_synset_data_premise:
    near_synset_data_premise[k] = dict(near_synset_data_premise[k])

near_synset_data_hypothesis = []
with open(sense_file_hypothesis, 'r') as fp:
    for line in fp:
        near_synset_data_hypothesis.append(line)
temp = [x.split() for x in near_synset_data_hypothesis]
temp_dict = {}
for x in temp:
    if len(x) == 2:
        k, v = x
        k = k.split(".")[1]
        if k in temp_dict:
            temp_dict[k].append((v.split("%")[0], v))
        else:
            temp_dict[k] = [(v.split("%")[0], v)]
near_synset_data_hypothesis = temp_dict.copy()
for k in near_synset_data_hypothesis:
    near_synset_data_hypothesis[k] = dict(near_synset_data_hypothesis[k])


def _init_grammar_model():
    global grammar
    grammar = GrammarCheck()


# Get monotinicity based polarized sentences from file
def get_polarized_sentences(filename):
    data = []
    with open(filename, "r") as fp:
        for line in fp:
            data.append(line)
    return data


# Remove punctuations
def remove_punctuation(text):
    text = "".join([i for i in text if i not in string.punctuation])
    return text


# Remove contractions
def decontract(text):
    return contractions.fix(text)


# Prepeocess text before transformations
def preprocess_text(text):
    text = text.lower()
    text = decontract(text)
    text = remove_punctuation(text)
    return text


# Get synonyms of word with same sense using wordnet

def get_synonyms(word, text, near_synset):
    synonyms = set()
    # get synset with the nearest sense based on the text and sentence
    doc = nlp(text)
    for token in doc:
        if token.text == word:
            tag = token.tag_
            break
    try:
        # get the words with the same sense as the synset i.e the lemmas
        for lem in near_synset.lemmas():
            # pyinflect to get correct verb form
            if lem.name() != word and nlp(lem.name())[0]._.inflect(tag):
                synonyms.add(nlp(lem.name())[0]._.inflect(tag))
    except Exception:
        pass
        # print("No synonym found with the same sense")
        # synonyms.add(word)

    return list(synonyms)


# Get hypernyms of word with same sense using wordnet

def get_hypernyms(word, text, near_synset):
    hypernyms = set()
    doc = nlp(text)
    for token in doc:
        if token.text == word:
            tag = token.tag_
            break

    if near_synset:
        for s in near_synset.hypernym_distances():
            if s[1] <= 3:
                name = s[0].name().split('.')[0]
                if name != word and nlp(name)[0]._.inflect(tag):
                    hypernyms.add(name)
    else:
        pass
        # print("No hypernyms found with same sense")

    return list(hypernyms)


# Get hyponyms of word with same sense using wordnet

def get_hyponyms(word, text, near_synset):
    hyponyms = set()
    doc = nlp(text)
    for token in doc:
        if token.text == word:
            tag = token.tag_
            break

    if near_synset:
        for s in near_synset.hyponyms():
            name = s.name().split('.')[0]
            if name != word and nlp(name)[0]._.inflect(tag):
                hyponyms.add(name)
    else:
        pass
        # print("No hyponyms found with same sense")

    return list(hyponyms)


class GrammarCheck:
    def __init__(self):
        print("Initializing grammar model and tokenizer")
        self.device = "cuda:1"
        # with timer:
        grammar_model_path = str(Path(__file__).absolute().parent.parent.parent.
                                 joinpath("resources/models/bert-base-cased-finetuned-cola"))
        self.grammar_tokenizer = AutoTokenizer.from_pretrained(grammar_model_path)
        self.grammar_model = AutoModelForSequenceClassification.from_pretrained(grammar_model_path)
        self.grammar_model = self.grammar_model.to(self.device)
        self.grammar_model.eval()
        # print(f"Initialized grammar model and tokenizer in {timer.time} seconds")

    def check_grammar(self, text):
        tokens = self.grammar_tokenizer.encode_plus(text)
        tokens_tensor = torch.tensor([tokens['input_ids']]).to(self.device)
        segments_tensor = torch.tensor([tokens['token_type_ids']]).to(self.device)

        with torch.no_grad():
            outputs = self.grammar_model(tokens_tensor, segments_tensor)
            logits = outputs[0]

        probs = nnf.softmax(logits, dim=1)
        index = probs.argmax()
        return (probs, index)


# return top 5 markers from both premise and hypothesis
class Markers:
    def __init__(self):
        print("Initializing Key model")
        # with timer:
        self.key_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.key_model = self.key_model.to('cpu')
        self.key_model.eval()
        # print(f"Initialized Key model in {timer.time} seconds")

    def extract_markers(self, input_pair):
        top_n = 5
        keywords, keyindices = [], []
        for doc in input_pair:
            try:
                count = CountVectorizer(ngram_range=(1, 1), stop_words="english").fit([doc])
                candidates = count.get_feature_names_out()
                doc_embeddings = self.key_model.encode([doc])
                candidate_embeddings = self.key_model.encode(candidates)
                distances = cosine_similarity(doc_embeddings, candidate_embeddings)
                indices = [index for index in distances.argsort()[0][-top_n:]]
                keyindices.append(indices)
                keywords.append([candidates[index] for index in indices])
            except Exception:
                pass
        return keywords


# Get input from dataset file
def get_dataset(filename):
    data = []
    dataset = []
    with open(filename, 'r') as fp:
        for line in fp:
            data.append(json.loads(line))

    # labels_dict = {"entailment": 0, "neutral": 1, "contradiction": 2}
    labels_dict = {"entailment": 1, "neutral": 2, "contradiction": 0}
    for row in data:
        premise, hypothesis = (preprocess_text(row["sentence1"]),
                               preprocess_text(row["sentence2"]))
        label = row['gold_label']
        if label != "-":
            if label == 0 or label == 1 or label == 2:
                dataset.append(((premise, hypothesis), label))
            else:
                dataset.append(((premise, hypothesis), labels_dict[label]))
    return dataset


def remove_transformations_based_on_grammar_score(text, transformations):
    original_score = grammar.check_grammar(text)[0][0][1]
    for index, t in enumerate(transformations):
        score_t = grammar.check_grammar(t)[0][0][1]
        if abs(original_score - score_t) >= 0.1:
            transformations.remove(t)
    return transformations


# Replace marker[0] word with synonym to get equivalent transforations

def equi_transform(marker, text, mode, index):
    # with timer:
    near_synset = extract_sense_from_file(mode, index, marker)
    # print(f"Time for equi_transform extract sense {timer.time} seconds")
    # with timer:
    replacements = get_synonyms(marker, text, near_synset)
    # print(f"Time for equi_transform get_synonyms {timer.time} seconds")
    transformations = [text.replace(marker, word) for word in replacements]
    # with timer:
    transformations = remove_transformations_based_on_grammar_score(text, transformations)
    # print(f"Time for equi_transform grammar check {timer.time} seconds")
    # grammar_score = [check_grammar(t) for t in transformations]
    return transformations


# Replace marker[0] word with hypernym or hyponym to get
# monotonicity-based transforations to get entailed sentences
# Upward monotone- Hypernyms, Downward monotone- Hyponyms

def mono_transform_entail(marker, text, index, polarized_file, mode):
    # get monotonicity of marker[0] in text
    near_synset = extract_sense_from_file(mode, index, marker)
    monotone_data = get_polarized_sentences(polarized_file)
    replacements = []
    try:
        monotone_text = monotone_data[index]
        monotone_index = text.split().index(marker)
        if monotone_text.split()[monotone_index][-1] == '↑':
            replacements = get_hypernyms(marker, text, near_synset)
        elif monotone_text.split()[monotone_index][-1] == '↓':
            replacements = get_hyponyms(marker, text, near_synset)[:15]
    except Exception:
        pass
    transformations = [text.replace(marker, word) for word in replacements]
    transformations = remove_transformations_based_on_grammar_score(text, transformations)
    # grammar_score = [check_grammar(t) for t in transformations]
    return transformations


# Replace marker[0] word with hypernym or hyponym to get
# monotonicity-based transforations to get neutral sentences
# Upward monotone- Hyponyms, Downward monotone- Hypernyms

def mono_transform_neutral(marker, text, index, polarized_file, mode):
    # get monotonicity of marker[0] in text
    near_synset = extract_sense_from_file(mode, index, marker)
    monotone_data = get_polarized_sentences(polarized_file)
    replacements = []
    try:
        monotone_text = monotone_data[index]
        monotone_index = text.split().index(marker)
        if monotone_text.split()[monotone_index][-1] == '↑':
            replacements = get_hyponyms(marker, text, near_synset)[:15]
        elif monotone_text.split()[monotone_index][-1] == '↓':
            replacements = get_hypernyms(marker, text, near_synset)
    except Exception:
        pass
    transformations = [text.replace(marker, word) for word in replacements]
    transformations = remove_transformations_based_on_grammar_score(text, transformations)
    # grammar_score = [check_grammar(t) for t in transformations]
    return transformations


def parse_tree_into_json(filename, dataset):
    retval = []

    for index, example in enumerate(dataset):
        row = {}
        print(index)
        premise = nlp(example[0][0])
        hypothesis = nlp(example[0][1])

        row['premise'] = premise.to_json()
        row['hypothesis'] = hypothesis.to_json()

        retval.append(row)

    with open('/home/gen/brahmani/code/resources/corpus/'+filename+'.jsonl', "w") as fp:
        for val in retval:
            fp.writelines(json.dumps(val))
            fp.writelines('\n')


# Get sense from senses file in /code/esc
def extract_sense_from_file(mode, index, word):
    word = wordnet.morphy(word)
    lemma_synset = None

    if mode == 'premise':
        data = near_synset_data_premise
    else:
        data = near_synset_data_hypothesis

    sent_id = f"s{index}"
    if sent_id in data:
        sense_id = data[sent_id].get(word)
        if sense_id:
            try:
                lemma = wordnet.lemma_from_key(sense_id)
                lemma_synset = lemma.synset()
            except Exception:
                pass

    # for row in data:
    #     if len(row.split()) == 2:
    #         sent_id = row.split()[0]
    #         sense_id = row.split()[1]

    #         if sent_id.split('.')[1] == f's{index}' and word == sense_id.split('%')[0]:
    #             lemma = wordnet.lemma_from_key(sense_id)
    #             lemma_synset = lemma.synset()

    return lemma_synset


def compare_textfooler(data):
    spell_checker = jamspell.TSpellCorrector()
    # spell_checker.LoadLangModel('/home/gen/brahmani/code/resources/en.bin'
    tf_transformations_file = '/home/gen/brahmani/code/adversarial_analysis/textfooler_best_trans.jsonl'
    polarized_file = "/home/gen/brahmani/code/adversarial_analysis/polarized_files/sentences.depccg.parsed.txt.polarized"
    fail = 0
    success = 0
    no_word_match = 0
    no_label = 0
    no_polarity = 0
    tf_trans = []
    results = []
    mode = None
    label_changes = {'PEH^': {0: [0], 1: [1], 2: [2]},
                     'PMEH^': {0: [], 1: [1], 2: [2]},
                     'PMNH^': {0: [0], 1: [], 2: []}}

    with open(tf_transformations_file, 'r') as fp:
        for line in fp:
            tf_trans.append(json.loads(line))

    # import ipdb; ipdb.set_trace()
    for tindex, row in enumerate(tf_trans):
        print(f'{tindex}/{len(tf_trans)}')
        prem = preprocess_text(row['premise'])
        orig_hyp = preprocess_text(row['orig_hyp'])
        trans_hyp = preprocess_text(row['trans_hyp'])
        try:
            orig_word = [x for i, x in enumerate(orig_hyp.split()) if x != trans_hyp.split()[i]][0]
            repl_word = [x for i, x in enumerate(trans_hyp.split()) if x != orig_hyp.split()[i]][0]
        except Exception:
            row['status'] = 'index error'
            results.append(row)
            continue

        index = data.index(((prem, orig_hyp), row['ground_truth']))
        near_synset = extract_sense_from_file('hypothesis', index, orig_word)

        synonyms = get_synonyms(orig_word, orig_hyp, near_synset)
        hypernyms = get_hypernyms(orig_word, orig_hyp, near_synset)
        hyponyms = get_hyponyms(orig_word, orig_hyp, near_synset)
        row['synonyms'] = synonyms
        row['hypernyms'] = hypernyms
        row['hyponyms'] = hyponyms

        monotone_data = get_polarized_sentences(polarized_file)
        monotone_text = monotone_data[index]
        monotone_index = orig_hyp.split().index(orig_word)
        try:
            monotone_symbol = monotone_text.split()[monotone_index][-1]
            row['monotone_symbol'] = monotone_symbol
        except Exception:
            no_polarity += 1
            row['status'] = 'No polarity'
            results.append(row)
            continue

        if repl_word in synonyms:
            mode = 'PEH^'
        elif repl_word in hypernyms:
            if monotone_symbol == '↑':
                mode = 'PMEH^'
            else:
                mode = 'PMNH^'
        elif repl_word in hyponyms:
            if monotone_symbol == '↓':
                mode = 'PMEH^'
            else:
                mode = 'PMNH^'
        else:
            no_word_match += 1
            row['status'] = 'No word match'
            results.append(row)
            continue

        if label_changes[mode][row['ground_truth']]:
            if row['pred_labels'] in label_changes[mode][row['ground_truth']]:
                row['status'] = 'Attack Failed'
                fail += 1
            else:
                row['status'] = 'Attack Success'
                success += 1
        else:
            spellchecks = []
            no_label += 1
            row['status'] = 'No label'
            results.append(row)
            continue

        print(f'P, H: {prem} <<>> {orig_hyp}')
        print(f'Orig word: {orig_word} <<>> Repl_word: {repl_word}')
        print(f'Mode: {mode}')
        print(f'Pred: {row["pred_labels"]}, GT: {row["ground_truth"]}')
        print(f'Status: {row["status"]}')
        print(f'Final Status count: Success = {success}, Fail = {fail}, no_label_shift= {no_label}, no_word_match = {no_word_match}, no_polarity = {no_polarity}')

    with open('outfile_tf.jsonl', 'w') as fp:
        for row in results:
            fp.write(f'{json.dumps(row)}\n')
        fp.write(f'Final Status count: Success = {success}, Fail = {fail}, no_label_shift= {no_label}, no_word_match = {no_word_match}, no_polarity = {no_polarity}')

    return success, fail, no_label, no_word_match, no_polarity
