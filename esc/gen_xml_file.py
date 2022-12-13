import json
import string
import contractions
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize


# Remove punctuations

def remove_punctuation(text):
    text = "".join([i for i in text if i not in string.punctuation])
    return text


# Remove contractions

def decontract(text):
    return contractions.fix(text)


def remove_stopwords(text):
    words = set(stopwords.words('english'))
    text = "".join([i+" " for i in text.split() if i not in words])

    return text


# Prepeocess text before transformations

def preprocess_text(text):
    text = text.lower()
    text = decontract(text)
#    text = remove_stopwords(text)
    text = remove_punctuation(text)
    return text


# Get input from dataset file

def get_dataset(filename):
    data = []
    dataset = []
    with open(filename, 'r') as fp:
        for line in fp:
            data.append(json.loads(line))

    labels_dict = {"entailment": 1, "neutral": 2, "contradiction": 0}
    for row in data:
        #premise, hypothesis = (remove_punctuation(row["sentence1"]), remove_punctuation(row["sentence2"]))
        premise, hypothesis = (preprocess_text(row["sentence1"]), preprocess_text(row["sentence2"]))

        label = row['labels']
        if label != "-":
            dataset.append(((premise, hypothesis), labels_dict[label]))
    return dataset


# Generate semantic concordance file:mode-premise or hypothesis

def xml_file_for_single_example(line):
    text_id = 'd000'
    instance_id = 0
    outfile  = "line.xml"
    filename = "owt-10k"
    index = 0
    pos_map = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'AFX', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'RB', 'RP', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WRB'}
    with open(outfile, "w") as fp:
        fp.write('<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n')
        fp.write(f'<corpus lang=\"en\" source=\"{filename.split("/")[-1]}\">\n')
        fp.write(f'<text id=\"{text_id}\">\n')
        fp.write(f'<sentence id=\"{text_id}.s{index}\">\n')
        instance_id = 0
        for windex, word in enumerate(line.split()):
            tag, pos = pos_tag(word_tokenize(line), tagset='universal')[windex]
            if pos in pos_map: 
                fp.write(f'<instance id=\"{text_id}.s{index}.t{instance_id}\" lemma=\"{tag}\" pos=\"{pos}\">{word}</instance>\n') 
                instance_id+=1
        fp.write('</sentence>\n')
        fp.write('</text>\n')
        fp.write('</corpus>\n')

from tqdm import tqdm
import string
string1 = ''

def xml_for_dataset(dataset, filename):
    text_id = 'd000'
    instance_id = 0
    outfile = 'batch-owt-lines.data.xml' if filename is None else filename
    mindex = 0

    # as per the code in esc,only these are available
    pos_map = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'AFX', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'RB', 'RP', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WRB'}

    with open(outfile, 'w') as fp:
        fp.write('<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n')
        fp.write(f'<corpus lang=\"en\" source=\"{filename.split("/")[-1]}\">\n')
        fp.write(f'<text id=\"{text_id}\">\n')

        for index, row in tqdm(enumerate(dataset)):
            fp.write(f'<sentence id=\"{text_id}.s{index}\">\n')
            instance_id = 0

            for windex, word in enumerate(row.split()):
                tag, pos = pos_tag(word_tokenize(row), tagset='universal')[windex]

                if pos in pos_map:
                    if word not in string.punctuation and word != string1:
    #                    fp.write(f'<instance id=\"{text_id}.s{index}.t{instance_id}\" lemma=\"{tag}\" pos=\"{pos}\">{word}</instance>\n')
                        fp.write(f'<instance id=\"{text_id}.s{index}.t{windex}\" lemma=\"{tag}\" pos=\"{pos}\">{word}</instance>\n')
                        instance_id += 1
                #fp.write(f'<instance id=\"{text_id}.s{index}.t{instance_id}\" lemma=\"{tag}\" pos=\"{pos}\">{word}</instance>\n')
                #instance_id+=1
            fp.write('</sentence>\n')
        fp.write('</text>\n')
        fp.write('</corpus>\n')

def generate_concordance_file(dataset, filename, mode):
    text_id = 'd000'
    instance_id = 0
    if mode == 'premise':
        outfile = '100-lines-snli.xml'
        mindex = 0
    elif mode == 'hypothesis':
        outfile = '100-lines-snli.xml'
        mindex = 1

    # as per the code in esc,only these are available
    pos_map = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'AFX', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'RB', 'RP', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WRB'}

    with open(outfile, 'w') as fp:
        fp.write('<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n')
        fp.write(f'<corpus lang=\"en\" source=\"{filename.split("/")[-1]}\">\n')
        fp.write(f'<text id=\"{text_id}\">\n')

        for index, row in enumerate(dataset):
            fp.write(f'<sentence id=\"{text_id}.s{index}\">\n')
            instance_id = 0

            for windex, word in enumerate(row[0][mindex].split()):
                tag, pos = pos_tag(word_tokenize(row[0][mindex]), tagset='universal')[windex]

                #if pos in pos_map:
                #    fp.write(f'<instance id=\"{text_id}.s{index}.t{instance_id}\" lemma=\"{tag}\" pos=\"{pos}\">{word}</instance>\n')
                #    instance_id += 1
                fp.write(f'<instance id=\"{text_id}.s{index}.t{instance_id}\" lemma=\"{tag}\" pos=\"{pos}\">{word}</instance>\n')
                instance_id+=1
            fp.write('</sentence>\n')
        fp.write('</text>\n')
        fp.write('</corpus>\n')

if __name__ == "__main__":
    ds = get_dataset("../resources/datasets/snli/snli_1.0_train_100_lines.jsonl")
#    import ipdb;ipdb.set_trace()
    generate_concordance_file(ds, "snli_100_lines", mode="premise")
