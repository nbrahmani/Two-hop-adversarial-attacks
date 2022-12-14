# Systematic Monotonicity and Consistency for Natural Language Inference

Implementatio of our paper 'Systematic Monotonicity and Consistency for Natural Language Infernece' at https://link.springer.com/chapter/10.1007/978-3-031-22695-3_25

## Prerequisites

* ccg2mono: Monotonicity information is extracted using this module. We use depccg parser to parse the sentences.
    * Refer https://github.com/huhailinguist/ccg2mono for installation.
* ESC: This module is used to extract the senses.
    * Refer https://github.com/SapienzaNLP/esc for installation.

## Attack Pipeline

### Get polarity annotations of the dataset using ccg2mono.
```
cd ccg
source env/bin/activate
python3 parser.py 'sentences.txt' depccg
```
### Extract senses using ESC
```
cd esc
source env/bin/activate
```
Generate semantic concordance file.
```
python3 gen_xml_file.py
```
Extract senses
```
python3 predict.py --ckpt escher_semcor_best.ckpt\
                   --dataset-paths batch.data.xml\
                   --prediction-types probabilistic\
                   --prediction-path predictions/
```
### Extract markers using the markers class in util.py

Use the Markers class in util.py to extract markers and store in a file, for premises and hypotheses.

### Attack the models

```
source env/bin/activate
python3 main.py
```

## Cite

```
@InProceedings{10.1007/978-3-031-22695-3_25,
author="Nutakki, Brahmani
and Badola, Akshay
and Padmanabhan, Vineet",
editor="Aziz, Haris
and Corr{\^e}a, D{\'e}bora
and French, Tim",
title="Systematic Monotonicity and Consistency for Adversarial Natural Language Inference",
booktitle="AI 2022: Advances in Artificial Intelligence",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="353--366",
isbn="978-3-031-22695-3"
}
```

## ToDo

* Add arguments for input filenames: ccg-parser.py, esc-gen_xml_file.py, transformations-main.py
* Add script for marker extraaction
* Clean code
