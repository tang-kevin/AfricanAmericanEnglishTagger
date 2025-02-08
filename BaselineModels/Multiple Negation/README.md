# Multiple Negation Classification with Tagger

This project uses a tagger to classify sentences as **multiple negation** (+1) or **non-multiple negation** (-1). The 'Multiple Negator Training Model Final.ipynb' script is used to tag sentences and generates evaluation metrics for the tagger's classification results. 

## Requirements

Ensure the following modules are loaded:

- `python`
- `numpy`
- `pandas`
- `sklearn`
- `nltk`
- `spacy`

## Directory Structure

The data file containing test data is called `test_multiple_negation.txt`. 

The results of the tagger on the testing set are displayed in `Multiple Negation Results on Testing Set.png`.

The predictions that the tagger generated for each sentences of the testing set are shown in `Multiple Negation Predictions for Testing Set.txt`.

`Multiple Negator Training Model Final.ipynb` is the tagger. 

Read `Multiple Negation Full and Final Documentation.docx` for more information.

## Running Evaluation

Run `Multiple Negator Training Model Final.ipynb` on `test_multiple_negation.txt`. The predictions and classification report results should align with those in `Multiple Negation Predictions for Testing Set.txt` and `Multiple Negation Results on Testing Set.png`, respectively. 
