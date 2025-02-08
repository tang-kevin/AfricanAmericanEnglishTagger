# Hypothesis 2

We used a logistic regression model to determine whether a sentence containing syntax error such as missing subject, sentence fragments, or run-on elements is associated with it being tagged with an AAVE feature by the LLM, controlling for whether it actually contains that AAVE feature. 

# Directory Structure

`Hypothesis 2 Regression.ipynb` was the file used to run the logistic regression models for each of the features for zero and few shot.

`Data` contains a series of .txt files, each containing the the LLM predictions for the labeled feature and zero/few shot input, as well as a Hyp_2 tag representing whether that sentence contains at least one of the aftermentioned syntax errors. 

The `Results` folder contains the model output for each of the files in `Data`.
