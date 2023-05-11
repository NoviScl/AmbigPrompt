# Measuring Inductive Biases of In-Context Learning with Underspecified Demonstrations (ACL 2023)

This repository contains our code for running the experiments in our paper. It includes all the processed datasets as well as the script for running experiments with the OpenAI API. Please see below for more detailed instructions for running the code. 

<p align="center">
    <img src="TeaserFigure.png" width="70%" height="auto"/>
</p>

## Data

## Dependencies

## Running Experiments

## Evaluation and Sample Predictions

The raw metrics in `run.py` measure the frequency that model predicts `1` as the label. 

We provide an additional script `collect_results.py` that computes the h_1 accuracy and h_2 accuracy based on these raw results. They are referred to as `h1_ambig_acc` and `h2_ambig_acc` in the script. The script should automatically gather all results in `logs_ambiguous` and print out the h1 accuracy and h2 accuracy results.

## Citation

