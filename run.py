import os
import csv
import argparse
import numpy as np
import multiprocessing
import time
import json
import time
from time import sleep
import random
import openai
from tqdm import tqdm
from utils import *
from transformers import GPT2TokenizerFast
gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

import string
puncs = set(string.punctuation)

import random
random.seed(2022)

subset_mappings = {
    "nli_entailment_genre_gov_fic": "testsets_ambiguous/nli_entailment_genre_gov_fic.json",
    "nli_entailment_genre_gov_tele": "testsets_ambiguous/nli_entailment_genre_gov_tele.json",
    "nli_entailment_length": "testsets_ambiguous/nli_entailment_length.json",
    "nli_entailment_negation": "testsets_ambiguous/nli_entailment_negation.json",
    "nli_entailment_overlap": "testsets_ambiguous/nli_entailment_overlap.json",
    "sa_sentiment_domain": "testsets_ambiguous/sa_sentiment_domain.json",
    "sa_sentiment_length": "testsets_ambiguous/sa_sentiment_length.json",
    "sa_sentiment_lexicon_food": "testsets_ambiguous/sa_sentiment_lexicon_food.json",
    "sa_sentiment_lexicon_nice": "testsets_ambiguous/sa_sentiment_lexicon_nice.json",
    "sa_sentiment_punctuation": "testsets_ambiguous/sa_sentiment_punctuation.json",
    "sa_sentiment_uppercase": "testsets_ambiguous/sa_sentiment_uppercase.json",
    "comments_toxicity_gender": "testsets_ambiguous/comments_toxicity_gender.json",
    "comments_toxicity_gender_lgbtq": "testsets_ambiguous/comments_toxicity_gender_lgbtq.json",
    "comments_toxicity_length": "testsets_ambiguous/comments_toxicity_length.json",
    "comments_toxicity_race_black_white": "testsets_ambiguous/comments_toxicity_race_black_white.json",
    "comments_toxicity_religion_muslim_christian": "testsets_ambiguous/comments_toxicity_religion_muslim_christian.json",
    "comments_toxicity_religion_muslim_jewish": "testsets_ambiguous/comments_toxicity_religion_muslim_jewish.json",
    "comments_toxicity_uppercase": "testsets_ambiguous/comments_toxicity_uppercase.json",
    "boolq_answer_length": "testsets_ambiguous/boolq_answer_length.json",
    "boolq_answer_overlap": "testsets_ambiguous/boolq_answer_overlap.json",
    "boolq_answer_qword": "testsets_ambiguous/boolq_answer_qword.json",
    "boolq_answer_structure": "testsets_ambiguous/boolq_answer_structure.json",
    "cluster11_vs_others": "testsets_ambiguous_unknown/cluster11_vs_others.json",
    "cluster18_vs_others": "testsets_ambiguous_unknown/cluster18_vs_others.json",
    "cluster19_vs_others": "testsets_ambiguous_unknown/cluster19_vs_others.json",
    "civilcomments_cluster0_vs_others": "testsets_ambiguous_unknown/civilcomments_cluster0_vs_others.json",
    "civilcomments_cluster3_vs_others": "testsets_ambiguous_unknown/civilcomments_cluster3_vs_others.json",
}

z1_label_prefix_map = {
    "nli_entailment_genre_gov_fic": "Entailed?",
    "nli_entailment_genre_gov_tele": "Entailed?",
    "nli_entailment_length": "Entailed?",
    "nli_entailment_negation": "Entailed?",
    "nli_entailment_overlap":  "Entailed?",
    "comments_toxicity_gender": "Toxic?",
    "comments_toxicity_gender_lgbtq": "Toxic?",
    "comments_toxicity_length": "Toxic?",
    "comments_toxicity_race_black_white": "Toxic?",
    "comments_toxicity_religion_muslim_christian": "Toxic?",
    "comments_toxicity_religion_muslim_jewish": "Toxic?",
    "comments_toxicity_uppercase": "Toxic?",
    "boolq_answer_length": "Answer?",
    "boolq_answer_overlap": "Answer?",
    "boolq_answer_qword": "Answer?",
    "boolq_answer_structure": "Answer?",
}

z1_label_set_map = {
    "nli_entailment_genre_gov_fic": {'1': 'yes', '0': 'no'},
    "nli_entailment_genre_gov_tele": {'1': 'yes', '0': 'no'},
    "nli_entailment_length": {'1': 'yes', '0': 'no'},
    "nli_entailment_negation": {'1': 'yes', '0': 'no'},
    "nli_entailment_overlap": {'1': 'yes', '0': 'no'},
    "comments_toxicity_gender": {'1': 'yes', '0': 'no'},
    "comments_toxicity_gender_lgbtq": {'1': 'yes', '0': 'no'},
    "comments_toxicity_length": {'1': 'yes', '0': 'no'},
    "comments_toxicity_race_black_white": {'1': 'yes', '0': 'no'},
    "comments_toxicity_religion_muslim_christian": {'1': 'yes', '0': 'no'},
    "comments_toxicity_religion_muslim_jewish": {'1': 'yes', '0': 'no'},
    "comments_toxicity_uppercase": {'1': 'yes', '0': 'no'},
    "boolq_answer_length": {'1': 'yes', '0': 'no'},
    "boolq_answer_overlap": {'1': 'yes', '0': 'no'},
    "boolq_answer_qword": {'1': 'yes', '0': 'no'},
    "boolq_answer_structure": {'1': 'yes', '0': 'no'},
}

z2_label_prefix_map = {
    "nli_entailment_genre_gov_fic": "Source?",
    "nli_entailment_genre_gov_tele": "Source?",
    "nli_entailment_length": "Shorter?",
    "nli_entailment_negation": "Negation?",
    "nli_entailment_overlap": "Overlap?",
    "sa_sentiment_domain": "Source?",
    "sa_sentiment_length": "Length?",
    "sa_sentiment_lexicon_food": "Has 'food'?",
    "sa_sentiment_lexicon_nice": "Has 'nice'?",
    "sa_sentiment_punctuation": "End punctuation?",
    "sa_sentiment_uppercase": "Uppercase words?",
    "comments_toxicity_gender": "Gender?",
    "comments_toxicity_gender_lgbtq": "LGBTQ?",
    "comments_toxicity_length": "Length?",
    "comments_toxicity_race_black_white": "Race?",
    "comments_toxicity_religion_muslim_christian": "Religion?",
    "comments_toxicity_religion_muslim_jewish": "Religion?",
    "comments_toxicity_uppercase": "Uppercase words?",
    "boolq_answer_length": "Length?",
    "boolq_answer_overlap": "Overlap?",
    "boolq_answer_qword": "Question word?",
    "boolq_answer_structure": "Has 'same as'?",
}

z2_label_set_map = {
    "nli_entailment_genre_gov_fic": {'1': 'government', '0': 'fiction'},
    "nli_entailment_genre_gov_tele": {'1': 'government', '0': 'phone'},
    "nli_entailment_length": {'1': 'yes', '0': 'no'},
    "nli_entailment_negation": {'1': 'yes', '0': 'no'},
    "nli_entailment_overlap": {'1': 'yes', '0': 'no'},
    "sa_sentiment_domain": {'1': 'movie', '0': 'other'},
    "sa_sentiment_length": {'1': 'short', '0': 'long'},
    "sa_sentiment_lexicon_food": {'1': 'yes', '0': 'no'},
    "sa_sentiment_lexicon_nice": {'1': 'yes', '0': 'no'},
    "sa_sentiment_punctuation": {'1': 'other', '0': 'period'},
    "sa_sentiment_uppercase": {'1': 'yes', '0': 'no'},
    "comments_toxicity_gender": {'1': 'female', '0': 'male'},
    "comments_toxicity_gender_lgbtq": {'1': 'yes', '0': 'no'},
    "comments_toxicity_length": {'1': 'short', '0': 'long'},
    "comments_toxicity_race_black_white": {'1': 'black', '0': 'white'},
    "comments_toxicity_religion_muslim_christian": {'1': 'Muslim', '0': 'Christian'},
    "comments_toxicity_religion_muslim_jewish": {'1': 'Muslim', '0': 'Jewish'},
    "comments_toxicity_uppercase": {'1': 'yes', '0': 'no'},
    "boolq_answer_length": {'1': 'long', '0': 'short'},
    "boolq_answer_overlap": {'1': 'yes', '0': 'no'},
    "boolq_answer_qword": {'1': 'be', '0': 'do'},
    "boolq_answer_structure": {'1': 'yes', '0': 'no'},
}

z1_instruction_map = {
    "nli_entailment_genre_gov_fic": "In this task, you will be presented with a premise sentence (the first sentence) and a hypothesis sentence (the second sentence). Determine whether the premise sentence entails (implies) or does not entail the hypothesis sentence. Please answer with \"1\" for entailment and \"0\" for non-entailment.",
    "nli_entailment_genre_gov_tele": "In this task, you will be presented with a premise sentence (the first sentence) and a hypothesis sentence (the second sentence). Determine whether the premise sentence entails (implies) or does not entail the hypothesis sentence. Please answer with \"1\" for entailment and \"0\" for non-entailment.",
    "nli_entailment_length": "In this task, you will be presented with a premise sentence (the first sentence) and a hypothesis sentence (the second sentence). Determine whether the premise sentence entails (implies) or does not entail the hypothesis sentence. Please answer with \"1\" for entailment and \"0\" for non-entailment.",
    "nli_entailment_negation": "In this task, you will be presented with a premise sentence (the first sentence) and a hypothesis sentence (the second sentence). Determine whether the premise sentence entails (implies) or does not entail the hypothesis sentence. Please answer with \"1\" for entailment and \"0\" for non-entailment.",
    "nli_entailment_overlap": "In this task, you will be presented with a premise sentence (the first sentence) and a hypothesis sentence (the second sentence). Determine whether the premise sentence entails (implies) or does not entail the hypothesis sentence. Please answer with \"1\" for entailment and \"0\" for non-entailment.",
    "sa_sentiment_domain": "Given a review, you need to predict whether the review is good (positive) or bad (negative). Please answer with \"1\" for positive and \"0\" for negative.",
    "sa_sentiment_length": "Given a review, you need to predict whether the review is good (positive) or bad (negative). Please answer with \"1\" for positive and \"0\" for negative.",
    "sa_sentiment_lexicon_food": "Given a review, you need to predict whether the review is good (positive) or bad (negative). Please answer with \"1\" for positive and \"0\" for negative.",
    "sa_sentiment_lexicon_nice": "Given a review, you need to predict whether the review is good (positive) or bad (negative). Please answer with \"1\" for positive and \"0\" for negative.",
    "sa_sentiment_punctuation": "Given a review, you need to predict whether the review is good (positive) or bad (negative). Please answer with \"1\" for positive and \"0\" for negative.",
    "sa_sentiment_uppercase": "Given a review, you need to predict whether the review is good (positive) or bad (negative). Please answer with \"1\" for positive and \"0\" for negative.",
    "comments_toxicity_gender": "Categorize the comment on the basis of toxicity. If the comment is toxic (harmful, malicious, derogatory, threat, insult, identity attack, etc.) output \"1\", otherwise output \"0\".",
    "comments_toxicity_gender_lgbtq": "Categorize the comment on the basis of toxicity. If the comment is toxic (harmful, malicious, derogatory, threat, insult, identity attack, etc.) output \"1\", otherwise output \"0\".",
    "comments_toxicity_length": "Categorize the comment on the basis of toxicity. If the comment is toxic (harmful, malicious, derogatory, threat, insult, identity attack, etc.) output \"1\", otherwise output \"0\".",
    "comments_toxicity_race_black_white": "Categorize the comment on the basis of toxicity. If the comment is toxic (harmful, malicious, derogatory, threat, insult, identity attack, etc.) output \"1\", otherwise output \"0\".",
    "comments_toxicity_religion_muslim_christian": "Categorize the comment on the basis of toxicity. If the comment is toxic (harmful, malicious, derogatory, threat, insult, identity attack, etc.) output \"1\", otherwise output \"0\".",
    "comments_toxicity_religion_muslim_jewish": "Categorize the comment on the basis of toxicity. If the comment is toxic (harmful, malicious, derogatory, threat, insult, identity attack, etc.) output \"1\", otherwise output \"0\".",
    "comments_toxicity_uppercase": "Categorize the comment on the basis of toxicity. If the comment is toxic (harmful, malicious, derogatory, threat, insult, identity attack, etc.) output \"1\", otherwise output \"0\".",
    "boolq_answer_length": "Based on the information present in the given passage, decide whether the answer to the given question is yes or no. Please answer with \"1\" for yes and \"0\" for no.",
    "boolq_answer_overlap": "Based on the information present in the given passage, decide whether the answer to the given question is yes or no. Please answer with \"1\" for yes and \"0\" for no.",
    "boolq_answer_qword": "Based on the information present in the given passage, decide whether the answer to the given question is yes or no. Please answer with \"1\" for yes and \"0\" for no.",
    "boolq_answer_structure": "Based on the information present in the given passage, decide whether the answer to the given question is yes or no. Please answer with \"1\" for yes and \"0\" for no.",
}

z2_instruction_map = {
    "nli_entailment_genre_gov_fic": "In this task, you will be presented with a premise sentence (the first sentence) and a hypothesis sentence (the second sentence). Determine whether they come from government files or fictions. Please answer with \"1\" for government and \"0\" for fiction",
    "nli_entailment_genre_gov_tele": "In this task, you will be presented with a premise sentence (the first sentence) and a hypothesis sentence (the second sentence). Determine whether they come from government files or telephone recordings. Please answer with \"1\" for government and \"0\" for telephone.",
    "nli_entailment_length": "In this task, you will be presented with a premise sentence (the first sentence) and a hypothesis sentence (the second sentence). Determine whether the second sentence is shorter than the first sentence. Please answer with \"1\" for shorter and \"0\" for longer.",
    "nli_entailment_negation": "In this task, you will be presented with a premise sentence (the first sentence) and a hypothesis sentence (the second sentence). Determine whether there are any negation words in the second sentence (\"not\", \"no\", \"n't\"). Please answer with \"1\" for not having negations and \"0\" for having negations.",
    "nli_entailment_overlap": "In this task, you will be presented with a premise sentence (the first sentence) and a hypothesis sentence (the second sentence). Determine whether all words in the second sentence also appear in the first sentence. If so, answer \"1\"; if not, answer \"0\".",
    "sa_sentiment_domain": "Given a review, you need to predict whether it comes from IMDB movie reviews or Yelp restaurant and service reviews. Please answer with \"1\" for IMDB and \"0\" for Yelp.",
    "sa_sentiment_length": "Given a review, you need to predict whether it is longer than 40 words. Please answer with \"1\" for shorter than 40 words and \"0\" for longer.",
    "sa_sentiment_lexicon_food": "Given a review, you need to predict whether the review mentions the word \"food\". Please answer with \"1\" for mentioning \"food\" and \"0\" for not mentioning.",
    "sa_sentiment_lexicon_nice": "Given a review, you need to predict whether the review mentions the word \"nice\". Please answer with \"1\" for mentioning \"food\" and \"0\" for not mentioning.",
    "sa_sentiment_punctuation": "Given a review, you need to predict whether the review ends with an exclamation mark ('!') or period ('.'). Please answer with \"1\" for exclamation mark and \"0\" for period.",
    "sa_sentiment_uppercase": "Given a review, you need to predict whether the review contains any uppercase words where all letters are uppercased (like 'THIS'). Please answer with \"1\" for having uppercase words and \"0\" for not.",
    "comments_toxicity_gender": "Given a comment, you need to predict whether the comment targets males or females. Please answer with \"1\" for female and \"0\" for male.",
    "comments_toxicity_gender_lgbtq": "Given a comment, you need to predict whether the comment targets LGBTQ people. Please answer with \"1\" if it does and \"0\" if not.",
    "comments_toxicity_length": "Given a comment, you need to predict whether the comment is longer than 40 words. Please answer with \"1\" for shorter and \"0\" for longer.",
    "comments_toxicity_race_black_white": "Given a comment, you need to predict whether the comment targets black or white people. Please answer with \"1\" for black people and \"0\" for white people.",
    "comments_toxicity_religion_muslim_christian": "Given a comment, you need to predict whether the comment targets Muslim or Christian people. Please answer with \"1\" for Muslim and \"0\" for Christian.",
    "comments_toxicity_religion_muslim_jewish": "Given a comment, you need to predict whether the comment targets Muslim or Jewish people. Please answer with \"1\" for Muslim and \"0\" for Jewish.",
    "comments_toxicity_uppercase": "Given a comment, you need to predict whether the comment contains any uppercase words where all letters are uppercased (like 'THIS'). Please answer with \"1\" for having uppercase words and \"0\" for not.",
    "boolq_answer_length": "Given the passage and question, determine whether the passage is longer than 50 words. Please answer with \"1\" for longer than 50 words and \"0\" for shorter.",
    "boolq_answer_overlap": "Given the passage and question, determine whether all words in the question also appear in the passage. If so, answer \"1\"; if not, answer \"0\".",
    "boolq_answer_qword": "Given the passage and question, determine whether the question word is \"is/was\" or \"do/does/did\". Please answer with \"1\" for \"is/was\" and \"0\" for \"do/does/did\".",
    "boolq_answer_structure": "Given the passage and question, determine whether the question contains the phrase \"same as\". Please answer with \"1\" for having \"same as\" and \"0\" if not.",
}


z1_explanation_map = {
    "nli_entailment_genre_gov_fic": {"1": "The first sentence entails the second sentence. Therefore, the answer is 1.", "0": "The first sentence does not entail the second sentence. Therefore, the answer is 0."},
    "nli_entailment_genre_gov_tele": {"1": "The first sentence entails the second sentence. Therefore, the answer is 1.", "0": "The first sentence does not entail the second sentence. Therefore, the answer is 0."},
    "nli_entailment_length": {"1": "The first sentence entails the second sentence. Therefore, the answer is 1.", "0": "The first sentence does not entail the second sentence. Therefore, the answer is 0."},
    "nli_entailment_negation": {"1": "The first sentence entails the second sentence. Therefore, the answer is 1.", "0": "The first sentence does not entail the second sentence. Therefore, the answer is 0."},
    "nli_entailment_overlap": {"1": "The first sentence entails the second sentence. Therefore, the answer is 1.", "0": "The first sentence does not entail the second sentence. Therefore, the answer is 0."},
    "sa_sentiment_domain": {"1": "The review has a positive sentiment. Therefore, the answer is 1.", "0": "The review has a negative sentiment. Therefore, the answer is 0."},
    "sa_sentiment_length": {"1": "The review has a positive sentiment. Therefore, the answer is 1.", "0": "The review has a negative sentiment. Therefore, the answer is 0."},
    "sa_sentiment_lexicon_food": {"1": "The review has a positive sentiment. Therefore, the answer is 1.", "0": "The review has a negative sentiment. Therefore, the answer is 0."},
    "sa_sentiment_lexicon_nice": {"1": "The review has a positive sentiment. Therefore, the answer is 1.", "0": "The review has a negative sentiment. Therefore, the answer is 0."},
    "sa_sentiment_punctuation": {"1": "The review has a positive sentiment. Therefore, the answer is 1.", "0": "The review has a negative sentiment. Therefore, the answer is 0."},
    "sa_sentiment_uppercase": {"1": "The review has a positive sentiment. Therefore, the answer is 1.", "0": "The review has a negative sentiment. Therefore, the answer is 0."},
    "comments_toxicity_gender": {"1": "The comment is toxic. Therefore, the answer is 1.", "0": "The comment is not toxic. Therefore, the answer is 0."},
    "comments_toxicity_gender_lgbtq": {"1": "The comment is toxic. Therefore, the answer is 1.", "0": "The comment is not toxic. Therefore, the answer is 0."},
    "comments_toxicity_length": {"1": "The comment is toxic. Therefore, the answer is 1.", "0": "The comment is not toxic. Therefore, the answer is 0."},
    "comments_toxicity_race_black_white": {"1": "The comment is toxic. Therefore, the answer is 1.", "0": "The comment is not toxic. Therefore, the answer is 0."},
    "comments_toxicity_religion_muslim_christian": {"1": "The comment is toxic. Therefore, the answer is 1.", "0": "The comment is not toxic. Therefore, the answer is 0."},
    "comments_toxicity_religion_muslim_jewish": {"1": "The comment is toxic. Therefore, the answer is 1.", "0": "The comment is not toxic. Therefore, the answer is 0."},
    "comments_toxicity_uppercase": {"1": "The comment is toxic. Therefore, the answer is 1.", "0": "The comment is not toxic. Therefore, the answer is 0."},
    "boolq_answer_length": {"1": "The answer to the question is yes. Therefore, the answer is 1.", "0": "The answer to the question is no. Therefore, the answer is 0."},
    "boolq_answer_overlap": {"1": "The answer to the question is yes. Therefore, the answer is 1.", "0": "The answer to the question is no. Therefore, the answer is 0."},
    "boolq_answer_qword": {"1": "The answer to the question is yes. Therefore, the answer is 1.", "0": "The answer to the question is no. Therefore, the answer is 0."},
    "boolq_answer_structure": {"1": "The answer to the question is yes. Therefore, the answer is 1.", "0": "The answer to the question is no. Therefore, the answer is 0."},
}

z2_explanation_map = {
    "nli_entailment_genre_gov_fic": {"1": "The text is from government files. Therefore, the answer is 1.", "0": "The text is from fictions. Therefore, the answer is 0."},
    "nli_entailment_genre_gov_tele": {"1": "The text is from government files. Therefore, the answer is 1.", "0": "The text is from telephone recordings. Therefore, the answer is 0."},
    "nli_entailment_length": {"1": "The second sentence is shorter than the first sentence. Therefore, the answer is 1.", "0": "The second sentence is longer than the first sentence. Therefore, the answer is 0."},
    "nli_entailment_negation": {"1": "The second sentence contains negation words. Therefore, the answer is 1.", "0": "The second sentence does not contain negation words. Therefore, the answer is 0."},
    "nli_entailment_overlap": {"1": "All words from the second sentence also appear in the first sentence. Therefore, the answer is 1.", "0": "Not all words from the second sentence also appear in the first sentence. Therefore, the answer is 0."},
    "sa_sentiment_domain": {"1": "The review is from IMDB movie reviews. Therefore, the answer is 1.", "0": "The review is from Yelp reviews. Therefore, the answer is 0."},
    "sa_sentiment_length": {"1": "The review is shorter than 40 words. Therefore, the answer is 1.", "0": "The review is longer than 40 words. Therefore, the answer is 0."},
    "sa_sentiment_lexicon_food": {"1": "The review contains the word 'food'. Therefore, the answer is 1.", "0": "The review does not contain the word 'food'. Therefore, the answer is 0."},
    "sa_sentiment_lexicon_nice": {"1": "The review contains the word 'nice'. Therefore, the answer is 1.", "0": "The review does not contain the word 'nice'. Therefore, the answer is 0."},
    "sa_sentiment_punctuation": {"1": "The review ends with an exclamation mark ('!'). Therefore, the answer is 1.", "0": "The review ends with a period ('.'). Therefore, the answer is 0."},
    "sa_sentiment_uppercase": {"1": "The review contains an uppercase word with all uppercase letters. Therefore, the answer is 1.", "0": "The review does not contain an uppercase word with all uppercase letters. Therefore, the answer is 0."},
    "comments_toxicity_gender": {"1": "The comment mentions females. Therefore, the answer is 1.", "0": "The comment mentions males. Therefore, the answer is 0."},
    "comments_toxicity_gender_lgbtq": {"1": "The comment mentions LGBTQ. Therefore, the answer is 1.", "0": "The comment does not mention LGBTQ. Therefore, the answer is 0."},
    "comments_toxicity_length": {"1": "The comment is shorter than 40 words. Therefore, the answer is 1.", "0": "The comment is longer than 40 words. Therefore, the answer is 0."},
    "comments_toxicity_race_black_white": {"1": "The comment mentions black people. Therefore, the answer is 1.", "0": "The comment mentions white people. Therefore, the answer is 0."},
    "comments_toxicity_religion_muslim_christian": {"1": "The comment mentions Muslim people. Therefore, the answer is 1.", "0": "The comment mentions Christian people. Therefore, the answer is 0."},
    "comments_toxicity_religion_muslim_jewish": {"1": "The comment mentions Muslim people. Therefore, the answer is 1.", "0": "The comment mentions Jewish people. Therefore, the answer is 0."},
    "comments_toxicity_uppercase": {"1": "The comment contains an uppercase word with all uppercase letters. Therefore, the answer is 1.", "0": "The comment contains an uppercase word with all uppercase letters. Therefore, the answer is 0."},
    "boolq_answer_length": {"1": "The passage is longer than 50 words. Therefore, the answer is 1.", "0": "The passage is shorter than 50 words. Therefore, the answer is 0."},
    "boolq_answer_overlap": {"1": "All words from the question also appear in the passage. Therefore, the answer is 1.", "0": "Not all words from the question also appear in the passage. Therefore, the answer is 0."},
    "boolq_answer_qword": {"1": "The question word is 'is' or 'was'. Therefore, the answer is 1.", "0": "The question word is 'do' or 'does' or 'did'. Therefore, the answer is 0."},
    "boolq_answer_structure": {"1": "The question contains the phrase 'same as'. Therefore, the answer is 1.", "0": "The question does not contain the phrase 'same as'. Therefore, the answer is 0."},
}

mnli_label_map = {
    '1': "yes",
    '0': "no"
}

def PromptStep(args, prompt, label_set, temp=0.0):
    stoplist = ["<|endoftext|>", "\n\n"]
    
    ## length truncation
    tokenized = gpt_tokenizer.convert_tokens_to_ids(gpt_tokenizer.tokenize(prompt))
    
    if args.engine in ["davinci", "curie"] and len(tokenized) >= 2000:
        prompt = gpt_tokenizer.decode(tokenized[-2000 : ])

    if args.engine in ["text-davinci-002"] and len(tokenized) >= 4000:
        prompt = gpt_tokenizer.decode(tokenized[-4000 : ])

    ## base models can't end with spaces
    if args.engine in ["davinci", "curie"]:
        prompt = prompt.strip()

    ## A Single Prompt Step
    response = None
    ite = 0
    while response is None and ite < 10:
        ite += 1
        try:
            response = openai.Completion.create(
                engine=args.engine,
                prompt=prompt,
                max_tokens=args.maxlen,
                logprobs=5,
                temperature=temp,
                stream=False,
                stop=stoplist
            )
        except:
            sleep(10)
            continue

    # print ("response: ", response)
    label_probs = [0] * len(label_set)  ## initialized as small values
    # print ("top logprob: ", response['choices'][0]["logprobs"]["top_logprobs"])
    for i, label in enumerate(label_set):
        if label in response['choices'][0]["logprobs"]["top_logprobs"][0]:
            label_probs[i] = np.exp(response['choices'][0]["logprobs"]["top_logprobs"][0][label])

    output = response['choices'][0]["text"].strip()
    return output, prompt, label_probs

def SelfConPrompt(args, counter, prompt, eg,):
    if args.print:
        print ("**********     Question #{}    **********".format(str(counter)))
        print (prompt)
    
    all_outputs = []
    ## self-consistency prompting
    ## we sample 10 different outputs with temperature 0.7
    for i in range(10):
        output, newprompt, _ = PromptStep(args, prompt, temp=0.7)
        ans = answer_extract_mapping[args.task](output)
        
        if args.print:
            print ("\nOutput #{}".format(str(i+1)))
            print (output)
            print ("\nExtracted answer string: ", ans)
        
        ## exclude no-answer cases
        if ans is not None:
            all_outputs.append(ans)
    
    final_ans = most_common(all_outputs)
    gold = eg["answer"]
    
    em = single_ans_em(final_ans, gold)
    f1 = single_ans_f1(final_ans, gold)
    
    if args.print:
        print ("\n\nQuestion #{} Summary: ".format(str(counter)))
        print ("All predicted answers: ", all_outputs)
        print ("Final prediction: ", final_ans)
        print ("Prob (frequency): ", all_outputs.count(final_ans) / len(all_outputs))
        print ("Gold answer: ", gold)
        print ("EM: ", em)
        print ("F1: ", f1)
        print ("\n\n")
    
    return em, f1

def SinglePrompt(args, counter, prompt, eg, label_set, empty_label_probs=None):
    ## greedy decoding by default
    output, newprompt, label_probs = PromptStep(args, prompt, label_set, temp=0.)
    # print ("label_probs: ", label_probs)
    label_probs = label_probs / np.sum(np.array(label_probs)) # normalize to 1
    raw_label_probs = label_probs

    if empty_label_probs is not None:
        ## calibrate before use
        
        W = np.linalg.inv(np.identity(len(label_set)) * empty_label_probs)
        b = np.zeros([len(label_set), 1])
        # print ("W: ", W, "b: ", b)

        # W = np.identity(len(label_set))
        # b = -1 * np.expand_dims(empty_label_probs, axis=-1)
        # print ("W: ", W, "b: ", b)

        label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b
        ans_label = np.argmax(label_probs, axis=0)[0]
        output = label_set[ans_label]
    
    ## we measure the freuqency of predicting '1' as the answer
    ## we have a separate script for computing the h-accuracy based on this
    gold = label_set[1] 
    em = 0
    
    orig_output = output[:]
    prefix = "answer is "
    if prefix in output:
        idx = output.rfind(prefix)
        output = output[idx + len(prefix) : ]

    em = single_ans_em(output, gold)
        
    if args.print:
        print ("**********     Question #{}    **********".format(str(counter)))
        # print (newprompt + orig_output)

        if counter <= 1:
            print (newprompt + orig_output)
        else:
            print ("Input:\n"+eg["question"].strip())
            if args.verbalizer_z1:
                print ("\n"+z1_label_prefix_map[args.task] + orig_output)
            elif args.verbalizer_z2:
                print ("\n"+z2_label_prefix_map[args.task] + orig_output)
            else:
                print ("\nCategory:" + orig_output)
        
        print ("\nGold answer: ", gold)
        if args.save_prob:
            print ("raw_label_probs: ", raw_label_probs)
            print ("label_probs: ", label_probs)
        print ("EM ", em)
        print ("\n\n")
    
    return em


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apikey', type=str, required=True, help='api key; https://openai.com/api/')
    parser.add_argument('--engine', type=str, default='davinci', help='api engine; https://openai.com/api/')
    parser.add_argument('--task', type=str, help='specify the task that you want to evaluate')
    parser.add_argument('--prompt_source', type=str, help='specify the where the prompt demos should come from')
    parser.add_argument('--prompt_method', type=str, default=None, help='specify the prompting method', choices=["zeroshot", "zeroshot-step", "fewshot", "fewshot-selfcon", "fewshot-cot", "fewshot-cot-selfcon"])
    parser.add_argument('--print', default=False, action='store_true', help='Whether to print out every prompt')
    parser.add_argument('--extract', default=False, action='store_true', help='Whether to add an additional answer extraction step')
    parser.add_argument('--subset', default=False, action='store_true', help='Whether to use a small subset for debugging')
    parser.add_argument('--subset_size', type=int, default=32, help='how many examples to sample for quick evaluation')
    parser.add_argument('--maxlen', type=int, default=256, help='max number of tokens to be generated')
    parser.add_argument('--shots', type=int, default=0, help='how many demos to use in the prompt')
    parser.add_argument('--no_unanswerable', default=False, action='store_true', help='Whether to filter out unanswerable questions in the demo')
    parser.add_argument('--label_shuffle', default=False, action='store_true', help='Whether to shuffle the gold labels')
    parser.add_argument('--save_prob', default=False, action='store_true', help='Whether to save top token logprobs and perplexity')
    parser.add_argument('--continue_from', type=int, default=0, help='evaluate on part of test set, starting from this index')
    parser.add_argument('--demo_index', type=int, default=0, help='used to select different demo examples')
    parser.add_argument('--test_split', type=str, default=None, help='specify which test split to eval on', choices=["testset_0_0", "testset_0_1", "testset_1_1", "testset_1_0"])
    parser.add_argument('--verbalizer', default=False, action='store_true', help='Whether to use semantically meaningful label words')
    parser.add_argument('--instruction', default=False, action='store_true', help='Whether to add instructions.')
    parser.add_argument('--disambig_z1', default=False, action='store_true', help='Add some disambig examples to the demo (4/4/4/4). Support z1.')
    parser.add_argument('--disambig_z2', default=False, action='store_true', help='Add some disambig examples to the demo (4/4/4/4). Support z2.')
    parser.add_argument('--verbalizer_z1', default=False, action='store_true', help='Use yes/no as verbalizers.')
    parser.add_argument('--verbalizer_z2', default=False, action='store_true', help='Use yes/no as verbalizers.')
    parser.add_argument('--instruction_z1', default=False, action='store_true', help='Use yes/no as verbalizers.')
    parser.add_argument('--instruction_z2', default=False, action='store_true', help='Use yes/no as verbalizers.')
    parser.add_argument('--explanation_z1', default=False, action='store_true', help='Use yes/no as verbalizers.')
    parser.add_argument('--explanation_z2', default=False, action='store_true', help='Use yes/no as verbalizers.')

    args = parser.parse_args()
    openai.api_key = args.apikey

    all_em = 0
    all_f1 = 0

    if args.task in subset_mappings:
        ## load test set
        task_dir = subset_mappings[args.task]
        with open(task_dir, "r") as f:
            data = json.load(f)
        test_set = data[args.test_split]
        if args.subset:
            test_set = test_set[ : args.subset_size]
        print ("Size of test set:", len(test_set))

        ## load demos
        if args.disambig_z1:
            demos_1_1 = data["demos_1_1"][args.demo_index : args.demo_index + int(args.shots // 4)]
            demos_0_0 = data["demos_0_0"][args.demo_index : args.demo_index + int(args.shots // 4)]
            demos_1_0 = data["testset_1_0"][args.demo_index : args.demo_index + int(args.shots // 4)]
            demos_0_1 = data["testset_0_1"][args.demo_index : args.demo_index + int(args.shots // 4)]
            for i in range(len(demos_1_0)):
                demos_1_0[i]["answer"] = "1"
            for i in range(len(demos_0_1)):
                demos_0_1[i]["answer"] = "0"
            demos = demos_1_1 + demos_0_0 + demos_1_0 + demos_0_1
        elif args.disambig_z2:
            demos_1_1 = data["demos_1_1"][args.demo_index : args.demo_index + int(args.shots // 4)]
            demos_0_0 = data["demos_0_0"][args.demo_index : args.demo_index + int(args.shots // 4)]
            demos_1_0 = data["testset_1_0"][args.demo_index : args.demo_index + int(args.shots // 4)]
            demos_0_1 = data["testset_0_1"][args.demo_index : args.demo_index + int(args.shots // 4)]
            for i in range(len(demos_1_0)):
                demos_1_0[i]["answer"] = "0"
            for i in range(len(demos_0_1)):
                demos_0_1[i]["answer"] = "1"
            demos = demos_1_1 + demos_0_0 + demos_1_0 + demos_0_1
        else:
            demos_1_1 = data["demos_1_1"][args.demo_index : args.demo_index + int(args.shots // 2)]
            demos_0_0 = data["demos_0_0"][args.demo_index : args.demo_index + int(args.shots // 2)]
            demos = demos_1_1 + demos_0_0 
        random.shuffle(demos)
        print ("#shots: ", len(demos))

        if args.verbalizer_z1:
            label_set = [z1_label_set_map[args.task]['0'], z1_label_set_map[args.task]['1']]
        elif args.verbalizer_z2:
            label_set = [z2_label_set_map[args.task]['0'], z2_label_set_map[args.task]['1']]
        else:
            label_set = ['0', '1']
        print ("label_set: ", label_set)

        if args.verbalizer_z1:
            verbalizer_map = z1_label_set_map[args.task]
        elif args.verbalizer_z2:
            verbalizer_map = z2_label_set_map[args.task]
        elif args.explanation_z1:
            verbalizer_map = z1_explanation_map[args.task]
        elif args.explanation_z2:
            verbalizer_map = z2_explanation_map[args.task]
        else:
            verbalizer_map = {"1": "1", "0": "0"}
        
        # print (demos)
        # print (test_set[:10])
        for i in range(len(demos)):
            demos[i]["answer"] = verbalizer_map[demos[i]["answer"]]
        
        # print (demos)
        # print (test_set[:10])
        for i in range(len(test_set)):
            # print (test_set[i]["answer"])
            test_set[i]["answer"] = verbalizer_map[test_set[i]["answer"]]

    else:
        print ("Task is out of our data collection")
        return 

    test_set = test_set[ args.continue_from : ]
    counter = args.continue_from
    demos_questions = [d["question"].strip() for d in demos]

    # prefix = "" ## shared prefix for all test examples
    # if args.instruction and "mnli" in args.task:
    #     prefix = "Your task is to decide whether the second sentence is entailed or implied by the first sentence. Answer yes if so, no otherwise. You should rely on the relationship between the sentences for making the prediction.\n\n"
    # else:
    #     prefix = ""
    if args.instruction_z1:
        prefix = z1_instruction_map[args.task] + "\n\n"
    elif args.instruction_z2:
        prefix = z2_instruction_map[args.task] + "\n\n"
    else:
        prefix = ""

    ## few-shot demo
    if args.prompt_method in ["fewshot", "fewshot-selfcon", "fewshot-cot", "fewshot-cot-selfcon"]:
        if args.label_shuffle:
            labels = [d["answer"] for d in demos]
            random.shuffle(labels)
            for i in range(len(labels)):
                demos[i]["answer"] = labels[i]

        for demo in demos:
            # prefix += "Input:"
            demo["question"] = demo["question"].strip()

            if args.verbalizer:
                if "mnli" in args.task:
                    if demo["question"][-1] in puncs:
                        prefix += demo["question"][:-1] + '?\n'
                    else:
                        prefix += demo["question"] + '?\n'
                    answer = mnli_label_map[demo["answer"]]

            elif args.instruction:
                if "mnli" in args.task:
                    if demo["question"][-1] in puncs:
                        prefix += demo["question"][:-1] + '?\n'
                    else:
                        prefix += demo["question"] + '?\n'
                    answer = mnli_label_map[demo["answer"]]

            else:
                prefix += demo["question"] + "\n"
                answer = demo["answer"]
            
            if args.verbalizer_z1:
                prefix += "\n" + z1_label_prefix_map[args.task] + answer.strip() + "\n\n"
            elif args.verbalizer_z2:
                prefix += "\n" + z2_label_prefix_map[args.task] + answer.strip() + "\n\n"
            else:
                prefix += "\nCategory:" + answer.strip() + "\n\n"
                
            if args.prompt_method == "fewshot-cot":
                ## with cot
                prompt += "Answer: "
                if "cot" in demo:
                    prompt += demo["cot"]
                    prompt += "\n"
                    prompt += "Therefore, the final answer is " + answer.strip() + "\n\n"
                elif "explanation" in demo:
                    prompt += demo["explanation"]
                    prompt += "\n"
                    prompt += "Therefore, the final answer is " + answer.strip() + "\n\n"
                else:
                    prompt += answer.strip() + "\n\n"


    # calibrate before use
    empty_templates = ["[MASK]", "N/A", ""]
    if args.explanation_z1 or args.explanation_z2:
        ## calibration not applicable for generating explanations
        empty_label_probs = None
    else:
        empty_label_probs = [0] * len(label_set)
        for templetate in empty_templates:
            prompt = prefix
            # prompt += "Input:"
            prompt += templetate
            if args.verbalizer_z1:
                prompt += "\n\n" + z1_label_prefix_map[args.task]
            elif args.verbalizer_z2:
                prompt += "\n\n" + z2_label_prefix_map[args.task]
            else:
                prompt += "\n\nCategory:"
            # print (prompt)
            try:
                output, newprompt, label_probs = PromptStep(args, prompt, label_set, temp=0.)
            except:
                continue
            # print ("label_probs (empty): ", label_probs)
            if 0 in label_probs:
                continue
            for i in range(len(label_probs)):
                empty_label_probs[i] += label_probs[i]
        for i in range(len(empty_label_probs)):
            empty_label_probs[i] = empty_label_probs[i] / len(empty_templates)
        empty_label_probs = empty_label_probs / sum(empty_label_probs) ## normalize
        empty_label_probs = np.array(empty_label_probs)
    print ("empty_label_probs: ", empty_label_probs)
    # empty_label_probs = None

    
    ## eval on actual test set
    for eg in tqdm(test_set):
        prompt = prefix
        
        if eg["question"].strip() in demos_questions:
            continue

        counter += 1
        
        ## current test instance 
        if len(eg["question"].split()) > 400 and "text" in args.engine:
            eg["question"] = ' '.join(eg["question"].split()[-400 : ])
        
        # prompt += "Input:"
        eg["question"] = eg["question"].strip()
        if args.verbalizer:
            if "mnli" in args.task:
                if eg["question"][-1] in puncs:
                    prompt += eg["question"][:-1] + '?\n'
                else:
                    prompt += eg["question"] + '?\n'
                
                eg["answer"] = mnli_label_map[eg["answer"]]

        elif args.instruction:
        #     if "mnli" in args.task:
        #         if eg["question"][-1] in puncs:
        #             prompt += eg["question"][:-1].replace("\n", "\nDoes this mean ") + '?\n'
        #         else:
        #             prompt += eg["question"].replace("\n", "\nDoes this mean ") + '?\n'
                
        #         eg["answer"] = mnli_label_map[eg["answer"]]
            if "mnli" in args.task:
                if eg["question"][-1] in puncs:
                    prompt += eg["question"][:-1] + '?\n'
                else:
                    prompt += eg["question"] + '?\n'
                
                eg["answer"] = mnli_label_map[eg["answer"]]

        else:
            prompt += eg["question"]  + "\n"
        
        # if args.engine == "davinci" and "mmlu" in args.task:
        if args.verbalizer_z1:
            prompt += "\n" + z1_label_prefix_map[args.task]
        elif args.verbalizer_z2:
            prompt += "\n" + z2_label_prefix_map[args.task]
        else:
            prompt += "\nCategory:"
        
        # elif args.engine in ["davinci", "curie"]:
        #     prompt += "Answer:"
        # elif args.task not in ["subqa-goldsub1", "subqa-goldqa1", "subqa-step1"]:
        #     prompt += "Answer: "


        # if "cot" in args.prompt_method:
        #     prompt += "Let's think step by step. "

        # if args.prompt_method in ["zeroshot-step", "fewshot-cot", "fewshot-cot-selfcon"]:
        #     prompt += "Let’s think step by step. "
        
        # if args.prompt_method == "fewshot-cot":
        #     prompt += "Based on the fact that "
        # else:
        #     prompt += "Answer: "
        
        if "selfcon" in args.prompt_method:
            em = SelfConPrompt(args, counter, prompt, eg)
        else:
            em = SinglePrompt(args, counter, prompt, eg, label_set, empty_label_probs)

        all_em += em 
    
    print ("EM: {}/{}={}%".format(all_em, counter, all_em / counter * 100))
    

if __name__ == '__main__':
    main()