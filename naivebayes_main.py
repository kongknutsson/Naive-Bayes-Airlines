import math
import random
import time
import pandas as pd
import numpy as np

 # returns a dict of words where the value is the frequency of the word.
def vocabulary_from(doc):
    vocab = {}
    for rating, entry in doc:
        entry_words = entry.split(" ")
        for word in entry_words:
            word = word.lower()
            vocab[word] = 1 if word not in vocab else vocab[word]+1
    return vocab

def train_naive_bayes_binary(documents, classes, vocab):
    # Assume here that classes[0] is what we are trying to find. (TARGET)
    # Therefore classes[1] is everything else than what we are trying to find.
    documents = documents.copy()
    for i in range(len(documents)):
        rating, entry = documents[i]
        if rating != classes[0]:
            documents[i] = (classes[1], entry)

    num_of_entries = len(documents)
    prior = {}      # The probability of an entry being in a class.
    bigdoc = {}     # All negative entries are contained in bigdoc[negative]
    likelihood = {} # likelihood[word][class] will return the chance that a word is a part of a specific class.

    # Adding class specifics
    for current_class in classes:
        class_occurence = 0
        bigdoc[current_class] = []
        for rating, entry in documents:
            if rating == current_class:
                class_occurence += 1
                if entry not in bigdoc[current_class]:
                    bigdoc[current_class].append(entry)

        prior[current_class] = (class_occurence/num_of_entries)

        for word in vocab:
            occurence_in_total = vocab[word]
            occurence_in_class = 0
            for entry in bigdoc[current_class]:
                occurence_in_class += entry.count(" " + word + " ")
            prob = (occurence_in_class + 1) / (occurence_in_total + 1)
            if word not in likelihood:
                likelihood[word] = {current_class:prob}
            else:
                likelihood[word][current_class] = prob

    return prior, likelihood

def test_naive_bayes(document, prior, likelihood, classes, vocab):
    summed = {}
    for current_class in classes:
        summed[current_class] = prior[current_class]
        words = document.split(" ")
        for word in words:
            if word in vocab:
                summed[current_class] += likelihood[word][current_class]
    max_val = 0
    max_class = ""
    for i in summed:
        if summed[i] > max_val:
            max_val = summed[i]
            max_class = i
    return max_class, max_val

def clean_up(file):
    f = pd.read_csv(file)
    f = f.fillna("na")
    f.to_csv(file,index=False)

def list_from_file(file):
    f = open(file, "r", encoding="utf8")
    i = 0
    lines = []
    for line in f:
        line = line.split(",")
        while len(line) > 15:
            line[10] = line[10] + line.pop(11)
        if len(line) < 15:
            i += 1
            continue
        airline_sentiment = line[1]
        text = line[10]
        lines.append((airline_sentiment, text))
        i += 1
    f.close()
    return lines

def split_training_test(dataset, seed=0):
    random.seed(seed)
    test_set = []
    for x in range(len(dataset)//4):     # takes 1/4th of the dataset and makes it the test set
        data_length = len(dataset)-1
        r = random.randint(0, data_length)
        new_test_data = dataset.pop(r)
        test_set.append(new_test_data)
    training_set = dataset      # the rest 3/4ths are used for training
    return training_set, test_set

# Removes entries from the dataset.
def shorten(dataset, n):
    pos_max = n
    neg_max = n
    neu_max = n
    tmp = []
    for rating, line in training_set:
        if rating == "negative" and neg_max > 0:
            tmp.append((rating, line))
            neg_max -= 1
        elif rating == "positive" and pos_max > 0:
            tmp.append((rating, line))
            pos_max -= 1
        elif rating == "neutral" and neu_max > 0:
            tmp.append((rating, line))
            neu_max -= 1
        if pos_max == 0 and neg_max == 0 and neu_max == 0:
            break
    return tmp

def add_negation_words(dataset, negation_words):
    for rating, line in dataset:
        index = 0
        line = line.split(" ")
        for word in line:
            if word in negation_words and index+1 < len(line):
                line[index+1] = "NOT_" + line[index+1]
            index += 1
    return dataset

def explanation_generator(tweet, prior, likelihood, cl, vocab):
    tweet = tweet.split()[1:]
    print("\n- prediction is", cl)
    print("- word       \t weight")
    for word in tweet:
        try:
            print(" ", word, "      \t", likelihood[word][cl])
        except:
            print(" ", word, "      \t", "not in vocab.")

if __name__ == "__main__":

    # We only train on some n of each class, because the training is so slow.
    # if you want all the data used just set to 15 000
    print("Num of entries from each class to train with.")
    n = int(input("(recommend between 100-1000): "))

    clean = input("\nDo you want to clean the dataset (if its uncleaned)?\ny/n: ")
    filename = "airlinedata.txt"
    if clean == "y":
        clean_up(filename)
        print("Dataset from file", filename, "cleaned.")

    start_time = time.time()
    dataset = list_from_file(filename)
    negation_words = ["dont", "don’t", "didnt", "didn’t", "not", "doesnt", "doesn’t"]
    dataset = add_negation_words(dataset, negation_words)
    print("\nDataset load \tT+", time.time()-start_time)

    training_set, test_set = split_training_test(dataset, seed=random.randint(0,10000))
    training_set = shorten(training_set, n)     # n is from input
    vocab = vocabulary_from(training_set)

    # Training 3 different classifiers. One for each class.
    positive = ("positive", "not positive")
    negative = ("negative", "not negative")
    neutral = ("neutral", "not neutral")
    pos_prior, pos_likelihood = train_naive_bayes_binary(training_set, positive, vocab)
    print("Trained pos \tT+", time.time()-start_time)
    neg_prior, neg_likelihood = train_naive_bayes_binary(training_set, negative, vocab)
    print("Trained neg \tT+", time.time()-start_time)
    neu_prior, neu_likelihood = train_naive_bayes_binary(training_set, neutral, vocab)
    print("Trained neu \tT+", time.time()-start_time)


    # NB: Here there be monsters. 


    # Bunch of variables for stats.
    correct = 0
    n_ratings = 0

    pred_negatives = 0
    corr_negatives = 0
    act_negatives = 0

    pred_positives = 0
    corr_positives = 0
    act_positives = 0

    pred_neutrals = 0
    corr_neutrals = 0
    act_neutrals = 0

    # starting testing.
    for rating, entry in test_set:
        pos_pred, pos_val = test_naive_bayes(entry, pos_prior, pos_likelihood, positive, vocab)
        neg_pred, neg_val = test_naive_bayes(entry, neg_prior, neg_likelihood, negative, vocab)
        neu_pred, neu_val = test_naive_bayes(entry, neu_prior, neu_likelihood, neutral, vocab)

        is_negative = True if neg_pred == "negative" else False
        is_positive = True if pos_pred == "positive" else False
        is_neutral = True if neu_pred == "neutral" else False

        # If all the algorithms "agree" on the right answer:
        if is_negative and not is_positive and not is_neutral:
            pred_negatives += 1
            prediction = "negative"
            if rating == prediction:
                corr_negatives += 1
        elif is_neutral and not is_negative and not is_positive:
            pred_neutrals += 1
            prediction = "neutral"
            if rating == prediction:
                corr_neutrals += 1
        elif is_positive and not is_negative and not is_neutral:
            pred_positives += 1
            prediction = "positive"
            if rating == prediction:
                corr_positives += 1
        elif is_negative == False and is_positive == False and is_neutral == False:
            # Everything is False, so we take the one with the least confidence
            # of being false as the prediction.
            min_conf = min(neu_val, neg_val, pos_val)
            if neu_val == min_conf:
                pred_neutrals += 1
                prediction = "neutral"
                if rating == prediction:
                    corr_neutrals += 1
            elif pos_val == min_conf:
                pred_positives += 1
                prediction = "positive"
                if rating == prediction:
                    corr_positives += 1
            elif neg_val == min_conf:
                pred_negatives += 1
                prediction = "negative"
                if rating == prediction:
                    corr_negatives += 1

        act_positives += 1 if rating == prediction else 0
        act_neutrals += 1 if rating == prediction else 0
        act_negatives += 1 if rating == prediction else 0

        correct += 1 if rating == prediction else 0
        n_ratings += 1

    neg_precision = round(corr_negatives/pred_negatives, 3)
    pos_precision = round(corr_positives/pred_positives, 3)
    neu_precision = round(corr_neutrals/pred_neutrals, 3)

    neg_recall = round(corr_negatives/act_negatives, 3)
    pos_recall = round(corr_positives/act_positives, 3)
    neu_recall = round(corr_neutrals/act_neutrals, 3)

    print("\nnum ratings:\t", n_ratings)
    print("corr ratings:\t", correct)
    print("accuracy:\t", round(correct/n_ratings, 4))
    print("\nclasses \t  neg \t  pos \t  neu")
    print("\nprecision \t", neg_precision, "\t", pos_precision, "\t", neu_precision)
    print("\nrecall  \t", neg_recall, "\t", pos_recall, "\t", neu_recall)

    tweet = input("\nWrite custom tweet to be checked: ")
    pos_pred, pos_val = test_naive_bayes(tweet, pos_prior, pos_likelihood, positive, vocab)
    neg_pred, neg_val = test_naive_bayes(tweet, neg_prior, neg_likelihood, negative, vocab)
    neu_pred, neu_val = test_naive_bayes(tweet, neu_prior, neu_likelihood, neutral, vocab)

    is_negative = True if neg_pred == "negative" else False
    is_positive = True if pos_pred == "positive" else False
    is_neutral = True if neu_pred == "neutral" else False

    # If all the algorithms "agree" on the right answer:
    if is_negative and not is_positive and not is_neutral:
        prediction = "negative"
        explanation_generator(tweet, neg_prior, neg_likelihood, prediction, vocab)
    elif is_neutral and not is_negative and not is_positive:
        prediction = "neutral"
        explanation_generator(tweet, neu_prior, neu_likelihood, prediction, vocab)
    elif is_positive and not is_negative and not is_neutral:
        prediction = "positive"
        explanation_generator(tweet, pos_prior, pos_likelihood, prediction, vocab)
    else:
        min_conf = min(neu_val, neg_val, pos_val)
        if neu_val == min_conf:
            prediction = "neutral"
            explanation_generator(tweet, neu_prior, neu_likelihood, prediction, vocab)
        elif pos_val == min_conf:
            prediction = "positive"
            explanation_generator(tweet, pos_prior, pos_likelihood, prediction, vocab)
        elif neg_val == min_conf:
            prediction = "negative"
            explanation_generator(tweet, neg_prior, neg_likelihood, prediction, vocab)
