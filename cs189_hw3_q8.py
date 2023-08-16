# -*- coding: utf-8 -*-
"""cs189_hw3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gqu-Ttee1zYZ_a_2sOTFRf3StAd_dNXE
"""

from google.colab import drive
drive.mount('/content/gdrive')
!unzip -q "/content/gdrive/My Drive/Sophomore/hw3.zip"

cd hw3-2023

cd scripts

ls

import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from numpy import linalg as LA
from scipy.stats import norm
from   scipy.stats import multivariate_normal

np.random.seed(42)

mnist = np.load(f"../data/mnist-data-hw3.npz")

#get training data and training label
train_data = mnist["training_data"]
train_labels = mnist["training_labels"]

# part a
from scipy.cluster.vq import whiten
from scipy.optimize import curve_fit

digits = np.unique(train_labels)
gaussian_fits = {}

train_data_normalized = train_data / np.linalg.norm(train_data, axis= 0) [:, None]
data_normalized_reshaped = train_data_normalized.reshape((-1, 28*28))
# print("shape is", data_normalized_reshaped)
#now fit gaussian for each digit by computing mu and covariance matrix
def fit(tr_data, labels, digit):
  #select only data on that specific digit
  idx = (labels == digit).flatten()
  digit_data = tr_data[idx][:]

  #compute mean and covariance
  mu = np.mean(digit_data, axis = 0)
  cov = np.cov(digit_data.T)
  return (mu, cov)

for digit in digits:
  gaussian_fits[digit] = fit(data_normalized_reshaped, train_labels, digit)

# part b
plt.figure(figsize=(10,10))
cov_plot = gaussian_fits[7][1]
cov_plot[np.isnan(cov_plot)] = 0
plt.imshow(cov_plot)
plt.colorbar()
plt.show()

#from hw1
def shuffle_and_partition(data, labels, val_size):
  data_size = len(data)
  if val_size < 1.0:
    val_size = int(data_size * val_size)
  training_size = data_size - val_size
  
  index_perm = np.random.permutation(data_size)
  train_index = index_perm[:training_size]
  val_index= index_perm[training_size:]
  train_pts = data[train_index]
  train_labels = labels[train_index]
  val_pts = data[val_index]
  val_labels = labels[val_index]

  return train_pts, train_labels, val_pts, val_labels

train_pts, train_lbs, val_pts, val_lbs = shuffle_and_partition(
    train_data.reshape((-1, 28 * 28)), train_labels, 10000)

from numpy.ma.core import argmax
import math
import scipy
#function for normalizing data
def normalize_data(data):
  norm_data = data / np.linalg.norm(data, axis= 0) [:, None]
  return norm_data

#function for fitting an lda with training set (given size) and train_labels
def lda_fit(data, labels):
  #need to compute mu_c and covariance
  number_of_each_digit = {}
  mu_for_each_digit = {}
  digits_present = np.unique(labels)
  d = data.shape[1]
  n = data.shape[0]
  pooled_cov = np.zeros((d, d))
  for digit in digits_present:
    data_for_this_digit = data[labels == digit] #get the dataset for each digit
    number_of_each_digit[digit] = data_for_this_digit.shape[0]
    mu_c = np.mean(data_for_this_digit, axis=0)
    mu_for_each_digit[digit] = mu_c
    cov = np.dot((data_for_this_digit - mu_c).T, data_for_this_digit - mu_c)
    pooled_cov = np.add(pooled_cov, cov)
  pooled_cov = pooled_cov / n
  #now to compute prior probabilities
  prior = {}
  for digit in digits_present:
    prior[digit] = number_of_each_digit[digit] / n
  return prior, mu_for_each_digit, pooled_cov

def lda_predict(test_data, prior, mu_for_each_digit, pooled_cov):
  q = []
  for digit in range(10):
    alpha = np.linalg.eig(pooled_cov)[0].min()
    matrix = np.identity(784) * (-1) * alpha

    pooled_cov = np.add(pooled_cov, matrix)
    q.append(scipy.stats.multivariate_normal.logpdf(
        test_data, allow_singular=True, cov=pooled_cov, 
        mean=mu_for_each_digit[digit]) + math.log(prior[digit]))
  predicted_labels = argmax(q, axis = 0)

  return predicted_labels

def error_rate(predicted_labels, actual_labels):
  return (1 - np.sum(predicted_labels == actual_labels) / len(actual_labels))

train_size = [100, 200, 500, 1000, 2000, 5000, 10000, 30000]
error_rates_lda = []
predictions_lda = []
for size in train_size:
  tr_pts = train_pts[:size]
  tr_labels = train_lbs[:size]
  prior, mu_for_each_digit, pooled_cov = lda_fit(tr_pts, tr_labels)
  predicted_labels = lda_predict(val_pts, prior, mu_for_each_digit, pooled_cov)
  error_rates_lda.append(error_rate(predicted_labels, val_lbs))
  predictions_lda.append(predicted_labels)

plt.figure()
plt.plot(train_size, error_rates_lda)
plt.title("MNIST Validation Sets Error Rate vs Training Size Using LDA")
plt.xlabel("Training Size")
plt.ylabel("Error Rate")

# part 3 b) QDA
def qda_fit(data, labels):
  #need to compute mu_c and covariance
  number_of_each_digit = {}
  mu_for_each_digit = {}
  cov_for_each_digit = {}
  digits_present = np.unique(labels)
  d = data.shape[1]
  n = data.shape[0]
  for digit in digits_present:
    data_for_this_digit = data[labels == digit] #get the dataset for each digit
    number_of_each_digit[digit] = data_for_this_digit.shape[0]
    mu_c = np.mean(data_for_this_digit, axis=0)
    mu_for_each_digit[digit] = mu_c
    cov = np.cov(data_for_this_digit, rowvar = False)
    cov_for_each_digit[digit] = cov
  #now to compute prior probabilities
  prior = {}
  for digit in digits_present:
    prior[digit] = number_of_each_digit[digit] / n
  return prior, mu_for_each_digit, cov_for_each_digit

def qda_predict(test_data, prior, mu_for_each_digit, cov_for_each_digit):
  q = []
  for digit in range(10):
    cov = cov_for_each_digit[digit]
    q.append(scipy.stats.multivariate_normal.logpdf(
        test_data, allow_singular=True, cov=cov, 
        mean=mu_for_each_digit[digit]) + math.log(prior[digit]))
  predicted_labels = argmax(q, axis = 0)
  return predicted_labels

train_size = [100, 200, 500, 1000, 2000, 5000, 10000, 30000]
error_rates_qda = []
predictions_qda = []
for size in train_size:
  tr_pts = train_pts[:size]
  tr_labels = train_lbs[:size]
  prior, mu_for_each_digit, cov_for_each_digit = qda_fit(tr_pts, tr_labels)
  predicted_labels_qda = qda_predict(val_pts, prior, mu_for_each_digit, 
                                     cov_for_each_digit)
  error_rates_qda.append(error_rate(predicted_labels_qda, val_lbs))
  predictions_qda.append(predicted_labels_qda)

plt.figure()
plt.plot(train_size, error_rates_qda)
plt.title("MNIST Validation Sets Error Rate vs Training Size Using QDA")
plt.xlabel("Training Size")
plt.ylabel("Error Rate")

#extra plot to help compare lda with qda on mnist
plt.figure()
plt.plot(train_size, error_rates_qda, label = "QDA")
plt.plot(train_size, error_rates_lda, label = "LDA")
plt.legend()
plt.title("MNIST Validation Sets Error Rate vs Training Size Comparing LDA and QDA")
plt.xlabel("Training Size")
plt.ylabel("Error Rate")

# part 3 d)
digit_dict = {}
for i in range(10):
  digit_dict[i] = []
for i in range(len(train_size)):
  for digit in range(10):
    digit_pred = predictions_lda[i][val_lbs == digit]
    digit_error_rt = 1 - (np.sum(digit_pred == digit) / len(digit_pred))
    digit_dict[digit].append(digit_error_rt)

plt.figure()
for digit in range(10):
  plt.plot(train_size, digit_dict[digit], label = f'Digit {digit}')
plt.legend()
plt.title("LDA classification, digitwise")
plt.xlabel("# of training points")
plt.ylabel("Error rate")

digit_dict_qda = {}
for i in range(10):
  digit_dict_qda[i] = []
for i in range(len(train_size)):
  for digit in range(10):
    digit_pred = predictions_qda[i][val_lbs == digit]
    digit_error_rt = 1 - (np.sum(digit_pred == digit) / len(digit_pred))
    digit_dict_qda[digit].append(digit_error_rt)

plt.figure()
for digit in range(10):
  plt.plot(train_size, digit_dict_qda[digit], label = f'Digit {digit}')
plt.legend(loc = 'upper right')
plt.title("QDA classification, digitwise")
plt.xlabel("# of training points")
plt.ylabel("Error rate")

mnist_test = mnist["test_data"]
mnist_test = mnist_test.reshape((-1, 28 * 28))

# part 4 best classifier for MNIST
# train on all training points
prior, mu_for_each_digit, cov_for_each_digit = lda_fit(train_pts, train_lbs)
predicted_labels = lda_predict(mnist_test, prior, mu_for_each_digit, 
                               cov_for_each_digit)
predicted_labels

import pandas as pd
def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1 # Ensures that the index starts at 1
    df.to_csv('submission.csv', index_label='Id')

results_to_csv(predicted_labels)

spam = np.load(f"../data/spam-data-hw3.npz")

spam_train_features, spam_train_labels, spam_val_features, spam_val_labels = shuffle_and_partition(
    spam["training_data"], spam["training_labels"], 0.20)

def lda_fit_two_class(data, labels):
  #need to compute mu_c and covariance
  number_of_each_digit = {}
  mu_for_each_digit = {}
  digits_present = np.unique(labels)
  d = data.shape[1]
  n = data.shape[0]
  pooled_cov = np.zeros((d, d))
  for digit in digits_present:
    data_for_this_digit = data[labels == digit] #get the dataset for each digit
    number_of_each_digit[digit] = data_for_this_digit.shape[0]
    mu_c = np.mean(data_for_this_digit, axis=0)
    mu_for_each_digit[digit] = mu_c
    cov = np.dot((data_for_this_digit - mu_c).T, data_for_this_digit - mu_c)
    pooled_cov = np.add(pooled_cov, cov)
  pooled_cov = pooled_cov / n
  #now to compute prior probabilities
  prior = {}
  for digit in digits_present:
    prior[digit] = number_of_each_digit[digit] / n
  return prior, mu_for_each_digit, pooled_cov

def lda_predict(test_data, prior, mu_for_each_digit, pooled_cov):
  q = []
  for digit in range(2):
    alpha = np.linalg.eig(pooled_cov)[0].min()
    d = test_data.shape[1]
    matrix = np.identity(d) * (-1) * alpha
    pooled_cov = np.add(pooled_cov, matrix)
    q.append(scipy.stats.multivariate_normal.logpdf(
        test_data, allow_singular=True, cov=pooled_cov, 
        mean=mu_for_each_digit[digit]) + math.log(prior[digit]))
  predicted_labels = argmax(q, axis = 0)
  return predicted_labels

def qda_predict_two_class(test_data, prior, mu_for_each_digit, cov_for_each_digit):
  q = []
  for digit in range(2):
    cov = cov_for_each_digit[digit]
    q.append(scipy.stats.multivariate_normal.logpdf(
        test_data, allow_singular=True, cov=cov, 
        mean=mu_for_each_digit[digit]) + math.log(prior[digit]))
  predicted_labels = argmax(q, axis = 0)
  return predicted_labels

#testing lda on spam
prior, mu_for_each_digit, cov_for_each_digit = lda_fit_two_class(spam_train_features, 
                                                                 spam_train_labels)
spam_pred_labels = lda_predict(spam_val_features, prior, mu_for_each_digit, 
                               cov_for_each_digit)
error_rate(spam_pred_labels, spam_val_labels)

#testing qda on spam
prior, mu_for_each_digit, cov_for_each_digit = qda_fit(spam_train_features, spam_train_labels)
spam_pred_labels = qda_predict_two_class(spam_val_features, prior, mu_for_each_digit, 
                                         cov_for_each_digit)
error_rate(spam_pred_labels, spam_val_labels)

#best model for spam classification
prior, mu_for_each_digit, cov_for_each_digit = lda_fit_two_class(spam["training_data"], 
                                                                 spam["training_labels"])
spam_pred_labels = lda_predict(spam["test_data"], prior, mu_for_each_digit, cov_for_each_digit)

#qda fit for spam, but doesn't work as well as lda
prior, mu_for_each_digit, cov_for_each_digit = qda_fit(spam["training_data"], 
                                                       spam["training_labels"])
spam_pred_labels_qda = qda_predict_two_class(spam["test_data"], prior, 
                                             mu_for_each_digit, cov_for_each_digit)

results_to_csv(spam_pred_labels_qda)



