import pandas as pd
from sklearn import metrics
import nltk
import numpy as np
import string

# obtenemos los datos de entrenamiento
train_dataset = pd.read_csv("text_classification_train.csv")
# obtenemos una lista de los labels a utilizar: 
labels = open("emotions.txt", "r").read().splitlines()

def balance_one_class(data, amount_of_samples):
   delta_amount_of_samples = amount_of_samples - data.shape[0]
   replace = delta_amount_of_samples>0
   delta_amount_of_samples = abs(delta_amount_of_samples)
   indexes = data.index
   sampled_indexes = np.random.choice(indexes, size=delta_amount_of_samples, replace=replace)

   return sampled_indexes.tolist()


def balance_dataset(dataset, amount_of_samples):
    dataset = dataset.copy()
    labels = [f"label_{i}" for i in range(28)]
    for a_label in labels:
        label_data = dataset[dataset[a_label]==1]
        over = amount_of_samples>label_data.shape[0]       
        resample_indexes = balance_one_class(label_data, amount_of_samples)
        if over:
            # we oversample the data:
            copied_data = dataset.iloc[resample_indexes]
            dataset = pd.concat([dataset, copied_data]).reset_index(drop=True)
        else: 
            dataset = dataset.drop(resample_indexes).reset_index(drop=True)
    return dataset


# matriz binaria que permita saber las categorizaciones multiples que pueda tener cada registro:
def binarize_labels(df):
    N_categories = 28
    N_registries = df.shape[0]

    binary_matrix = np.zeros((N_registries, N_categories))
    for i, a_emotion in enumerate(df["emotion"].tolist()):
        category_list = list(map(int, a_emotion.split(",")))
        binary_matrix[i, category_list] = 1
    B_df = pd.DataFrame(binary_matrix, columns=[f"label_{i}" for i in range(N_categories)])
    # append to dataset: 
    df = df.join(B_df)
    return df

train_dataset["amount emotions"] = train_dataset["emotion"].apply(lambda x: len(x.split(",")))
train_dataset["text_emotions"] = train_dataset["emotion"].apply(lambda x: ",".join([labels[int(i)] for i in x.split(",")]))
train_dataset = binarize_labels(train_dataset)
balanced_train_dataset = balance_dataset(train_dataset, 5000)

import matplotlib.pyplot as plt
(balanced_train_dataset["emotion"].value_counts()[:30]).plot(kind="bar", title="Histograma de etiquetas multiples y únicas")
plt.grid()
plt.show()