
"""
@author: Anonymous
"""

# Generic
import os 
os.environ["CUDA_VISIBLE_DEVICES"]= "1" # use the gpu number 1 
device ="cuda"
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings, json
warnings.filterwarnings('ignore')
# TensorFlow
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
# Transformer Model
from transformers import BertTokenizer, TFAutoModel
# SKLearn Library
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import matthews_corrcoef
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, average_precision_score, confusion_matrix


# build the complete model with the classification head
def build_model(transformer):
    
        transformer_encoder = TFAutoModel.from_pretrained(transformer)  #Pretrained BERT Transformer Model
        input_layer = Input(shape=(max_len,), dtype=tf.int32, name="input_layer")
        sequence_output = transformer_encoder(input_layer)[0]
        cls_token = sequence_output[:, 0, :]
        output_layer = Dense(1, activation='sigmoid')(cls_token)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            Adam(lr=1e-5), 
            loss='binary_crossentropy', 
            metrics=[tf.keras.metrics.Recall()] 
        )
        return model
    
 
# select 50% of entailement and 50% neutral from german snli train set, test set, val set
def select_examples(csvfile, suffix):
    samples = pd.read_csv(csvfile)
    samples_entailment = samples[samples.gold_label=="entailment"].reset_index(drop=True)
    samples_neutral = samples[samples.gold_label=="neutral"].reset_index(drop=True)
    samples_contradiction = samples[samples.gold_label=="contradiction"].reset_index(drop=True)
    entailment1, entailment2 = train_test_split(samples_entailment, test_size=0.5)
    neutral1, neutral2 = train_test_split(samples_neutral, test_size=0.5)
    entail_neutr =shuffle(entailment1.append(neutral1).reset_index(drop=True))
    samples_final = shuffle(samples_contradiction.append(entail_neutr).reset_index(drop=True))
    samples_final.to_csv("snli_1.0_"+suffix+".csv",  index = False)
   
   
# read snli data    
def read_snli_data(train_file, test_file, val_file):
    select_examples(train_file, "train_1")
    select_examples(test_file, "test_1")
    select_examples(val_file, "val_1")
    train_data = pd.read_csv("snli_1.0_train_1.csv")
    test_data = pd.read_csv("snli_1.0_test_1.csv")
    val_data = pd.read_csv("snli_1.0_val_1.csv")
    # train_data = pd.read_csv(train_file)
    # test_data = pd.read_csv(test_file)
    # val_data = pd.read_csv(val_file)
    train_data.loc[train_data['gold_label'] == "contradiction", 'gold_label'] = 1
    train_data.loc[train_data['gold_label'] == "entailment", 'gold_label'] = 0
    train_data.loc[train_data['gold_label'] == "neutral", 'gold_label'] = 0
    train = train_data[train_data.gold_label != "-"]
    train = train.loc[train['sentence1'].apply(lambda x: isinstance(x, str))] 
    train = train.loc[train['sentence2'].apply(lambda x: isinstance(x, str))] 
    test_data.loc[test_data['gold_label'] == "contradiction", 'gold_label'] = 1
    test_data.loc[test_data['gold_label'] == "entailment", 'gold_label'] = 0
    test_data.loc[test_data['gold_label'] == "neutral", 'gold_label'] = 0
    test = test_data[test_data.gold_label != "-"]
    test = test.loc[test['sentence1'].apply(lambda x: isinstance(x, str))]
    test = test.loc[test['sentence2'].apply(lambda x: isinstance(x, str))]
    val_data.loc[val_data['gold_label'] == "contradiction", 'gold_label'] = 1
    val_data.loc[val_data['gold_label'] == "entailment", 'gold_label'] = 0
    val_data.loc[val_data['gold_label'] == "neutral", 'gold_label'] = 0
    val = val_data[val_data.gold_label != "-"]
    val = val.loc[val['sentence1'].apply(lambda x: isinstance(x, str))]
    val = val.loc[val['sentence2'].apply(lambda x: isinstance(x, str))]
    # mix train and validation
    train_final = train.append(val, ignore_index=True)
    return train_final, test


if __name__ == "__main__":

            # Transformer Model Name
            #transformer_model = 'bert-base-multilingual-cased'
            #transformer_model = 'bert-base-multilingual-uncased'
            #transformer_model = 'distilbert-base-multilingual-cased'
            #transformer_model = "distilbert-base-german-cased"
            #transformer_model = "uklfr/gottbert-base"
            #transformer_model = "deepset/gelectra-base-germanquad"
            transformer_model = "deepset/gelectra-base"
            #transformer_model = "bert-base-german-cased"
            #transformer_model = "deepset/gbert-base"
            # Define Tokenizer
            tokenizer = BertTokenizer.from_pretrained(transformer_model)
            # Define Max Length
            max_len = 80   
            AUTO = tf.data.experimental.AUTOTUNE
            batch_size = 16
            epochs = 5 # worked well with snli
            # snli_data, train, test, val
            train_file = "snli_1.0_train.csv"
            test_file =  "snli_1.0_test.csv"
            val_file =   "snli_1.0_dev.csv"
            #dummy data to test test
            # train_file = "dummy.csv"
            # test_file =  "dummy.csv"
            # val_file =   "dummy.csv"
            train_data, test_data = read_snli_data(train_file, test_file, val_file)
            # Encode the training & test data 
            train = train_data[['sentence1','sentence2']].values.tolist()
            test = test_data[['sentence1','sentence2']].values.tolist()
            train_encode = tokenizer.batch_encode_plus(train, pad_to_max_length=True, max_length=max_len)
            test_encode = tokenizer.batch_encode_plus(test, pad_to_max_length=True, max_length=max_len)
            # Split the Training Data into Training (90%) & Validation (10%)
            test_size = 0.1  
            x_train, x_val, y_train, y_val = train_test_split(train_encode['input_ids'], train_data.gold_label.values, test_size=test_size)
            x_test = test_encode['input_ids']
            y_train = y_train.tolist()
            y_val = y_val.tolist()
            # x_test = np.asarray(x_test)
            train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train)).repeat().shuffle(3072).batch(batch_size).prefetch(AUTO))
            val_ds = (tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).prefetch(AUTO))
            test_ds = (tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size))
            #  compute weights of the two classes
            train_classes = train_data[["gold_label"]].to_numpy()[:,0]
            class_weights = compute_class_weight( class_weight = "balanced", classes = np.unique(train_classes), y = train_classes)
            class_weights_final = dict(zip(np.unique(train_classes), class_weights))
            # Applying the build model function
            model = build_model(transformer_model)
            # train model
            n_steps = len(train_data) // batch_size 
            # create a folder and call it saved_model in the home dir
            checkpoint_path = "saved_model/cp.ckpt"
            checkpoint_dir = os.path.dirname(checkpoint_path)
            # Create a callback that saves the model's weights
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
            model.fit(train_ds, 
                      steps_per_epoch = n_steps, 
                      class_weight = class_weights_final,
                      validation_data = val_ds,
                      epochs = epochs,
                      callbacks=[cp_callback])
            
            # predictions
            prediction = model.predict(test_ds, verbose=0)
            prediction_label = prediction>0.5
            target = test_data["gold_label"].tolist()
            # compute metrics
            f1_sco = f1_score(target, prediction_label) # average = 'macro' 'weighted'
            f1_sco_weighted = f1_score(target, prediction_label, average='weighted') # average = 'macro' 'weighted'
            precision = precision_score(target, prediction_label)
            recall= recall_score(target, prediction_label)
            print("F1:                 {:.2f}".format(f1_sco * 100))
            print("F1_weighetd:                 {:.2f}".format(f1_sco_weighted * 100))
            print("Precision:          {:.2f}".format(precision * 100))
            print("Recall:             {:.2f}".format(recall * 100))
            acc = accuracy_score(target, prediction_label)
            print("Accuracy:           {:.2f}".format(acc * 100))
            avgp = average_precision_score(target,prediction_label)
            print("Average precision:           {:.2f}".format(avgp * 100))
            conf_mat = confusion_matrix(target, prediction_label, labels=[1, 0])
            print("confMatrix")
            print(conf_mat)
            diction = {}
            diction["f1"] = round(f1_sco*100, 2)
            diction["precision"] = round(precision*100,2)
            diction["recall"] = round(recall*100,2)
            diction["avgp"] = round(avgp*100,2) 
            diction["accuracy"] = round(acc*100,2)
            diction["conf_mat"] = conf_mat.tolist()
            #Compute the Matthews correlation coefficient (MCC)
            y_true = target
            print(matthews_corrcoef(y_true, prediction_label))
            diction["mcc"] = matthews_corrcoef(y_true, prediction_label)
            with open('Metricsdata.json', 'w') as fp:
                json.dump(diction, fp)
            
            
