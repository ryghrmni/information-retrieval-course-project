# Importing necessary libraries
import numpy as np
import os
import subprocess
from keras.layers import LSTM
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Attention

# Define file paths and settings
data_npy_path = './data/data_all.npy'
scorer_perl_script = "./dataset/SemEval2010_task8_scorer-v1.2/semeval2010_task8_scorer-v1.2.pl"
format_checker_script = "./dataset/SemEval2010_task8_scorer-v1.2/semeval2010_task8_format_checker.pl"
train_answer_key = "./files/train_answer_keys.txt"
val_answer_key = "./files/val_answer_keys.txt"
test_answer_key = "./files/test_answer_keys.txt"
perl_exe_path = "C:/Users/Lenovo/AppData/Local/ActiveState/cache/bin/perl.exe"
proposed_ans = "./Model/proposed_ans2.txt"
scorer_output = "./Model/scorer_output2.txt"

# Load data and embeddings
train_set, val_set, test_set, embedding, label_to_int, int_to_label = np.load(data_npy_path, allow_pickle=True)
train_x, train_y = train_set
val_x, val_y = val_set
test_x, test_y = test_set
train_x_copy = train_x
train_y_copy = train_y
max_sentence_len = train_x.shape[1]
n = len(label_to_int)
number_of_epoch = 64
save_model = True

# Initialize variables for tracking max scores
train_max_accuracy = 0
train_max_f1 = 0
val_max_acc_accuracy = 0
val_max_f1 = 0
test_max_accuracy = 0
test_max_f1 = 0
test_f1_final = 0
test_f1_final_max = 0

# Print dataset and model details
print("train_x.shape", train_x.shape)
print("train_y.shape", train_y.shape)
print("val_x.shape", val_x.shape)
print("val_y.shape", val_y.shape)
print("test_x.shape", test_x.shape)
print("test_y.shape", test_y.shape)
print("embedding.shape", embedding.shape)
print("len(label_to_int)", len(label_to_int))
print("len(int_to_label)", len(int_to_label))
print("max_sentence_len", max_sentence_len)
print("n", n)
print()

# Define a function to build the LSTM
def build_lstm_model():
    input_layer = Input(shape=(max_sentence_len,))
    embedding_layer = Embedding(input_dim=len(embedding), output_dim=embedding.shape[1], input_length=max_sentence_len, weights=[embedding], trainable=False)(input_layer)
    lstm_layer = LSTM(128, return_sequences=True)(embedding_layer)
    attention = Attention()([lstm_layer, lstm_layer])
    pool_layer = GlobalMaxPooling1D()(attention)
    output_layer = Dense(n, activation='softmax')(pool_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


# Create the LSTM
lstm_model = build_lstm_model()

# Compile the model
lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
lstm_model.summary(line_length=120)

# Define function to calculate precision
def get_precision(test_y_pred, test_y, label): 
    label_count = 0
    label_count_correct = 0
    
    for i in range(len(test_y_pred)):
        if test_y_pred[i] == label: 
            label_count += 1 
            if test_y_pred[i] == test_y[i]: 
                label_count_correct += 1
    
    if label_count_correct == 0: 
        return 0.0
    else: 
        ret = float(label_count_correct) / float(label_count)
        return ret
        
# Define function to calculate macro-averaged F1 score
def get_macro_averaged_f1(test_y_pred, test_y, n):
    f1_sum = 0
    f1_count = 0
    for label in range(1, n):        
        prec = get_precision(test_y_pred, test_y, label)
        recall = get_precision(test_y, test_y_pred, label)
        f1 = 0 if float(prec+recall)==float(0) else float(2*prec*recall/float(prec+recall))
        f1_sum += f1
        f1_count += 1
    macro_f1 = float(f1_sum) / float(f1_count)    
    return macro_f1
    
# Define function to calculate accuracy
def get_accuracy(test_y_pred, test_y):
    acc =  float(np.sum(test_y_pred == test_y)) / float(len(test_y))
    return acc
    
# Define function to predict classes   
def predict_classes(prediction):
    return prediction.argmax(axis=-1)

# Define function to get all scores (P,R,F1) for SemEval
def get_all_score_semeval(y_pred, answer_key):
    
    f_out = open(proposed_ans, 'w')
    for i in range(len(y_pred)):
        f_out.write(str(i+1) + "\t" + int_to_label[y_pred[i]] + "\n" )
    f_out.close()
    
    os.system(perl_exe_path + " " + scorer_perl_script + " " + proposed_ans + " " + answer_key + " > " + scorer_output)
        
    f_in = open(scorer_output, 'r')
    lines = f_in.readlines()
    f_in.close()
    #print(lines)
    lines = [ l  for l in lines[-30:] if l.strip() != '']
    acc = float(lines[-17].strip().split()[-1][:-1]) / 100.0
    PRF1 = lines[-2].strip().split()
    P = float(PRF1[2][:-1]) / 100.0
    R = float(PRF1[5][:-1]) / 100.0
    macro_f1 = float(PRF1[8][:-1]) / 100.0
    
    return (acc, P, R, macro_f1)


# Training loop
print("Start training... \n")

step_epoch = 0 

for epoch in range(number_of_epoch):
    print("Epoch: ", epoch+1, "/", number_of_epoch)
    step_epoch += 1
    index = np.arange(len(train_x))
    np.random.shuffle(index)
    train_x = train_x[index]
    train_y = train_y[index]

    train_input_list = [train_x]
    train_input_list_copy = [train_x_copy]
    val_input_list = [val_x]
    test_input_list = [test_x]

    lstm_model.fit(train_input_list, train_y, verbose=1, epochs=1, batch_size=32)
    
    train_y_pred = predict_classes(lstm_model.predict(train_input_list_copy, verbose=1))
    val_y_pred = predict_classes(lstm_model.predict(val_input_list, verbose=1))
    test_y_pred = predict_classes(lstm_model.predict(test_input_list, verbose=1))

    train_all = get_all_score_semeval(train_y_pred, train_answer_key)
    val_all = get_all_score_semeval(val_y_pred, val_answer_key)
    test_all = get_all_score_semeval(test_y_pred, test_answer_key)

    train_max_f1 = max(train_max_f1, train_all[3])
    val_max_f1 = max(val_max_f1, val_all[3])
    test_max_f1 = max(test_max_f1, test_all[3])

    train_max_accuracy = max(train_max_accuracy, train_all[0])
    val_max_acc_accuracy = max(val_max_acc_accuracy, val_all[0])
    test_max_accuracy = max(test_max_accuracy, test_all[0])

    if val_max_f1 == val_all[3]: 
        test_f1_final = test_all[3]
        test_f1_final_max = max(test_f1_final, test_f1_final_max)
        if save_model : 
            lstm_model.save('./Model/lstm_relation_extraction_model.keras')
            print("Model saved", "lstm_relation_extraction_model.keras")
        step_epoch = 0
    
    print("Train Accuracy: %.4f (max: %.4f)" % (train_all[0], train_max_accuracy))
    print("Val   Accuracy: %.4f (max: %.4f)" % (val_all[0], val_max_acc_accuracy))
    print("Test  Accuracy: %.4f (max: %.4f)" % (test_all[0], test_max_accuracy))
    
    print("Train Macro F1 Semeval Official: %.4f (max: %.4f)" % (train_all[3], train_max_f1))
    print("Val   Macro F1 Semeval Official: %.4f (max: %.4f)" % (val_all[3], val_max_f1))
    print("Test  Macro F1 Semeval Official: %.4f (max: %.4f)" % (test_all[3], test_max_f1))
    print("Test P %.4f | R %.4f | macro_F1: %.4f" % test_all[1:] )
    print("Test test_max_f1_final: %.4f (max: %.4f)" % (test_f1_final, test_f1_final_max) )
        
    print("**************************************")
print("Done !")
