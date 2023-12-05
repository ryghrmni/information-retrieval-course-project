import numpy as np

train_file_path = './files/new_train_file_with_line.txt'
val_file_path = './files/val_file.txt'
test_file_path = './files/new_test_file_with_line.txt'

relation_file_path = './files/relations.txt'
emb_google_txt = './GoogleNews/GoogleNews-vectors-negative300.txt'
average_vector_file = './GoogleNews/GoogleNews-vectors-negative300_average_vector.txt'

data_all_file_path = "./data/data_all"

files = [train_file_path, val_file_path, test_file_path]

label_to_int = {}
int_to_label= {} 

with open(relation_file_path, 'r') as f:
    for line in f: 
        line = line.strip().split()
        key = str(line[0])
        val = int(line[1])
        label_to_int[key] = val 
        int_to_label[val] = key 
        
print(int_to_label)

words_dataset = set()

def create_words_dataset():
    for f_name in files: 
        f = open(f_name, 'r')
        lines = f.readlines()
        f.close()
        
        lines = [ l.strip().split(" ")[2:]  for l in lines]
    
        for line in lines:
            for w in line:
                words_dataset.add(w)
    
    print("len(words_dataset)", len(words_dataset))
    
    with open('./files/words_dataset.txt', 'w') as f: 
        for w in sorted(words_dataset):
            f.write(str(w) + "\n")
        print('./files/words_dataset.txt created')
    
    
create_words_dataset()

word_to_emb = {}

with open(emb_google_txt, 'r', encoding='utf-8') as f: 
    first = True
    
    for line in f:
        if first == True:
            first = False
            continue
        line = line.strip().split()
        if len(line) != 301:
            continue 
        word = str(line[0])
        vec = [float(x) for x in line[1:]]
        vec = np.array(vec, dtype='float64')
        
        if word in words_dataset:
            word_to_emb[word] = vec
        elif word.lower() in words_dataset and word.lower() not in word_to_emb: 
            word_to_emb[word.lower()] = vec

def get_average_vector(file_name):
    with open(file_name, 'r') as f:
        line = f.readline()
        line = line.strip().split()
        line = [float(x) for x in line]
        average_vector = np.array(line, dtype='float64')
        print("average_vector.shape", average_vector.shape)
        return average_vector

average_vector = get_average_vector(average_vector_file)

word_to_int = {}
embedding = []

unknown_words = 0
word_to_int['PADDING'] = len(word_to_int)
embedding.append(np.zeros(300, dtype='float64'))

for w in sorted(words_dataset): 
    word_to_int[w] = len(word_to_int)
    if w in word_to_emb:
        embedding.append(word_to_emb[w])
    elif w.lower() in word_to_emb:
        embedding.append(word_to_emb[w.lower()])
    else:
        unknown_words += 1
        embedding.append(average_vector)

embedding = np.array(embedding, dtype='float64')
print("len(word_to_int)", len(word_to_int)) # 25656
print("embedding.shape", embedding.shape) # (25656, 300)
print("unknown_words", unknown_words) # 2652

def get_max_sent_len(files):
    max_sent_len = 0 
    for fname in files: 
        f = open(fname, 'r')
        lines = f.readlines()
        f.close()
        for l in lines: 
            l = l.strip().split(" ")[2:]
            max_sent_len = max(max_sent_len, len(l))
    return max_sent_len

max_sent_len = get_max_sent_len(files)
print("max_sent_len", max_sent_len) # 102-1 = 101

def create_matrices(file_name, word_to_int, label_to_int, max_sent_len):
    X = []
    Y = []
    
    f = open(file_name, 'r')
    lines = f.readlines()
    f.close()
    lines = [line.strip().split() for line in lines]
    
    for line in lines: 
        Y.append(label_to_int[line[1]])
        line = line[2:]
        tmp = np.zeros(max_sent_len, dtype='int32')
        for i in range(len(line)):
            tmp[i] = word_to_int[line[i]]
        X.append(tmp)

        
    X = np.array(X, dtype='int32')
    Y = np.array(Y, dtype='int32')
    
    return [X, Y]

train_set = create_matrices(train_file_path, word_to_int, label_to_int, max_sent_len)
val_set = create_matrices(val_file_path, word_to_int, label_to_int, max_sent_len)
test_set = create_matrices(test_file_path, word_to_int, label_to_int, max_sent_len)

data_all = [train_set, val_set, test_set, embedding, label_to_int, int_to_label]

np.save(data_all_file_path, data_all)
