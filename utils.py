from sklearn.utils import shuffle
import pickle
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from model import CNN
import numpy as np
import copy
import os
import shutil



# cnt = 0
def read_ted(lang,category,options):
    if lang == "en":
        lang = "en-fr"
    if lang == "fr":
        lang = "fr-en"
    data = {}
    def read(mode):
        x, y = [], []
        porns = ["positive","negative"]
        posi = []
        nega = []
        for porn in porns:
            with open("data/ted-cldc/" + lang + "/" + mode + "/" + category + "/all/" + porn + ".new.txt", "r", encoding="utf-8") as f:
                for line in f:
                    if line[-1] == "\n":
                        line = line[:-1]
                    sentence = line.split()
                    if porn == "positive":
                        posi.append(sentence)
                    else:
                        nega.append(sentence)

        posi = shuffle(posi,random_state=options.random_state)
        nega = shuffle(nega,random_state=options.random_state)

        if mode == "train":
            dev_idx = 500
            data["dev_x"], data["dev_y"] = (posi[:dev_idx]+nega[:dev_idx]),(["positive"]*dev_idx + ["negative"]*dev_idx)
            data["dev_x"], data["dev_y"] = shuffle(data["dev_x"], data["dev_y"],random_state=options.random_state)
            data["train_positive"] = posi[dev_idx:]
            data["train_negative"] = nega[dev_idx:]

        else:
            y = ["positive"]*len(posi) + ["negative"]*len(nega)
            x = posi + nega
            x, y = shuffle(x, y,random_state=options.random_state)
            data["test_x"], data["test_y"] = x, y

            
    read("train")
    read("test")
        
    return data


## set and data
def data_setting(data,word_vectors,category,options):
    
    #sentence cut
    mode_arr = ["train_positive","train_negative","dev_x","test_x"]
    for mode in mode_arr:
        for i,sen in enumerate(data[mode]):
            if len(sen)>50:
                data[mode][i] = data[mode][i][:50]
                
                
    data["vocab"] = sorted(list(set([w for sent in data["train_positive"] + data["train_negative"] + data["dev_x"] + data["test_x"] for w in sent])))
    
    data["classes"] = sorted(list(set(data["dev_y"])))
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}
    
    params = {
    #add
    "MODEL_NAME": options.model_name,
    "CATEGORY": category,
    "RANDOM_STATE": options.random_state,
        
    "MODEL": options.model,
    "DATASET": options.dataset,
    "SAVE_MODEL": options.save_model,
    "EARLY_STOPPING": options.early_stopping,
    "EPOCH": options.epoch,
    "LEARNING_RATE": options.learning_rate,
    "MAX_SENT_LEN": max([len(sent) for sent in data["train_positive"] + data["train_negative"] + data["dev_x"] + data["test_x"]]),
    "BATCH_SIZE": 50,
    "WORD_DIM": options.word_dim,
    "VOCAB_SIZE": len(data["vocab"]),
    "CLASS_SIZE": len(data["classes"]),
    "FILTERS": [2, 3, 4, 5],
    "FILTER_NUM": [8, 8, 8, 8],
    "DROPOUT_PROB": 0.5,
    "NORM_LIMIT": 3,
    "GPU": options.gpu
}
    
    
    if params["MODEL"] != "rand":
        # load word2vec
#         word_vectors = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
#         word_vectors = KeyedVectors.load_word2vec_format("../UNMT-JL_test/10Mtest/train.1000k.en.map.vec", binary=False)

        #辞書にない単語は全てUNK扱いして同じベクトルで扱う
        np.random.seed(options.random_state)
        unkvec = np.random.uniform(-0.01, 0.01, options.word_dim).astype("float32")
   
        wv_matrix = []

        for i in range(len(data["vocab"])):
            word = data["idx_to_word"][i]
            if word in word_vectors.vocab:     
                wv_matrix.append(word_vectors.word_vec(word))
            else:
                wv_matrix.append(unkvec)
  

        # one for UNK and one for zero padding
        wv_matrix.append(unkvec)
        
        N = 50000 - len(wv_matrix)
        for i in range(N):
            wv_matrix.append(np.zeros(options.word_dim).astype("float32"))
            
        wv_matrix = np.array(wv_matrix)
        params["WV_MATRIX"] = wv_matrix
    
    return data,params


def train(data, params, data2,params2,options):

    model = CNN(**params).cuda(params["GPU"])

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()

    pre_dev_acc = 0
    pre_dev_fscore = 0
    max_dev_acc = 0
    max_test_acc = 0
    max_F1 = -1
    
    randam_state_arr = range(params["EPOCH"])
    randam_state_arr = shuffle(randam_state_arr,random_state = options.random_state)
    
    for e in range(params["EPOCH"]):
        
        train_loss= 0
        
        #epochごとにnegativeはデータをランダムで選択
        #ここのランダムステートを一定にする必要がある
        
        lennega = len(data["train_positive"])
        data["train_negative"] = shuffle(data["train_negative"],random_state = randam_state_arr[e])
        data["train_positive"] = shuffle(data["train_positive"],random_state = randam_state_arr[e])
        data["train_x"] = data["train_positive"] + data["train_negative"][0:lennega]
        data["train_y"] = ["positive"] * lennega + ["negative"] * lennega   
        
        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"],random_state = options.random_state)

        for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)

            batch_x = [[data["word_to_idx"][w] for w in sent] +
                       [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                       for sent in data["train_x"][i:i + batch_range]]
            batch_y = [data["classes"].index(c) for c in data["train_y"][i:i + batch_range]]
            batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
            batch_y = Variable(torch.LongTensor(batch_y)).cuda(params["GPU"])

            optimizer.zero_grad()
            model.train()
            pred = model(batch_x)
            
            loss = criterion(pred, batch_y)
            train_loss += loss
            loss.backward()
            nn.utils.clip_grad_norm(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()

        
        train_loss /= len(data["train_x"])/params["BATCH_SIZE"]

        # dev_acc = test(data, model, params, mode="dev")
        dev_fscore = get_fscores(data2, model, params2, mode="dev")

        F1,tp1,fp1,tn1,fn1 = get_fscores(data2, model, params2, mode="test")

        print("epoch:", e + 1, "/ dev_fscore:", round(dev_fscore[0],3),"test_fscores:",round(F1,3),"/ loss:", round(train_loss.item(),3))

        if params["EARLY_STOPPING"] and dev_acc <= pre_dev_acc:
            print("early stopping by dev_acc!")
            break
        else:
            # pre_dev_acc = dev_acc
            pre_dev_fscore = dev_fscore[0]
            
        if dev_fscore[0] > max_F1:
            max_F1 = dev_fscore[0]

            best_model = copy.deepcopy(model)

    print("max dev_fscores:", round(max_F1,3))
    return best_model


def test(data, model, params, mode="test"):
    model.eval()

    if mode == "dev":
        x, y = data["dev_x"], data["dev_y"]
    elif mode == "test":
        x, y = data["test_x"], data["test_y"]

    x = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] +
         [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
         for sent in x]

    x = Variable(torch.LongTensor(x)).cuda(params["GPU"])
    y = [data["classes"].index(c) for c in y]

    preds = np.argmax(model(x).cpu().data.numpy(), axis=1)

    acc = sum([1 if p == y else 0 for p, y in zip(preds, y)]) / len(preds)
    return acc




def caluculate_f(preds,y):
    tp,fp,tn,fn = 0,0,0,0
    for i,label in enumerate(y):
        if preds[i] == 1:
            if preds[i] == label:
                tp += 1
            else:
                fp += 1
        else:
            if preds[i] == label:
                tn += 1
            else:
                fn += 1
    if tp == 0:
        f = 0
    else:   
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f = 2*precision*recall / (precision + recall)  
    return f,tp,fp,tn,fn


def get_fscores(data, model, params, mode = "test"):
    model.eval()
    if mode == "test":
        x, y = data["test_x"], data["test_y"]
    if mode == "dev":
        x, y = data["dev_x"], data["dev_y"]

    x = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] +
         [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
         for sent in x]
    x = Variable(torch.LongTensor(x)).cuda(params["GPU"])
    y = [data["classes"].index(c) for c in y]
    preds = np.argmax(model(x).cpu().data.numpy(), axis=1)
    acc = sum([1 if p == y else 0 for p, y in zip(preds, y)]) / len(preds)
    if mode == "test":
        print("test_acc:",acc)
    if mode == "dev":
        print("dev_acc:",acc)
    F,tp,fp,tn,fn = caluculate_f(preds,y)
    return F,tp,fp,tn,fn
    
    


def save_model(model, params):
    # path = f"saved_models/{params['DATASET']}_{params['MODEL']}_{params['EPOCH']}.pkl"
    new_dir_path = f"saved_models/{params['MODEL_NAME']}"
    if not(os.path.exists(new_dir_path)):
        os.mkdir(new_dir_path)
    path = f"saved_models/{params['MODEL_NAME']}/{params['CATEGORY']}_{params['EPOCH']}_s{params['RANDOM_STATE']}.pkl"
    pickle.dump(model, open(path, "wb"))
    print(f"A model is saved successfully as {path}!")


def load_model(params):
    # path = f"saved_models/{params['DATASET']}_{params['MODEL']}_{params['EPOCH']}.pkl"
    path = f"saved_models/{params['MODEL_NAME']}/{params['CATEGORY']}_{params['EPOCH']}_s{params['RANDOM_STATE']}.pkl"

    try:
        model = pickle.load(open(path, "rb"))
        print(f"Model in {path} loaded successfully!")

        return model
    except:
        print(f"No available model such as {path}.")
        exit()
