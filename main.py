import utils
from gensim.models.keyedvectors import KeyedVectors
import argparse

def main():

    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("wordvec_path1", help="wordvecor path1")
    parser.add_argument("wordvec_path2", help="wordvecor path2")
    parser.add_argument("--mode", default="train", help="train: train (with test) a model / test: test saved models")
    parser.add_argument("--model", default="static", help="available models: rand, static, non-static, multichannel")
    parser.add_argument("--dataset", default="ted")
    parser.add_argument("--save_model", default=True, action='store_true', help="whether saving model or not")
    parser.add_argument("--early_stopping", default=False, action='store_true', help="whether to apply early stopping")
    parser.add_argument("--epoch", default=30, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="learning rate")
    parser.add_argument("--gpu", default=1, type=int, help="the number of gpu to be used")
    parser.add_argument("--model_name", default="test",help="model's name")
    parser.add_argument("--word_dim", default=512,type=int,help="word dim")
    parser.add_argument("--random_state", default=1,type=int,help="random state")
    

    options = parser.parse_args()

    
    F = 0
    tp,fp,tn,fn = 0,0,0,0
    print("loading word2vec1...")
    word_vectors1 = KeyedVectors.load_word2vec_format(options.wordvec_path1, binary=False)
    print("loading word2vec2...")
    word_vectors2 = KeyedVectors.load_word2vec_format(options.wordvec_path2, binary=False)
    categories = ["art","arts","biology","business","creativity","culture","design","economics","education","entertainment","global","health","politics","science","technology"]
    for category in categories:
        en = utils.read_ted("en",category,options)
        fr = utils.read_ted("fr",category,options)

        data,params = utils.data_setting(en,word_vectors1,category,options)
        data2,params2 = utils.data_setting(fr,word_vectors2,category,options)
        if options.mode == "train":

            print(category)
            print(params["MODEL"])
            print("=" * 20 + "TRAINING STARTED" + "=" * 20)
            model = utils.train(data, params, data2, params2,options)
            if params["SAVE_MODEL"]:
                utils.save_model(model, params)
            print("=" * 20 + "TRAINING FINISHED" + "=" * 20)

        else:
            model = utils.load_model(params)




        F1,tp1,fp1,tn1,fn1 = utils.get_fscores(data2, model, params2)
        print("F_scores:",F1,"TP:",tp1,"FP:",fp1,"TN:",tn1,"FN:",fn1)
        print()

        tp += tp1
        fp += fp1
        tn += tn1
        fn += fn1
        F += F1

    all_precision =  tp / (tp + fp)
    all_recall =  tp / (tp + fn)
    f = 2*all_precision*all_recall / (all_precision + all_recall) 
    F = F / len(categories)

    print("micro Fscore:",round(f,3))
    print("macro Fscore:",round(F,3))




def main_jupyter(word_vectors1,word_vectors2,options):
    F = 0
    tp,fp,tn,fn = 0,0,0,0
    categories = ["art","arts","biology","business","creativity","culture","design","economics","education","entertainment","global","health","politics","science","technology"]
    for category in categories:
        en = utils.read_ted("en",category,options)
        fr = utils.read_ted("fr",category,options)

        data,params = utils.data_setting(en,word_vectors1,category,options)
        data2,params2 = utils.data_setting(fr,word_vectors2,category,options)
        if options.mode == "train":

            print(category)
            print(params["MODEL"])
            print("=" * 20 + "TRAINING STARTED" + "=" * 20)
            model = utils.train(data, params, data2, params2,options)
            if params["SAVE_MODEL"]:
                utils.save_model(model, params)
            print("=" * 20 + "TRAINING FINISHED" + "=" * 20)

        else:
            model = utils.load_model(params)




        F1,tp1,fp1,tn1,fn1 = utils.get_fscores(data2, model, params2)
        print("F_scores:",F1,"TP:",tp1,"FP:",fp1,"TN:",tn1,"FN:",fn1)
        print()

        tp += tp1
        fp += fp1
        tn += tn1
        fn += fn1
        F += F1

    all_precision =  tp / (tp + fp)
    all_recall =  tp / (tp + fn)
    f = 2*all_precision*all_recall / (all_precision + all_recall) 
    F = F / len(categories)

    print("micro Fscore:",round(f,3))
    print("macro Fscore:",round(F,3))


if __name__ == "__main__":
    main()

