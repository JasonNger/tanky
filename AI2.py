import pickle,pandas as pd ,numpy as np,matplotlib.pyplot as plt,seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.models import Model
from keras.layers import LSTM,Activation,Dense,Dropout,Input,Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.models import load_model
import time

start = time.clock()

if __name__ == '__main__':
    train_df = pd.read_csv("train.csv", encoding='gb18030')
    val_df = pd.read_csv("val.csv", encoding='gb18030')
    test_df = pd.read_csv("test.csv", encoding='gb18030')
    print(train_df.head())

    # plt.rcParams['font.sans-serif'] = ['KaiTi']
    # plt.rcParams['font.sans-serif'] = ['Arial Unicode']
    # plt.rcParams['axes.unicode_minus'] = False

    train_y = train_df.label
    print("Label:")
    print(train_y[:10])

    val_y = val_df.label
    test_y = test_df.label
    le = LabelEncoder()
    train_y = le.fit_transform(train_y).reshape(-1,1)
    print("LabelEncoder")
    print(train_y[:10])
    print(len(train_y))

    val_y = le.transform(val_y).reshape(-1,1)
    test_y = le.transform(test_y).reshape(-1,1)

    ## 对数据集的表情数据进行 one-hot 编码
    ohe = OneHotEncoder()
    train_y = ohe.fit_transform(train_y).toarray()
    val_y = ohe.fit_transform(val_y).toarray()
    test_y = ohe.fit_transform(test_y).toarray()
    print("OneHotEncoder:")
    print(train_y[:10])

    max_words = 5000
    max_len = 600
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(train_df.fenci)
    print(tok)

    with open("tok.pickle", 'wb') as handle:
        pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("tok.pickle", 'rb') as handle:
        tok = pickle.load(handle)

    for ii,iterm in enumerate(tok.word_index.items()):
        if ii<10:
            print(iterm)
        else:
            break
    print("==========================")
    for ii,iterm in enumerate(tok.word_counts.items()):
        if ii<10:
            print(iterm)
        else:
            break

    train_seq = tok.texts_to_sequences(train_df.fenci)
    val_seq = tok.texts_to_sequences(val_df.fenci)
    test_seq = tok.texts_to_sequences(test_df.fenci)

    train_seq_mat = sequence.pad_sequences(train_seq,maxlen=max_len)
    val_seq_mat = sequence.pad_sequences(val_seq,maxlen=max_len)
    test_seq_mat = sequence.pad_sequences(test_seq,maxlen=max_len)

    print(train_seq_mat.shape)
    print(val_seq_mat.shape)
    print(test_seq_mat.shape)
    print(train_seq_mat[:2])

## 建立LSTM模型

    inputs = Input(name='inputs',shape=[max_len])

    layper = Embedding(max_words+1,256,input_length=max_len)(inputs)
    layper = LSTM(128, return_sequences=True)(layper)
    layper = LSTM(128)(layper)
    layper = Dense(128,activation="relu",name="FC1")(layper)
    layper = Dense(64, activation="relu", name="FC2")(layper)
    layper = Dropout(0.3)(layper)
    layper = Dense(32, activation="relu", name="FC3")(layper)
    layper = Dropout(0.3)(layper)
    layper = Dense(2,activation="softmax",name="FC4")(layper)
    model = Model(inputs=inputs,outputs=layper)
    model.summary()
    model.compile(loss="categorical_crossentropy",
                  optimizer=RMSprop(),     #可以改Adam函数
                  metrics=["accuracy"])
    flag = "train"
    model_filename_path = "my_model.h5"

    if flag=="train":
        print("[+] 模型训练")
        model_fit = model.fit(train_seq_mat,train_y,batch_size=128,epochs=15,
                              use_multiprocessing=True,
                              validation_data=(val_seq_mat,val_y)
                              )
        model.save(model_filename_path)
        del model
        elaped = (time.clock() - start)
        print("Time used:",elaped)
    else:
        print("[+] 模型预测")
        model = load_model(model_filename_path)

    test_pre = model.predict(test_seq_mat)
    # confm = metrics.confusion_matrix(np.argmax(test_y,axis=1),np.argmax(test_pre,axis=1))
    # print(np.argmax(test_pre,axis=1).shape)
    a = np.argmax(test_pre,axis=1)
    np.save('a.npy', a)
    #
    # Labname = ["正常","异常"]
    #
    # print(metrics.classification_report(np.argmax(test_y,axis=1),np.argmax(test_pre,axis=1)))
    # # classification_pj(np.argmax(test_pre,axis=1),np.argmax(test_y,axis=1))
    # plt.figure(figsize=(8,8))
    # sns.heatmap(confm.T,square=True,annot=True,
    #             fmt='d',cbar=False,linewidths=.6,
    #             cmap="YlGnBu"
    #             )
    # plt.xlabel("True label",size=14)
    # plt.ylabel("Predicted label",size=14)
    # plt.xticks(np.arange(2)+0.8,Labname,size = 12)
    # plt.yticks(np.arange(2)+0.4,Labname,size = 12)
    # plt.show()
    #
    # val_seq = tok.texts_to_sequences(val_df.fenci)
    # val_seq_mat = sequence.pad_sequences(val_seq,maxlen=max_len)
    #
    # val_pre = model.predict(val_seq_mat)
    # print(metrics.classification_report(np.argmax(val_y,axis=1),np.argmax(val_pre,axis=1)))
    # # classification_pj(np.argmax(val_pre,axis=1),np.argmax(val_y,axis=1))
    #
    # elaped = (time.clock() - start)
    # print("Time used:",elaped)
