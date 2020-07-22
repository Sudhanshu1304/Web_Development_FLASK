import tensorflow as tf
import matplotlib.pyplot as plt
import io
import seaborn as sns
sns.set()
import base64
from keras.preprocessing.sequence import  pad_sequences
import pandas as pd
import pickle
le=32

with open('tokenizer1.pickle', 'rb') as handle:
    token22 = pickle.load(handle)

nm=tf.keras.models.load_model('New_NLP_MODEL2 (1).h5')

his={}
c=[1]

def history_plot():

    img = io.BytesIO()
    y = list(his.values())
    a = list(his.keys())
    plt.xticks(c, a)
    c.append(c[-1] + 1)

    if y[len(his) - 1] > 0.01 and y[len(his) - 1] < 0.016:
        y[len(his) - 1] = 0.1 + int(str(y[len(his) - 1])[-1]) / 100

    elif y[len(his) - 1] > 0.016 and y[len(his) - 1] < 0.02:
        y[len(his) - 1] = 0.2 + int(str(y[len(his) - 1])[-1]) / 100

    elif y[len(his) - 1] > 0.02:
        y[len(his) - 1] = 0.3 + int(str(y[len(his) - 1])[-1]) / 100

    elif y[len(his) - 1] > 0.9 and y[len(his) - 1] < 0.9901:
        n = str(y[len(his) - 1])[-1]
        n = 0.8 + (int(n) / 100)
        y[len(his) - 1] = n

    plt.bar(len(his), y[len(his) - 1])
    font = {'family': 'italic',
            'color': 'red',
            'size': 17,
            }

    plt.ylabel('Positivity Level', fontdict=font)
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url


def output1(l,model= nm):
    l = [l]
    T = pd.DataFrame(l)
    T = T.iloc[:, 0]
    x11 = token22.texts_to_sequences(T)
    x1 = pad_sequences(x11, maxlen=32, padding='post')
    aa = model.predict(x1)
    his[l[0]] = aa[0][0]
    history_plot()
    return aa[0][0]

