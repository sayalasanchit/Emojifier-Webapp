from keras.models import model_from_json
import emoji
import pandas as pd
import numpy as np

if __name__=="__main__":
	glove_path="glove.6B.50d.txt"
	model_arch_path='model.json'
	model_wts_path="model.h5"
else:
	glove_path="services/emojifier/glove.6B.50d.txt"
	model_arch_path='services/emojifier/model.json'
	model_wts_path="services/emojifier/model.h5"

# To avoid error when running the LSTM model with tensorflow-gpu
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

emoji_dictionary={0: "\u2764\uFE0F", # Red heart in utf-8 as the heart in emoji changes colour
                 1: ":baseball:",
                 2: ":grinning_face_with_big_eyes:",
                 3: ":disappointed_face:",
                 4: ":fork_and_knife:",
                 5: ":hundred_points:",
                 6: ":fire:",
                 7: ":face_blowing_a_kiss:",
                 8: ":chestnut:",
                 9: ":flexed_biceps:"}

with open(model_arch_path) as f:
	model=model_from_json(f.read())
model.load_weights(model_wts_path)

# Making an embedding dictionary
f=open(glove_path, encoding='utf-8')
embedding_index={}
for line in f:
    values=line.split()
    word=values[0]
    word_vector=np.asarray(values[1:], dtype='float')
    embedding_index[word]=word_vector
f.close()

emb_dim=50

# Converting the words in train and test data into embedding vectors
def embedding_output(X, max_len=10):
    emb_out=np.zeros((X.shape[0], max_len, emb_dim))
    for i in range(X.shape[0]): # Iterating over each sentence
        xi=X[i].split()
        for j in range(len(xi)):
            try:
                emb_out[i][j]=embedding_index[xi[j].lower()]
            except:
                emb_out[i][j]=np.zeros((50,))
    return emb_out

def predict(X):
    X=pd.Series([X])
    emb_X=embedding_output(X)
    p=model.predict_classes(emb_X)
    return emoji.emojize(emoji_dictionary[p[0]])

if __name__ == '__main__':
	text=input("Enter text:")
	print(text, predict(text))