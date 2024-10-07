import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from scFTAT import Transformer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation, SpatialDropout1D, Convolution1D, GlobalMaxPooling1D
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

feature = []  
expression = pd.read_csv('finaldata/mouse_kidney/train_data.csv')
file = open('finaldata/mouse_kidney/train_data.csv')
lines = file.readlines() 
line_0 = lines[0].strip('\n').split(',') 

for i in range(1,len(line_0)):
    tem = list(expression[line_0[i]])    
    feature.append(list(tem))
    file.close()

feature_train = list(feature)
label = []
file = open('finaldata/mouse_kidney/train_labels.csv')
lable_lines = file.readlines()
lable_line_0 = lable_lines[0].strip('\n').split(',')
file.close()


for i in range(1,len(lable_line_0)):
    label.append(int(lable_line_0[i]))


y_train=[]
for i in label:
    tem =[]
    for j in range(0,17):
        tem.append(0)
    tem[i-1]=1
    y_train.append(tem)

feature = []  
testexpression = pd.read_csv('finaldata/mouse_kidney/test_data.csv')
file = open('finaldata/mouse_kidney/test_data.csv') 
lines = file.readlines() 
line_0 = lines[0].strip('\n').split(',') 
for i in range(1,len(line_0)):
    tem = list(testexpression[line_0[i]])
    feature.append(tem)
file.close()
feature_test = list(feature)

activation = 'relu'
dropout = 0.2
epoch = 5
params_dict = {'kernel_initializer': 'glorot_uniform','kernel_regularizer': l2(0.01),}
num_layers = 4 
model_size = 40
num_heads = 5
dff_size = 128
maxlen = 16
vocab_size = 121 

enc_inputs = tf.keras.layers.Input(shape=(maxlen,))
transformer = Transformer(num_layers=num_layers, model_size=model_size, num_heads=num_heads, dff_size=dff_size,
                          vocab_size=vocab_size+1, maxlen=maxlen)
final_output = transformer(enc_inputs,training=True)

final_output = SpatialDropout1D(0.2)(final_output)
final_output = Convolution1D(filters=64,kernel_size=15, padding='same', kernel_initializer='glorot_normal',
                             kernel_regularizer=l2(0.001))(final_output)
final_output = Activation('relu')(final_output)
final_output = GlobalMaxPooling1D()(final_output)
final_output = Dense(17,'softmax',**params_dict)(final_output)

class PrintEpochMetrics(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch: {epoch + 1}, Loss: {logs.get('loss'):.4f}, Accuracy: {logs.get('accuracy'):.4f}")


model = Model(inputs=enc_inputs,outputs=final_output)
model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

# feature_train = [list(i) for i in feature_train]
# feature_test = [list(i) for i in feature_test]

feature_train = np.array(feature_train)
feature_test = np.array(feature_test)
y_train = np.array(y_train)

# for i in range(epoch):
#     print(i)
#     model.fit(feature_train,y_train,verbose=1,epochs=i+1,initial_epoch=i,batch_size=32,shuffle=True)

model.fit(feature_train, y_train, verbose=0, epochs=epoch, batch_size=32, shuffle=True, callbacks=[PrintEpochMetrics()])

a = model.predict(x=feature_test,batch_size=32)
print(a)
with open('modelsave/mkidney_epoch200.txt','w',newline='') as f:
    for i in range(len(a)):
        f.write(str(i))
        f.write(',')
        for j in range(len(a[i])):
            f.write(str(a[i][j]))
            f.write(',')
        f.write('\n')