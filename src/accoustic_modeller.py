import numpy as np
import glob
import tensorflow as tf 

from keras.optimizers import Adam
from pathlib import Path
from keras.utils import to_categorical 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential 
from keras.layers import Dense, Activation, Flatten, Conv2D, InputLayer, MaxPooling2D 


def create_model(): 
    numClasses=10 
    model=Sequential() 
    model.add(InputLayer(input_shape=(40, 21, 1))) 
    model.add(Conv2D(64, (3, 3), activation='relu')) 
    model.add(Conv2D(64, (3, 3), activation='relu')) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3))) 
    model.add(Flatten()) 
    model.add(Dense(256)) 
    model.add(Activation('relu')) 
    model.add(Dense(numClasses)) 
    model.add(Activation('softmax')) 
    return model


# Load and pad MFCC data
mfcc_files = sorted(glob.glob('week4/mfccs/*.npy'))
mfcc_data_list = [np.load(file) for file in mfcc_files]
max_frames = max(data.shape[1] for data in mfcc_data_list)

data = np.array([np.pad(data, ((0, 0), (0, max_frames - data.shape[1]))) for data in mfcc_data_list])

# Extract labels from filenames
labels = np.array([Path(file).stem.split('_')[0] for file in mfcc_files])

# Normalise data
data = data / np.max(data)

print("Data shape:", data.shape)
print("Labels shape:", labels.shape)
print("Max frames:", max_frames)

LE=LabelEncoder() 
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] 
LE = LE.fit(classes) 
labels=to_categorical(LE.transform(labels)) 


X_train, X_tmp, y_train, y_tmp = train_test_split(data, labels, test_size=0.2, random_state=0) 
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0) 

num_epochs = 25
num_batch_size = 32

model = create_model() 
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam' )
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=num_batch_size, epochs=num_epochs, verbose="auto") 

predicted_probs=model.predict(X_test,verbose=0) 
predicted=np.argmax(predicted_probs,axis=1) 
actual=np.argmax(y_test,axis=1) 
accuracy = metrics.accuracy_score(actual, predicted) 
print(f'Accuracy: {accuracy * 100}%')

predicted_prob = model.predict(np.expand_dims(X_test[0,:,:], axis = 0), verbose='auto') 
predicted_id = np.argmax(predicted_prob,axis=1) 
predicted_class = LE.inverse_transform(predicted_id) 

print(f'Predicted class for first test sample: {predicted_class[0]}')

confusion_matrix = metrics.confusion_matrix( 
np.argmax(y_test,axis=1), predicted) 
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = 
confusion_matrix) 
cm_display.plot()

print(metrics.classification_report(np.argmax(y_test,axis=1), predicted, target_names=classes))