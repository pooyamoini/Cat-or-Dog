# Cat and Dog classification
##### In this notebook I will use CNN for this
classification model and I will use two datasets from Kaggle.

So lets download the datasets

```python
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d biaiscience/dogs-vs-cats
```

We have to unzip them !

```python
import zipfile
with zipfile.ZipFile("dogs-vs-cats.zip", 'r') as zip_ref:
    zip_ref.extractall(".")
```

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
print(tf)
```

So lets download the other dataset

```python
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c dogs-vs-cats
```

```python
!mv train.zip train1.zip
!mkdir more_data
!mv train1.zip more_data
```

```python
import zipfile
with zipfile.ZipFile("./more_data/train1.zip", 'r') as zip_ref:
    zip_ref.extractall("./more_data")
```

# Prepare Datasets

We have two dataset. So we should merge them and create an appropriate
dataframe.

```python
import os

training_dataset = []
labels = []

for x in os.listdir("./train/train/"):
    if x.startswith("dog"):
        labels.append("dog")
    else:    
        labels.append("cat")
    training_dataset.append("./train/train/" + x)

for x in os.listdir("./more_data/train/"):
    if x.startswith("dog"):
        labels.append("dog")
    else:    
        labels.append("cat")
    training_dataset.append("./more_data/train/" + x)

training_df = pd.DataFrame({
    'data': training_dataset,
    'labels': labels
})

train_df, validate_df = train_test_split(training_df, test_size=0.10, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

training_df.head()
```

We will use ImageDataGenerator from Tensorflow. The main reason is images have
not same size and we should force them to have same size. This module will do
this for us.

```python
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='data',
    y_col='labels',
    target_size=(300, 300),
    batch_size=128,
    class_mode="binary"
)

validate_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    x_col='data',
    y_col='labels',
    target_size=(300, 300),
    batch_size=128,
    class_mode="binary"
)
```

# Check dataset

Now lets see some images randomly with their labels.

```python
plt.figure(figsize=(18, 10))
for i in range(15):
    plt.subplot(3, 5, i + 1)
    rand = random.randrange(len(train_generator))
    second_rand = random.randrange(128)
    img = train_generator[rand][0][second_rand]
    label = train_generator[rand][1][second_rand]
    plt.title("cat" if int(label) == 0 else "dog")
    plt.imshow(img)
```

# Build the Model

And now this is time to create the CNN model!
We will use layers from keras.

```python
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['mae', 'acc'])

model.summary()
```

```python
from tensorflow.python.keras.callbacks import TerminateOnNaN, ProgbarLogger, History, EarlyStopping, ReduceLROnPlateau

earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1,factor=0.2, in_lr=0.00001)
prob_logger = ProgbarLogger(count_mode='steps')
callbacks = [earlystop, learning_rate_reduction, TerminateOnNaN(), History()]
```

So just lets fit the model!

```python
history = model.fit_generator(
    train_generator, 
    epochs=20,
    validation_data=validate_generator,
    validation_steps=len(validate_df) // 128,
    steps_per_epoch=len(train_df) // 128,
    callbacks=callbacks
)
```

```python
plt.figure(figsize=(12, 12))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.figure(figsize=(12, 12))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.figure(figsize=(12, 12))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```

# Check the Model

And lets check some images and model performance

```python
import zipfile
with zipfile.ZipFile("test1.zip", 'r') as zip_ref:
    zip_ref.extractall(".")
```

```python
import os
cats = []
dogs = []
testing_dataset = []
labels = []
for x in os.listdir("./test1/"):
    testing_dataset.append("./test1/" + x)

for x in os.listdir("./test/test/"):
    testing_dataset.append("./test/test/" + x)

testing_df = pd.DataFrame({
    'data': testing_dataset
})


testing_df.head()
```

```python
print(f'size of testing dataframe: {len(testing_df)}')
```

```python
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    testing_df,
    x_col='data',
    target_size=(300, 300),
    batch_size=128,
    class_mode=None,
    shuffle=False
)
```

```python
print(test_generator[0].shape, test_generator[1].shape)
print(len(test_generator))
```

```python
plt.figure(figsize=(12, 10))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    rand = random.randrange(len(test_generator))
    second_rand = random.randrange(128)
    img = test_generator[rand][second_rand]
    plt.imshow(img)
```

```python
predict = model.predict_generator(test_generator, steps=np.ceil(len(testing_df)//128))
```

```python
predict
```

```python
plt.figure(figsize=(15, 12))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    img = test_generator[0][10*i]
    label = predict[10*i]
    plt.title("cat" if label <= 0.5 else "dog")
    plt.imshow(img)
```

So as we can see out model is working well

```python
plt.figure(figsize=(15, 12))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    img = test_generator[0][10*i + 20]
    label = predict[10*i + 20]
    plt.title("cat" if label <= 0.5 else "dog")
    plt.imshow(img)
```

Another well result from the model. 
The rest of the notebook are some tests
too.

```python
import zipfile
with zipfile.ZipFile("gandom.zip", 'r') as zip_ref:
    zip_ref.extractall(".")
```

```python
import os
cats = []
dogs = []
gandom_dataset = []
labels = []
for x in os.listdir("./gandom/"):
    gandom_dataset.append("./gandom/" + x)

gandom_df = pd.DataFrame({
    'data': gandom_dataset
})


gandom_df.head()
```

```python
gandom_datagen = ImageDataGenerator(rescale=1./255)

gandom_generator = gandom_datagen.flow_from_dataframe(
    gandom_df,
    x_col='data',
    target_size=(300, 300),
    batch_size=128,
    class_mode=None,
    shuffle=False
)
```

```python
predict_gandom = model.predict(gandom_generator, steps=1)
```

```python
predict_gandom
```

```python
plt.figure(figsize=(15, 10))
for i in range(9):
    plt.subplot(2, 5, i + 1)
    img = gandom_generator[0][i]
    label = predict_gandom[i]
    plt.title("cat" if label <= 0.5 else "dog")
    plt.imshow(img)
```

```python
import zipfile
with zipfile.ZipFile("pinterest.zip", 'r') as zip_ref:
    zip_ref.extractall(".")
```

```python
import os

pinterest_dataset = []
labels = []
for x in os.listdir("./pinterest/"):
    pinterest_dataset.append("./pinterest/" + x)

pinterest_df = pd.DataFrame({
    'data': pinterest_dataset
})


pinterest_df.head()
```

```python
pinterest_datagen = ImageDataGenerator(rescale=1./255)

pinterest_generator = pinterest_datagen.flow_from_dataframe(
    pinterest_df,
    x_col='data',
    target_size=(300, 300),
    batch_size=128,
    class_mode=None,
    shuffle=False
)
```

```python
predict_pinterest = model.predict(pinterest_generator, steps=1)
```

```python
predict_pinterest
```

```python
plt.figure(figsize=(15, 10))
for i in range(6):
    plt.subplot(2, 5, i + 1)
    img = pinterest_generator[0][i]
    label = predict_pinterest[i]
    plt.title("cat" if label <= 0.5 else "dog")
    plt.imshow(img)
```
