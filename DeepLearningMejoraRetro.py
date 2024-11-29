#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Sección 1: Importar librerías y montar Google Drive
import os
import matplotlib.pyplot as plt
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

zip_path = 'datset.zip'
extract_path = './dataset'

# Descomprimir archivo ZIP
import zipfile
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Confirmar contenido
print("Archivos descomprimidos:")
print(os.listdir(extract_path))


# In[2]:


# Sección 2: Preparar datasets con image_dataset_from_directory

import tensorflow as tf

# Configuración de paths
train_dir = os.path.join(extract_path, 'seg_train/seg_train')
test_dir = os.path.join(extract_path, 'seg_test/seg_test')

# Crear datasets desde directorios
batch_size = 16
img_size = (224, 224)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode='int',
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True,
    seed=123
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    label_mode='int',
    batch_size=batch_size,
    image_size=img_size
)

# Dividir test_dataset en validación y prueba
val_size = int(len(test_dataset) * 0.3)  # 30% del conjunto de prueba
val_dataset = test_dataset.take(val_size)
test_dataset = test_dataset.skip(val_size)

# Guardar nombres de clases
class_names = train_dataset.class_names
print("Class names:", class_names)

# Normalización y prefetching
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


# In[3]:


# Sección 3: Contar imágenes en las carpetas
for dataset, name in zip([train_dataset, val_dataset, test_dataset], ['Train', 'Validation', 'Test']):
    print(f"{name} dataset batches: {len(dataset)}")


# In[4]:


# Sección 6: Visualizar imágenes del dataset de entrenamiento
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        img = (images[i].numpy() * 255).astype("uint8")
        plt.imshow(img)
        plt.title(class_names[labels[i].numpy()])
        plt.axis("off")


# In[5]:


# Sección 7: Verificar forma de un batch
for images, labels in train_dataset.take(1):
    print("Shape of image batch:", images.shape)
    print("Shape of label batch:", labels.shape)


# In[6]:


# Sección 6: Construir el modelo CNN con regularización y Dropout
input_layer = Input(shape=(224, 224, 3))

x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.2)(x)
output_layer = Dense(6, activation="softmax")(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()


# In[7]:


# Sección 7: Compilar y entrenar el modelo con Data Augmentation
from tensorflow.keras.optimizers import Adam

# Data augmentation
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),  # Rotación aleatoria hasta 20%
    RandomZoom(0.2)  # Zoom aleatorio hasta 20%
])

augmented_train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)

# Compilar el modelo con optimizador ajustado
optimizer = Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Ajustar EarlyStopping
early_stopping = EarlyStopping(patience=15, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(
    augmented_train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=[early_stopping]
)


# In[8]:


# Sección 10: Evaluación del modelo en el conjunto de entrenamiento
train_loss, train_acc = model.evaluate(train_dataset)
print(f"Training Accuracy: {train_acc:.3f}")
print(f"Training Loss: {train_loss:.3f}")


# In[ ]:


# Sección 11: Evaluación en el conjunto de prueba
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc:.3f}")
print(f"Test Loss: {test_loss:.3f}")


# In[10]:


# Sección 12: Visualización de métricas
plt.figure(figsize=(12, 4))

# Precisión
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()



# In[ ]:


import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from google.colab import files
from tensorflow.keras.models import load_model

# Subir hasta 6 imágenes
uploaded = files.upload()

# Verificar si se cargaron imágenes
if len(uploaded) > 0:
    # Lista para almacenar las imágenes preprocesadas
    batch_images = []

    for i, filename in enumerate(uploaded.keys()):
        if i >= 6:  # Limitar a 6 imágenes
            break

        # Cargar la imagen
        img_path = filename  # Nombre del archivo cargado
        img = image.load_img(img_path, target_size=(224, 224))  # Redimensionar la imagen
        plt.figure()
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Imagen Cargada: {filename}")
        plt.show()

        # Preprocesar la imagen
        img_array = image.img_to_array(img)  # Convertir a array
        img_array = img_array / 255.0  # Normalizar al rango [0, 1]
        batch_images.append(img_array)

    # Convertir la lista en un numpy array para predicción en batch
    batch_images = np.array(batch_images)

    # Realizar predicciones
    predictions = model.predict(batch_images)

    # Mostrar resultados para cada imagen
    for i, prediction in enumerate(predictions):
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]

        print(f"Imagen: {list(uploaded.keys())[i]}")
        print(f"Predicción: {class_names[predicted_class]}")
        print(f"Confianza: {confidence:.2f}")
        print("-" * 30)
else:
    print("No se cargaron imágenes.")

