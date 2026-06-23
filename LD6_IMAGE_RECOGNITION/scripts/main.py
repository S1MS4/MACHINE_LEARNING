import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # išjungiam TF info/warning spam
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # išjungiam oneDNN žinutes
import numpy as np
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras import layers, models, optimizers # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.callbacks import EarlyStopping # pyright: ignore[reportMissingModuleSource]

#
# K1.1 – naudojam Dangus duomenų aibę, nieko nekeičiam
#

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data',)
TRAIN_DIR = os.path.join(DATA_DIR, 'learning_set')
TEST_DIR  = os.path.join(DATA_DIR, 'testing_set')

KLASĖS = sorted(os.listdir(TRAIN_DIR))  # cloudy, rain, shine, sunrise
print("Klasės:", KLASĖS)

# K1.2 – jokio augmentavimo, tiesiog originalūs paveikslėliai
# (nėra ImageDataGenerator su flip/rotation ir pan.)

#
# K2.1 – sukeliam paveikslėlius, suvienodinam dydį ir normalizuojam
#

IMG_SIZE = (64, 64)
# IMG_SIZE = (96, 96)  # padidinam kad CNN turėtų daugiau informacijos

def load_dataset(folder, color_mode='RGB'):
    X, y = [], []
    for label_idx, klase in enumerate(KLASĖS):
        klase_dir = os.path.join(folder, klase)
        for fname in os.listdir(klase_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img = Image.open(os.path.join(klase_dir, fname)).convert(color_mode)
            img = img.resize(IMG_SIZE)
            X.append(np.array(img))
            y.append(label_idx)
    X = np.array(X, dtype=np.float32) / 255.0  # normalizuojam į [0, 1]
    y = np.array(y)
    return X, y

# užkraunam RGB variantą
X_train_rgb, y_train = load_dataset(TRAIN_DIR, 'RGB')
X_test_rgb,  y_test  = load_dataset(TEST_DIR,  'RGB')

print(f"Treniravimo aibė: {X_train_rgb.shape}, Testavimo aibė: {X_test_rgb.shape}")

# 
# K2.2 – priskyriam ir išvedam paveikslėlių pavadinimus / etiketes
# 

print("\nPaveikslėlių skaičius pagal klasę (train):")
for i, k in enumerate(KLASĖS):
    print(f"  {k}: {np.sum(y_train == i)} vnt.")

print("\nPaveikslėlių skaičius pagal klasę (test):")
for i, k in enumerate(KLASĖS):
    print(f"  {k}: {np.sum(y_test == i)} vnt.")

# 
# K3.1 – bandome skirtingus išankstinio apdorojimo būdus
#         1) RGB (jau turim)
#         2) Grayscale (pilkos spalvos)
# 

X_train_gray, _ = load_dataset(TRAIN_DIR, 'L')
X_test_gray,  _ = load_dataset(TEST_DIR,  'L')

# pilkai reik pridėt kanalą kad keras būtų laimingas
X_train_gray = X_train_gray[..., np.newaxis]
X_test_gray  = X_test_gray[...,  np.newaxis]

print(f"\nRGB forma: {X_train_rgb.shape}, Grayscale forma: {X_train_gray.shape}")

# 
# K3.2 – DL4J nenaudojam nes tai Java biblioteka, o čia Python projektas.
#         Vietoj jo naudojam du Keras modelius: paprastą Dense tinklą ir CNN
# 

NUM_CLASSES = len(KLASĖS)

def build_dense(input_shape, neurons=128):
    # paprastas pilnai sujungtas tinklas
    m = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(neurons, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return m

def build_cnn(input_shape, filters=32):
    # konvoliucinis tinklas – turėtų geriau dirbti su paveikslėliais
    m = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(filters, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(filters*2, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return m

# 
# K3.3 – bandome skirtingus mokymo parametrus (learning rate, epochs)
# K3.4 – bandome skirtingus sluoksnių parametrus (neurons/filters)
# 

eksperimentai = [
    # (pavadinimas, modelio_fn, parametrai, X_train, X_test, lr, epochs)
    # ("Dense  neurons=64  lr=0.001 ep=10", build_dense, {'neurons': 64},   X_train_rgb, X_test_rgb, 0.001, 10),
    # ("Dense  neurons=128 lr=0.001 ep=10", build_dense, {'neurons': 128},  X_train_rgb, X_test_rgb, 0.001, 10),
    # ("Dense  neurons=128 lr=0.01  ep=10", build_dense, {'neurons': 128},  X_train_rgb, X_test_rgb, 0.01,  10),
    ("CNN    filters=16  lr=0.001 ep=10", build_cnn,   {'filters': 16},   X_train_rgb, X_test_rgb, 0.001, 10),
    ("CNN    filters=32  lr=0.001 ep=5", build_cnn,   {'filters': 32},   X_train_rgb, X_test_rgb, 0.001, 15),
    # ("CNN    filters=64  lr=0.002 ep=12", build_cnn,   {'filters': 64},   X_train_rgb, X_test_rgb, 0.002, 12), 
    # ("CNN    filters=32  lr=0.001 gray     ep=10", build_cnn,   {'filters': 32},   X_train_gray, X_test_gray, 0.001, 10),
]

rezultatai = []

for pav, model_fn, params, X_tr, X_te, lr, ep in eksperimentai:
    print(f"\n>>> Treniruojasi: {pav}")
    modelis = model_fn(X_tr.shape[1:], **params)
    modelis.compile(optimizer=optimizers.Adam(learning_rate=lr),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    #early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = modelis.fit(X_tr, y_train, epochs=ep, batch_size=32, verbose=0,
                          validation_split=0.1) #, callbacks=[early_stop]
    _, acc = modelis.evaluate(X_te, y_test, verbose=0)
    print(f"    Tikslumas: {acc:.4f}")
    rezultatai.append((pav, modelis, acc, X_te, history))

# 
# K3.5 – lyginam modelius, rodome geriausius parametrus
# 

print("\n\nMODELIŲ PALYGINIMAS")
rezultatai.sort(key=lambda x: x[2], reverse=True)
for pav, _, acc, _, _ in rezultatai:
    print(f"  {acc:.4f}  |  {pav}")

geriausias_pav, geriausias_modelis, geriausias_acc, geriausias_X_te, geriausias_history = rezultatai[0]
print(f"\nGeriausias modelis: {geriausias_pav}  ({geriausias_acc:.4f})")

# 
# Mokymosi kreivės geriausiam modeliui
# 

hist = geriausias_history.history
epochs_range = range(1, len(hist['accuracy']) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs_range, hist['accuracy'],     label='Training')
ax1.plot(epochs_range, hist['val_accuracy'], label='Validation')
ax1.set_title('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend()

ax2.plot(epochs_range, hist['loss'],     label='Training')
ax2.plot(epochs_range, hist['val_loss'], label='Validation')
ax2.set_title('Loss')
ax2.set_xlabel('Epoch')
ax2.legend()

fig.suptitle(f"Learning Curves\n{geriausias_pav}")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'images', 'learning_curve.png'))
plt.show()

# 
# K3.6 – klasifikavimo matrica geriausiam modeliui
# 

y_pred = np.argmax(geriausias_modelis.predict(geriausias_X_te, verbose=0), axis=1)

print("\nKlasifikavimo ataskaita:")
print(classification_report(y_test, y_pred, target_names=KLASĖS))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=KLASĖS)
disp.plot(cmap='Blues')
plt.title(f"Confusion Matrix\n{geriausias_pav}")
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'images', 'confusion_matrix.png'))
