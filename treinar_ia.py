import os 
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

#tranformar o audio em espectograma 

"""essa primeira carrega a biblioteca librosa e 
devolve duas coisas , y(onda do som em si ou velocicidade e o sr a taxa de amostragem)
"""
def extrair_espectrograma(caminho_audio):
    y, sr = librosa.load(caminho_audio, duration=3)
    if len(y) < 66150:
        y = np.pad(y, (0, 66150 - len(y))) #isso é para garantir que audio tenha 3 segundos
        espectrograma = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        #converter para a escala decibes 
        espectrograma_db = librosa.power_to_db(espectrograma, ref=np.max)
        return espectrograma_db

#carregar os dados salvos 
categorias = ["concentrado", "conversando", "rage"]
x = []
Y = []

for i, categoria in enumerate(categorias):
    pasta = f"dados/{categoria}"
    for arquivo in os.listdir(pasta):
        if arquivo.endswith(".wav"):
            caminho = os.path.join(pasta, arquivo)
            img = extrair_espectrograma(caminho)
            x.append(img)
            Y.append(i)
       
x = np.array(x)
Y = np.array(Y)
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

#dividir os dados em treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, Y, test_size=0.2, random_state=42)

#modelo da rede neural 
modelo = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(x.shape[1], x.shape[2], 1)),
    layers.MaxPooling2D((2, 2)),

layers.Flatten(),
layers.Dense(64, activation='relu'),
layers.Dense(3, activation='softmax')
])

#compilar e treinar
modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Treinando a IA...")
modelo.fit(x_treino, y_treino, epochs=15, validation_data=(x_teste, y_teste))

modelo.save("modelo_jogadores.h5")
print("\n🎉 IA treinada com sucesso! Arquivo 'modelo_jogadores.h5' gerado.")