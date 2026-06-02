import os 
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

#tranformar o audio em espectograma 

"""essa primeira carrega a biblioteca librosa e 
devolve duas coisas , y(onda do som em si ou velocicidade e o sr a taxa de amostragem)
"""
def extrair_espectrograma(caminho_audio):
    try:
        # Tenta carregar o áudio
        y, sr = librosa.load(caminho_audio, duration=3, sr=22050)
        y = librosa.util.fix_length(y, size=66150)
            
        escolha = np.random.choice(["normal", "ruido", "pitch"])
        if escolha == "ruido":
            ruido = np.random.randn(len(y)) * 0.005
            y = y + ruido
        elif escolha == "pitch":
            y = librosa.effects.pitch_shift(y, sr= sr, n_steps=np.random.randint(-2, 3))
    
        espectrograma = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        espectrograma_db = librosa.power_to_db(espectrograma, ref=np.max)
        return espectrograma_db
        
    except Exception as e:
        # Se der QUALQUER erro ao abrir o arquivo, ele vai printar o motivo real aqui!
        print(f"\n Erro crítico ao ler o arquivo {caminho_audio}: {e}")
        return None
    
    
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

            # print de investigacao
            if img is None:
                print(f" Alerta: O arquivo {arquivo} na pasta {categoria} retornou None!")
            else:
                print(f" Arquivo: {arquivo} | Formato do Espectrograma: {img.shape}")
       
            x.append(img)
            Y.append(i)
       
x = np.array(x)
Y = np.array(Y)
print("O formato real de X é:", x.shape)
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

#dividir os dados em treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, Y, test_size=0.2, random_state=42)

#modelo da rede neural 
modelo = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(x.shape[1], x.shape[2], 1)),
    layers.MaxPooling2D((2, 2)),

    #segunda camada convolucional
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.GlobalAveragePooling2D(),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')
])

#compilar e treinar

modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

pesos = compute_class_weight('balanced', classes=np.unique(y_treino), y=y_treino)
dicionario_pesos = dict(enumerate(pesos))

print("\nTreinando a IA com proteção contra vícios (30 épocas)...")

modelo.fit(
    x_treino, y_treino, 
    epochs=30, 
    validation_data=(x_teste, y_teste),
    class_weight=dicionario_pesos # Força o equilíbrio matemático das pastas
)

modelo.save("modelo_jogadores.keras")
print("\n IA treinada com sucesso! Arquivo 'modelo_jogadores.keras' gerado e protegido.")