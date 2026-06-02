import sounddevice as sd
from scipy.io.wavfile import write
import os 

#configurações de aúdio 
fs = 22050 
segundos = 3 

print ("Gravando...")
print("escolha onde salvar")
print("1- Concentrado (Silêncio/Teclado)")
print("2- Conversando (Call do jogo)")
print("3 - Rage (Grito/Frustração)") 

opcao = input("Digite o número da opção: ")
pastas = {"1": "dados/concentrado", "2": "dados/conversando", "3": "dados/rage"}

if opcao in pastas:
    pasta_destino = pastas[opcao]
    nome_arquivo = input("Digite o nome para o arquivo (ex: audio_01):") + ".wav"
    caminho_final = os.path.join(pasta_destino, nome_arquivo)

    os.makedirs(pasta_destino, exist_ok=True)
    print("\n Gravando em 3... 2... 1... FALE!")
    gravacao = sd.rec(int(segundos * fs), samplerate=fs, channels=1)
    sd.wait()

    write(caminho_final, fs, gravacao)
    print(f"Salvo com sucesso em : {caminho_final}\n")
else:
    print("Opção inválida")


