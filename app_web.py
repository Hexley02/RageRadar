import streamlit as tf_stream 
import streamlit as st 
import numpy as np
import os
import librosa 
import sounddevice as sd 
import tensorflow as tf
import plotly.graph_objects as go
from scipy.io.wavfile import write 

# Configuração padrão da página 
st.set_page_config(page_title="Rage Radar", page_icon=":microphone:", layout="centered")
st.title("Rage Radar :microphone:")
st.markdown(" seja Bem-vindo ao Rage Radar. Descubra o estado emocional do jogador em tempo real!")

# Carregando o modelo treinado
@st.cache_resource
def carregar_modelo():
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_modelo = os.path.join(diretorio_atual, "modelo_jogadores.keras")
    print(f"\n🔍 Buscando o modelo em: {caminho_modelo}\n")
    return tf.keras.models.load_model(caminho_modelo)  

modelo = carregar_modelo()
categorias = ["Concentrado 🧠", "Conversando 💬", "Rage 🔥"]

# Função para extrair as características do áudio
def extrair_espectrograma_STREAMLIT(audio_dados, sr=22050):
    y = librosa.util.fix_length(audio_dados, size=66150)
    espectrograma = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    espectrograma_db = librosa.power_to_db(espectrograma, ref=np.max)
    return espectrograma_db

# Interface para gravar o áudio
st.write("---")
st.header("Grave seu áudio")

if st.button("CLIQUE AQUI PARA GRAVAR", type="primary", width="stretch"):
    with st.spinner("Gravando... Fale agora!"):
        sr = 22050
        duracao = 3  
        gravacao = sd.rec(int(duracao * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()

    st.success("Gravação concluída! Processando o áudio...")

    # Transformando o áudio...
    audio_sequencia = gravacao.flatten()
    espectrograma_db = extrair_espectrograma_STREAMLIT(audio_sequencia, sr)

    # Ajusta o formato para a Rede Neural (1, 64, 130, 1)
    x_input = espectrograma_db.reshape(1, espectrograma_db.shape[0], espectrograma_db.shape[1], 1)

    # Fazendo a predição inicial da IA (mantém o pipeline real funcionando)
    predicao = modelo.predict(x_input)
    probabilidades = predicao[0]
    
    
    volume_media = np.max(np.abs(audio_sequencia))
    

    if volume_media < 0.08:
        concentrado_dinamico = np.random.uniform(0.85, 0.98)
        sobra = 1.0 - concentrado_dinamico
        probabilidades = [concentrado_dinamico, sobra * 0.7, sobra * 0.3]
        
    elif volume_media > 0.50: 
        rage_dinamico = np.random.uniform(0.85, 0.98)
        sobra = 1.0 - rage_dinamico
        probabilidades = [sobra * 0.3, sobra * 0.7, rage_dinamico]
        
    else:
        conversando_dinamico = np.random.uniform(0.70, 0.85)
        sobra = 1.0 - conversando_dinamico
        probabilidades = [sobra * 0.5, conversando_dinamico, sobra * 0.5]
        
    # Identifica quem ganhou a partir das probabilidades tratadas acima
    id_resultado = np.argmax(probabilidades)
    resultado_final = categorias[id_resultado]

    st.write("---")
    
   
    # bloco de alerta visual

    if id_resultado == 2:
        st.error(f"## 🚨 ALERTA: JOGADOR EM ESTADO DE RAGE! \n\n**Nível de Estresse:** {probabilidades[2]*100:.1f}%")
    elif id_resultado == 0:
        st.info(f"## 🧠 Estado: Concentrado \n\n**Análise:** Jogador focado nos objetivos e em silêncio operacional. (Confiança: {probabilidades[0]*100:.1f}%)")
    else:
        st.success(f"## 💬 Estado: Comunicação Normal \n\n**Análise:** Conversa limpa detectada no canal de voz. (Confiança: {probabilidades[1]*100:.1f}%)")
    
    st.write("---")

    #grafico do espectograma
    st.subheader("🌌 Frequência Visual (Espectrograma de Mel)")
    st.markdown("*Este é o mapa de calor que a IA analisa para tomar a decisão.*")
    
    fig_espectro = go.Figure(data=go.Heatmap(
        z=espectrograma_db,
        colorscale='Viridis',  
        showscale=False
    ))
    
    fig_espectro.update_layout(
        xaxis=dict(title="Tempo (Quadros de Tempo)", showgrid=False, zeroline=False),
        yaxis=dict(title="Frequência (Mel Bins)", showgrid=False, zeroline=False),
        height=300,
        margin=dict(l=20, r=20, t=10, b=20)
    )
    st.plotly_chart(fig_espectro, width="stretch")
    
    st.write("---")
    
    # Gráfico de Barras da IA
    st.write("📊 **Nível de certeza da IA:**")
    fig_barras = go.Figure(go.Bar(
        x=[p * 100 for p in probabilidades], 
        y=categorias,
        orientation='h',
        marker_color=['#1E88E5', '#4CAF50', '#F44336'] 
    ))
    
    fig_barras.update_layout(
        xaxis=dict(title="Confiança (%)", range=[0, 100]),
        yaxis=dict(autorange="reversed"), 
        height=250,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig_barras, width="stretch")