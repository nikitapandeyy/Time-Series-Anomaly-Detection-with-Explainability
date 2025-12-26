import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from lstm_model import LSTM_Autoencoder

SEQ_LEN = 50

st.title("Time Series Anomaly Detection with Explainability")

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.line_chart(data)

    # Scale data and create sequences
    sensor_scaled = MinMaxScaler().fit_transform(data.values)
    sequences = np.array([sensor_scaled[i:i+SEQ_LEN] for i in range(len(sensor_scaled)-SEQ_LEN)])
    sequences = torch.FloatTensor(sequences)

    # Load trained model
    model = LSTM_Autoencoder(SEQ_LEN, sequences.shape[2])
    model.load_state_dict(torch.load("lstm_autoencoder.pth", map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        reconstructions = model(sequences)
        loss_per_seq = torch.mean((reconstructions - sequences)**2, dim=(1,2)).numpy()

    threshold = np.mean(loss_per_seq) + 3*np.std(loss_per_seq)
    anomalies = np.where(loss_per_seq > threshold)[0]

    st.write(f"Detected {len(anomalies)} anomalies")

    # Plot anomalies
    plt.figure(figsize=(12,4))
    plt.plot(data.values, label="Sensor Values")
    plt.scatter(anomalies, data.values[anomalies], color='red', label="Anomalies")
    plt.legend()
    st.pyplot()
