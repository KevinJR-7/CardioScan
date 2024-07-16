import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Simulamos un dataset y entrenamiento de un modelo
def train_model():
    # Generar datos aleatorios para simular el entrenamiento
    X = np.random.rand(100, 50)  # 100 muestras, 50 características cada una
    y = np.random.randint(2, size=100)  # 0 (normal), 1 (anormal)

    # Entrenar un modelo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    return model

# Entrenar el modelo
model = train_model()

# Preprocesar la imagen y extraer características
def preprocess_and_extract_features(image_path):
    # Cargar la imagen en escala de grises
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("No se pudo cargar la imagen.")
    
    # Redimensionar la imagen a un tamaño consistente
    image = cv2.resize(image, (500, 500))

    # Binarizar la imagen
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    # Sumar las filas para obtener la señal de ECG
    signal = np.sum(binary_image, axis=1)

    # Filtrar la señal para eliminar el ruido
    def butter_lowpass_filter(data, cutoff, fs, order):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    fs = 500  # Frecuencia de muestreo (asumida)
    cutoff = 50  # Frecuencia de corte
    order = 5  # Orden del filtro

    filtered_signal = butter_lowpass_filter(signal, cutoff, fs, order)

    # Extraer características (usamos la señal filtrada directamente)
    features = filtered_signal[:50]  # Usar las primeras 50 características
    return features, filtered_signal

# Clasificar la imagen de ECG
def classify_ecg(image_path):
    try:
        features, filtered_signal = preprocess_and_extract_features(image_path)
        features_reshaped = features.reshape(1, -1)  # Redimensionar para el modelo
        prediction = model.predict(features_reshaped)
        return "Normal" if prediction[0] == 0 else "Anormal", filtered_signal
    except Exception as e:
        print(e)
        return "Error en el procesamiento de la imagen.", None

# Detectar picos QRS
def detect_qrs_peaks(signal, fs=500):
    peaks, _ = find_peaks(signal, distance=150)
    return peaks

# Calcular intervalos PR y QT (aproximación)
def calculate_intervals(peaks, fs):
    pr_intervals = []
    qt_intervals = []
    for i in range(1, len(peaks)):
        pr_interval = (peaks[i] - peaks[i-1]) / fs * 1000  # en ms
        qt_interval = (peaks[i] - peaks[i-1]) / fs * 1000  # en ms (aproximado)
        pr_intervals.append(pr_interval)
        qt_intervals.append(qt_interval)
    return pr_intervals, qt_intervals

# Crear la interfaz gráfica de usuario (GUI)
class ECGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificación de ECG")
        self.root.geometry("1000x600")  # Tamaño de la ventana ajustado

        # Fondo de la imagen
        self.background_image = Image.open("portada.jpg")
        self.background_image = self.background_image.resize((1000, 600), Image.LANCZOS)
        self.bg_image = ImageTk.PhotoImage(self.background_image)

        self.bg_label = tk.Label(root, image=self.bg_image)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.label = tk.Label(root, text="Cargar una imagen de ECG", bg="white")
        self.label.pack(pady=10)

        self.upload_btn = tk.Button(root, text="Cargar Imagen", command=self.upload_image)
        self.upload_btn.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Helvetica", 16), bg="white")
        self.result_label.pack(pady=10)

        self.canvas_frame = tk.Frame(root, bg="white")
        self.canvas_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.image_panel = None
        self.signal_canvas = None
        self.peaks_canvas = None

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.display_image(file_path)
            result, filtered_signal = classify_ecg(file_path)
            self.result_label.config(text=f"Resultado: {result}")
            if filtered_signal is not None:
                self.display_signal(filtered_signal)
                self.display_qrs_peaks(filtered_signal)

    def display_image(self, file_path):
        if self.image_panel:
            self.image_panel.pack_forget()

        img = Image.open(file_path)
        img = img.resize((200, 200), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)

        self.image_panel = tk.Label(self.root, image=img, bg="white")
        self.image_panel.image = img
        self.image_panel.pack(pady=10)

    def display_signal(self, filtered_signal):
        if self.signal_canvas:
            self.signal_canvas.get_tk_widget().pack_forget()

        fig, ax = plt.subplots(figsize=(4, 2))
        ax.plot(filtered_signal, label='Señal Filtrada')
        ax.set_title('Señal de ECG Filtrada')
        ax.set_xlabel('Muestras')
        ax.set_ylabel('Amplitud')
        ax.legend()

        self.signal_canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.signal_canvas.draw()
        self.signal_canvas.get_tk_widget().pack(side=tk.LEFT, padx=10)

    def display_qrs_peaks(self, filtered_signal):
        fs = 500  # Frecuencia de muestreo (asumida)
        peaks = detect_qrs_peaks(filtered_signal, fs)
        pr_intervals, qt_intervals = calculate_intervals(peaks, fs)

        if self.peaks_canvas:
            self.peaks_canvas.get_tk_widget().pack_forget()

        fig, ax = plt.subplots(figsize=(4, 2))
        ax.plot(filtered_signal, label='Señal Filtrada')
        ax.plot(peaks, filtered_signal[peaks], "x", label='Picos QRS')
        ax.set_title('Detección de Picos QRS en la Señal de ECG')
        ax.set_xlabel('Muestras')
        ax.set_ylabel('Amplitud')
        ax.legend()

        self.peaks_canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.peaks_canvas.draw()
        self.peaks_canvas.get_tk_widget().pack(side=tk.RIGHT, padx=10)

        print("Intervalos PR (ms):", pr_intervals)
        print("Intervalos QT (ms):", qt_intervals)

# Crear la ventana principal de la aplicación
root = tk.Tk()
app = ECGApp(root)
root.mainloop()
