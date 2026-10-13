import json
import math
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import pyautogui
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import shift as nd_shift
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# PyTorch необов'язковий: якщо його немає, CNN замінюється на MLP
try:
    import torch
    import torch.nn as nn

    TORCH_OK = True
except ImportError:
    TORCH_OK = False

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
ARCHIVES_DIR = ROOT / "archives"
DATA_PATH = RESULTS_DIR / "gesture_samples.npz"
MODEL_PATH = MODELS_DIR / "gesture_classifier.joblib"
HAND_MODEL_PATH = ROOT / "hand_landmarker.task"
RUNTIME_HAND_MODEL_PATH = Path(tempfile.gettempdir()) / "lab3_hand_landmarker.task"

GESTURES = [
    "Ліва рука: 2 пальців",
    "Ліва рука: Лодонь",
    "Ліва рука: Палець вгору",
    "Ліва рука: Окей",
    "Ліва рука: Кулак",
    "Права рука: Лодонь",
    "Права рука: Серце",
]

FEATURES_PER_HAND = 21 * 3
MIN_SAMPLES_PER_CLASS = 10
CURSOR_MARGIN = 0.15

for directory in (MODELS_DIR, RESULTS_DIR, ARCHIVES_DIR):
    directory.mkdir(exist_ok=True)

pyautogui.FAILSAFE = False


def draw_vector_heart(img, center_x, center_y, scale=10, color=(0, 0, 255)):
    points = []
    for t in np.linspace(0, 2 * math.pi, 100):
        x = 16 * (math.sin(t) ** 3)
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        px = int(center_x + x * scale)
        py = int(center_y + y * scale)
        points.append([px, py])

    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], color)
    cv2.polylines(img, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)


class _SimplePoint:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class OneEuroFilter:
    def __init__(self, mincutoff=1.2, beta=0.6, dcutoff=1.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, t=None):
        if t is None:
            t = time.time()
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.dcutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class PaintCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(200, 200)
        self.image = QImage(200, 200, QImage.Format_Grayscale8)
        self.image.fill(Qt.black)
        self.drawing = False
        self.last_point = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.white, 14, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.position().toPoint())
            painter.end()
            self.last_point = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)

    def clear(self):
        self.image.fill(Qt.black)
        self.update()

    def get_image_array(self):
        # Повне зображення 200x200 у діапазоні 0..1 (без стиснення до 28x28)
        ptr = self.image.constBits()
        arr = np.frombuffer(ptr, np.uint8).reshape((200, 200)).copy()
        return arr / 255.0


# =========================================================
# 📝 ДОПОМІЖНІ ФУНКЦІЇ ДЛЯ ЛАБИ №8
# =========================================================
POSITIVE_TEXTS = [
    "Чудовий фільм, неймовірна гра акторів і цікавий сюжет!",
    "Мені дуже сподобалося, це був прекрасний досвід.",
    "Найкращий фільм, який я бачив, раджу всім!",
    "Відмінна якість, я дуже задоволений покупкою.",
    "Прекрасний сервіс і привітний персонал.",
    "Все супер, дякую за швидку доставку!",
    "Чудовий товар, працює відмінно і виглядає гарно.",
    "Дуже гарний досвід, рекомендую всім друзям.",
    "Якість перевершила мої очікування, вражаюче!",
    "Я в захваті, це просто чудово і красиво.",
    "Чудова акторська гра і зворушлива історія.",
    "Дуже задоволений, гроші витрачені не дарма.",
    "Зручно, комфортно і якісно, все сподобалось.",
    "Неймовірні враження, обов'язково повернуся ще раз.",
    "Найкраща покупка за останній час, супер!",
    "Приємний фільм, смішний і розумний, дивився із задоволенням.",
    "Відмінне обслуговування, ввічливі та уважні працівники.",
    "Смачна їжа, затишна атмосфера, все було чудово.",
    "Дуже радий, що вибрав саме цей товар, він бездоганний.",
    "Прекрасний день, настрій чудовий, все вдалося!",
    "Гарна робота, молодці, так тримати!",
    "Це було фантастично, я отримав справжню насолоду.",
    "Хороша ціна і висока якість, купуйте не вагаючись.",
    "Дякую, все пройшло чудово, залишився дуже задоволений.",
    "Дуже цікава і корисна книга, читається на одному диханні.",
    "Чудовий відпочинок, мальовнича природа і добрі люди.",
    "Мені подобається, зручний, надійний і красивий.",
    "Просто бездоганно, найкращий сервіс, який я зустрічав.",
    "Класний фільм, гарні емоції та чудова музика.",
    "Я щасливий, усе працює швидко і без помилок.",
]

NEGATIVE_TEXTS = [
    "Жахливий фільм, марна трата часу і грошей.",
    "Погана гра акторів, нудний сюжет і слабка режисура.",
    "Мені дуже не сподобалося, це було огидно.",
    "Жахливий досвід, погана якість.",
    "Найгірший сервіс, грубий персонал.",
    "Дуже розчарований, шкодую, що купив цей товар.",
    "Низька якість, зламалося вже за один день.",
    "Нудно і повільно, я заснув посеред фільму.",
    "Жахливий товар, зовсім не працює.",
    "Поганий досвід, більше ніколи сюди не прийду.",
    "Дурний сюжет і жахлива акторська гра.",
    "Бридка їжа і брудне місце, дуже неприємно.",
    "Не рекомендую, дешева і потворна якість.",
    "Повна дурниця, все погано і дратує.",
    "Я незадоволений, це найгірша покупка в моєму житті.",
    "Страшенно нудний фільм, тупий і безглуздий.",
    "Доставка запізнилася на тиждень, ніхто не відповідає на дзвінки.",
    "Обслуговування просто жахливе, мене обманули.",
    "Дуже поганий товар, гроші викинуті на вітер.",
    "Ненавиджу цей додаток, постійно вилітає з помилками.",
    "Жахлива робота, соромно за такий результат.",
    "Це було огидно, зіпсований настрій на весь день.",
    "Занадто дорого і дуже низька якість, не купуйте.",
    "Дякую за нічого, повний провал, залишився розчарованим.",
    "Нудна і марна книга, ледве дочитав до кінця.",
    "Жахливий відпочинок, брудно, галасливо і неприємний персонал.",
    "Мені не подобається, незручний, ненадійний і негарний.",
    "Просто катастрофа, найгірший сервіс, який я бачив.",
    "Поганий фільм, нудні діалоги та дратівлива музика.",
    "Я засмучений, усе працює повільно і з помилками.",
]


def make_text_vectorizer():
    # char_wb n-грами добре працюють з українською морфологією
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True, sublinear_tf=True)


def preprocess_digit(img):
    """200x200 (0..1) -> 8x8 (0..1) у стилі sklearn digits: обрізка, центрування, масштаб."""
    mask = img > 0.15
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.float32)
    top = (side - h) // 2
    left = (side - w) // 2
    square[top:top + h, left:left + w] = crop
    pad = int(side * 0.2)
    square = np.pad(square, pad)
    small = cv2.resize(square, (8, 8), interpolation=cv2.INTER_AREA)
    return small / max(small.max(), 1e-6)


def _augment_digits(images, labels, copies=4):
    rng = np.random.default_rng(42)
    xs, ys = [images], [labels]
    for _ in range(copies):
        aug = np.array([
            nd_shift(im, rng.uniform(-1, 1, 2), order=1, mode="constant")
            for im in images
        ])
        xs.append(aug)
        ys.append(labels)
    return np.concatenate(xs), np.concatenate(ys)


if TORCH_OK:
    class DigitCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),  # 8 -> 4
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),  # 4 -> 2
                nn.Flatten(),
                nn.Linear(32 * 2 * 2, 64), nn.ReLU(),
                nn.Linear(64, 10),
            )

        def forward(self, x):
            return self.net(x)


# =========================================================
# 📝 ВІКНО ЛАБОРАТОРНОЇ №8: РОЗПІЗНАВАННЯ ТЕКСТУ ТА CNN
# =========================================================
class TextCnnWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Lab 8 | Розпізнавання тексту та згорткові мережі (CNN)")
        self.resize(1000, 750)
        self.text_model = None
        self.vectorizer = make_text_vectorizer()
        self.cnn_model = None
        self.cnn_kind = None
        self.build_ui()
        self.init_default_text_model()

    def build_ui(self):
        layout = QVBoxLayout(self)

        top_bar = QHBoxLayout()
        load_text_btn = QPushButton("📁 Завантажити датасет текстів (CSV)")
        load_text_btn.clicked.connect(self.load_text_dataset)
        self.data_info_label = QLabel("Модель тексту: TF-IDF (символьні n-грами) + LogisticRegression")
        top_bar.addWidget(load_text_btn)
        top_bar.addWidget(self.data_info_label)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        main_split = QHBoxLayout()

        text_group = QGroupBox("1. Аналіз тексту (тональність)")
        text_layout = QVBoxLayout(text_group)

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Введіть текст повідомлення (наприклад: 'Цей фільм просто чудовий!')")
        self.text_input.setMaximumHeight(100)

        predict_text_btn = QPushButton("🔍 Розпізнати тональність тексту")
        predict_text_btn.clicked.connect(self.predict_text_sentiment)

        self.text_result_label = QLabel("Результат: --")
        self.text_result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #1a365d;")

        text_layout.addWidget(QLabel("Введення повідомлення:"))
        text_layout.addWidget(self.text_input)
        text_layout.addWidget(predict_text_btn)
        text_layout.addWidget(self.text_result_label)
        text_layout.addStretch()

        main_split.addWidget(text_group, 1)

        cnn_group = QGroupBox("2. Згорткова нейромережа / Малювання (CNN)")
        cnn_layout = QVBoxLayout(cnn_group)

        self.canvas_widget = PaintCanvas()
        clear_canvas_btn = QPushButton("🧹 Очистити полотно")
        clear_canvas_btn.clicked.connect(self.canvas_widget.clear)

        predict_cnn_btn = QPushButton("🧠 Розпізнати цифру (CNN)")
        predict_cnn_btn.clicked.connect(self.predict_canvas_cnn)

        self.cnn_result_label = QLabel("Результат CNN: --")
        self.cnn_result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2e7d32;")

        cnn_layout.addWidget(QLabel("Намалюйте цифру мишею (на все полотно):"))
        cnn_layout.addWidget(self.canvas_widget, alignment=Qt.AlignCenter)
        cnn_layout.addWidget(clear_canvas_btn)
        cnn_layout.addWidget(predict_cnn_btn)
        cnn_layout.addWidget(self.cnn_result_label)

        main_split.addWidget(cnn_group, 1)

        layout.addLayout(main_split)

        self.figure = Figure(figsize=(6, 3), tight_layout=True)
        self.plot_canvas = FigureCanvas(self.figure)
        layout.addWidget(self.plot_canvas)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(100)
        layout.addWidget(self.log_output)

    def init_default_text_model(self):
        texts = POSITIVE_TEXTS + NEGATIVE_TEXTS
        labels = [1] * len(POSITIVE_TEXTS) + [0] * len(NEGATIVE_TEXTS)
        self.vectorizer = make_text_vectorizer()
        x_vec = self.vectorizer.fit_transform(texts)
        self.text_model = LogisticRegression(C=20, max_iter=2000)
        self.text_model.fit(x_vec, labels)
        self.log_output.setPlainText(
            f"Базову модель аналізу тексту ініціалізовано ({len(texts)} українських прикладів)."
        )

    def load_text_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Завантажити текстовий датасет", str(RESULTS_DIR), "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
            self.vectorizer = make_text_vectorizer()
            x_vec = self.vectorizer.fit_transform(df.iloc[:, 0].astype(str))
            y = df.iloc[:, 1].values
            self.text_model = LogisticRegression(C=20, max_iter=2000)
            self.text_model.fit(x_vec, y)
            self.data_info_label.setText(f"Завантажено записів: {len(df)}")
            QMessageBox.information(self, "Успіх", "Текстовий датасет навчено!")
        except Exception as e:
            QMessageBox.critical(self, "Помилка", str(e))

    def predict_text_sentiment(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Увага", "Введіть текст повідомлення!")
            return

        x_vec = self.vectorizer.transform([text])
        pred = self.text_model.predict(x_vec)[0]
        probs = self.text_model.predict_proba(x_vec)[0]

        sentiment = "Позитивний 😊" if pred == 1 else "Негативний 😡"
        conf = max(probs) * 100
        self.text_result_label.setText(f"Результат: {sentiment} ({conf:.1f}%)")

        ax = self.figure.axes[0] if self.figure.axes else self.figure.add_subplot(111)
        ax.clear()
        ax.bar(["Негативний", "Позитивний"], probs, color=["red", "green"])
        ax.set_title("Ймовірності тональності тексту")
        ax.set_ylabel("Ймовірність")
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        self.plot_canvas.draw()

    def _ensure_cnn(self):
        if self.cnn_model is not None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            digits = load_digits()
            imgs = (digits.images / 16.0).astype(np.float32)
            x, y = _augment_digits(imgs, digits.target)
            x, y = x.astype(np.float32), y.astype(np.int64)

            if TORCH_OK:
                torch.manual_seed(42)
                model = DigitCNN()
                opt = torch.optim.Adam(model.parameters(), lr=2e-3)
                loss_fn = nn.CrossEntropyLoss()
                xt = torch.from_numpy(x).unsqueeze(1)
                yt = torch.from_numpy(y)
                for _epoch in range(15):
                    perm = torch.randperm(len(xt))
                    for i in range(0, len(xt), 128):
                        idx = perm[i:i + 128]
                        opt.zero_grad()
                        loss = loss_fn(model(xt[idx]), yt[idx])
                        loss.backward()
                        opt.step()
                model.eval()
                self.cnn_model = model
                self.cnn_kind = "CNN (PyTorch: Conv2D + MaxPool)"
            else:
                clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
                clf.fit(x.reshape(len(x), -1), y)
                self.cnn_model = clf
                self.cnn_kind = "MLP (sklearn, PyTorch не встановлено)"
        finally:
            QApplication.restoreOverrideCursor()

    def predict_canvas_cnn(self):
        img = self.canvas_widget.get_image_array()
        small = preprocess_digit(img)
        if small is None:
            QMessageBox.warning(self, "Увага", "Намалюйте цифру на полотні!")
            return

        self._ensure_cnn()
        if TORCH_OK and isinstance(self.cnn_model, nn.Module):
            with torch.no_grad():
                inp = torch.from_numpy(small.astype(np.float32)).reshape(1, 1, 8, 8)
                probs = torch.softmax(self.cnn_model(inp), dim=1)[0].numpy()
        else:
            probs = self.cnn_model.predict_proba(small.reshape(1, -1).astype(np.float32))[0]

        digit = int(np.argmax(probs))
        self.cnn_result_label.setText(
            f"Результат CNN: розпізнано цифру '{digit}' ({probs[digit] * 100:.1f}%)"
        )
        self.log_output.setPlainText(
            f"Модель: {self.cnn_kind}\nВхід 8x8 після обрізки та центрування."
        )

        ax = self.figure.axes[0] if self.figure.axes else self.figure.add_subplot(111)
        ax.clear()
        ax.bar(range(10), probs, color="#2e7d32")
        ax.set_xticks(range(10))
        ax.set_title("Ймовірності класів CNN (цифри 0-9)")
        ax.set_ylabel("Ймовірність")
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        self.plot_canvas.draw()


# =========================================================
# 🌳 ВІКНО ЛАБОРАТОРНОЇ №7: ДЕРЕВА РІШЕНЬ (Decision Trees)
# =========================================================
class DecisionTreeWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Lab 7 | Дерева рішень (Decision Forests)")
        self.resize(1000, 720)
        self.data_x = None
        self.data_y = None
        self.trained_model = None
        self.encoder = LabelEncoder()
        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)

        top_bar = QHBoxLayout()
        load_excel_btn = QPushButton("📁 Завантажити Excel/CSV")
        load_excel_btn.clicked.connect(self.load_dataset)
        load_npz_btn = QPushButton("📂 Завантажити датасет жестів (.npz)")
        load_npz_btn.clicked.connect(self.load_npz_dataset)
        self.data_info_label = QLabel("Дані не завантажено")

        top_bar.addWidget(load_excel_btn)
        top_bar.addWidget(load_npz_btn)
        top_bar.addWidget(self.data_info_label)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        main_split = QHBoxLayout()

        controls_box = QGroupBox("Параметри ансамблів дерев рішень")
        ctrl_layout = QFormLayout(controls_box)

        self.trees_count_spin = QSpinBox()
        self.trees_count_spin.setRange(5, 500)
        self.trees_count_spin.setValue(100)

        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(2, 50)
        self.max_depth_spin.setValue(10)

        self.test_size_spin = QDoubleSpinBox()
        self.test_size_spin.setRange(0.1, 0.5)
        self.test_size_spin.setSingleStep(0.05)
        self.test_size_spin.setValue(0.25)

        run_rf_btn = QPushButton("🌲 1. Випадковий Ліс (Random Forest)")
        run_rf_btn.clicked.connect(self.run_random_forest)

        run_gb_btn = QPushButton("⚡ 2. Градієнтний Бустінг (Gradient Boosting)")
        run_gb_btn.clicked.connect(self.run_gradient_boosting)

        load_model_btn = QPushButton("📂 Завантажити навчену модель (.joblib)")
        load_model_btn.clicked.connect(self.load_pretrained_model)

        save_res_btn = QPushButton("💾 Зберегти звіт / модель у файл")
        save_res_btn.clicked.connect(self.save_results)

        ctrl_layout.addRow("Кількість дерев (n_estimators):", self.trees_count_spin)
        ctrl_layout.addRow("Макс. глибина (max_depth):", self.max_depth_spin)
        ctrl_layout.addRow("Частка тестових даних:", self.test_size_spin)
        ctrl_layout.addRow(run_rf_btn)
        ctrl_layout.addRow(run_gb_btn)
        ctrl_layout.addRow(load_model_btn)
        ctrl_layout.addRow(save_res_btn)

        main_split.addWidget(controls_box, 1)

        self.figure = Figure(figsize=(6, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        main_split.addWidget(self.canvas, 2)

        layout.addLayout(main_split)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

    def load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Завантажити файл даних", str(RESULTS_DIR), "Data Files (*.xlsx *.csv)"
        )
        if not path:
            return
        try:
            df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
            self.data_x = df.iloc[:, :-1].values
            self.data_y = df.iloc[:, -1].values
            self.data_info_label.setText(f"Завантажено: {len(self.data_x)} рядків, {self.data_x.shape[1]} ознак")
            QMessageBox.information(self, "Успіх", "Дані завантажено!")
        except Exception as e:
            QMessageBox.critical(self, "Помилка", str(e))

    def load_npz_dataset(self):
        if not DATA_PATH.exists():
            QMessageBox.warning(self, "Помилка", "Датасет жестів не знайдено.")
            return
        data = np.load(DATA_PATH, allow_pickle=True)
        self.data_x = data["features"]
        self.data_y = self.encoder.fit_transform(data["labels"])
        self.data_info_label.setText(f"Завантажено з NPZ: {len(self.data_x)} прикладів")
        QMessageBox.information(self, "Успіх", "Жестовий датасет завантажено!")

    def run_random_forest(self):
        if self.data_x is None or self.data_y is None:
            QMessageBox.warning(self, "Увага", "Спочатку завантажте дані!")
            return

        x_train, x_test, y_train, y_test = train_test_split(
            self.data_x, self.data_y, test_size=self.test_size_spin.value(), random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=self.trees_count_spin.value(),
            max_depth=self.max_depth_spin.value(),
            random_state=42
        )
        model.fit(x_train, y_train)
        self.trained_model = model

        preds = model.predict(x_test)
        acc = accuracy_score(y_test, preds)

        self.plot_feature_importances(model.feature_importances_, "Важливість ознак (Random Forest)")
        self.log_output.setPlainText(
            f"Випадковий Ліс (Random Forest) навчено!\n"
            f"Кількість дерев: {self.trees_count_spin.value()}\n"
            f"Точність на тесті (Accuracy): {acc:.2%}"
        )

    def run_gradient_boosting(self):
        if self.data_x is None or self.data_y is None:
            QMessageBox.warning(self, "Увага", "Спочатку завантажте дані!")
            return

        x_train, x_test, y_train, y_test = train_test_split(
            self.data_x, self.data_y, test_size=self.test_size_spin.value(), random_state=42
        )

        model = GradientBoostingClassifier(
            n_estimators=self.trees_count_spin.value(),
            max_depth=min(self.max_depth_spin.value(), 10),
            random_state=42
        )
        model.fit(x_train, y_train)
        self.trained_model = model

        preds = model.predict(x_test)
        acc = accuracy_score(y_test, preds)

        self.plot_feature_importances(model.feature_importances_, "Важливість ознак (Gradient Boosting)")
        self.log_output.setPlainText(
            f"Градієнтний Бустінг (Gradient Boosting) навчено!\n"
            f"Кількість дерев: {self.trees_count_spin.value()}\n"
            f"Точність на тесті (Accuracy): {acc:.2%}"
        )

    def plot_feature_importances(self, importances, title):
        ax = self.figure.axes[0] if self.figure.axes else self.figure.add_subplot(111)
        ax.clear()
        top_indices = np.argsort(importances)[-15:]
        ax.barh(range(len(top_indices)), importances[top_indices], color="green")
        ax.set_yticks(range(len(top_indices)))
        ax.set_yticklabels([f"Ознака {i}" for i in top_indices])
        ax.set_title(title)
        ax.set_xlabel("Відносна важливість ознаки")
        ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def load_pretrained_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Завантажити навчену модель", str(MODELS_DIR), "Model Files (*.joblib)"
        )
        if path:
            try:
                bundle = joblib.load(path)
                self.trained_model = bundle.get("model", bundle)
                QMessageBox.information(self, "Успіх", f"Модель завантажено: {Path(path).name}")
                self.log_output.setPlainText(f"Завантажено модель дерев рішення з {path}")
            except Exception as e:
                QMessageBox.critical(self, "Помилка", str(e))

    def save_results(self):
        if self.trained_model is None:
            QMessageBox.warning(self, "Увага", "Немає навченої моделі для збереження.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Зберегти модель Дерева Рішень", str(MODELS_DIR / "tree_model.joblib"), "Model Files (*.joblib)"
        )
        if path:
            joblib.dump({"model": self.trained_model, "encoder": self.encoder}, path)
            QMessageBox.information(self, "Збережено", f"Модель збережено у {path}")


# =========================================================
# 📈 ВІКНО ЛАБОРАТОРНОЇ №6: ФУНКЦІОНАЛЬНІ ЗАЛЕЖНОСТІ
# =========================================================
class FunctionalDependencyWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Lab 6 | Функціональні залежності та регресія")
        self.resize(1000, 720)
        self.data_x = None
        self.data_y = None
        self.y_test = None
        self.y_pred = None
        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)

        top_bar = QHBoxLayout()
        load_excel_btn = QPushButton("📁 Завантажити Excel/CSV")
        load_excel_btn.clicked.connect(self.load_dataset)
        load_npz_btn = QPushButton("📂 Завантажити датасет жестів (.npz)")
        load_npz_btn.clicked.connect(self.load_npz_dataset)
        self.data_info_label = QLabel("Дані не завантажено")

        top_bar.addWidget(load_excel_btn)
        top_bar.addWidget(load_npz_btn)
        top_bar.addWidget(self.data_info_label)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        main_split = QHBoxLayout()

        controls_box = QGroupBox("Параметри регресійних моделей")
        ctrl_layout = QFormLayout(controls_box)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 1000)
        self.epochs_spin.setValue(200)

        self.test_size_spin = QDoubleSpinBox()
        self.test_size_spin.setRange(0.1, 0.5)
        self.test_size_spin.setSingleStep(0.05)
        self.test_size_spin.setValue(0.20)

        run_lin_reg_btn = QPushButton("📊 1. Лінійна / Множинна Регресія")
        run_lin_reg_btn.clicked.connect(self.run_linear_regression)

        run_dnn_reg_btn = QPushButton("🧠 2. Глибинна Нейронна Мережа (DNN)")
        run_dnn_reg_btn.clicked.connect(self.run_dnn_regression)

        save_res_btn = QPushButton("💾 Зберегти результати у файл")
        save_res_btn.clicked.connect(self.save_results)

        ctrl_layout.addRow("Епохи / Ітерації:", self.epochs_spin)
        ctrl_layout.addRow("Частка тестових даних:", self.test_size_spin)
        ctrl_layout.addRow(run_lin_reg_btn)
        ctrl_layout.addRow(run_dnn_reg_btn)
        ctrl_layout.addRow(save_res_btn)

        main_split.addWidget(controls_box, 1)

        self.figure = Figure(figsize=(6, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        main_split.addWidget(self.canvas, 2)

        layout.addLayout(main_split)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

    def load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Завантажити файл даних", str(RESULTS_DIR), "Data Files (*.xlsx *.csv)"
        )
        if not path:
            return
        try:
            df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
            self.data_x = df.iloc[:, :-1].values
            self.data_y = df.iloc[:, -1].values
            self.data_info_label.setText(f"Завантажено: {len(self.data_x)} рядків, {self.data_x.shape[1]} ознак")
            QMessageBox.information(self, "Успіх", "Дані завантажено!")
        except Exception as e:
            QMessageBox.critical(self, "Помилка", str(e))

    def load_npz_dataset(self):
        if not DATA_PATH.exists():
            QMessageBox.warning(self, "Помилка", "Датасет жестів не знайдено.")
            return
        data = np.load(DATA_PATH, allow_pickle=True)
        self.data_x = data["features"]
        encoder = LabelEncoder()
        self.data_y = encoder.fit_transform(data["labels"]).astype(float)
        self.data_info_label.setText(f"Завантажено з NPZ: {len(self.data_x)} прикладів")
        QMessageBox.information(self, "Успіх", "Жестовий датасет завантажено!")

    def run_linear_regression(self):
        if self.data_x is None or self.data_y is None:
            QMessageBox.warning(self, "Увага", "Спочатку завантажте дані!")
            return

        x_train, x_test, y_train, y_test = train_test_split(
            self.data_x, self.data_y, test_size=self.test_size_spin.value(), random_state=42
        )

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        model = LinearRegression()
        model.fit(x_train_scaled, y_train)

        self.y_test = y_test
        self.y_pred = model.predict(x_test_scaled)

        mse = mean_squared_error(y_test, self.y_pred)
        r2 = r2_score(y_test, self.y_pred)

        self.plot_predictions("Лінійна регресія (Реальні значення vs Прогноз)")
        self.log_output.setPlainText(
            f"Множинна Лінійна Регресія завершена!\n"
            f"Mean Squared Error (MSE): {mse:.4f}\n"
            f"Коефіцієнт детермінації (R² Score): {r2:.4f}"
        )

    def run_dnn_regression(self):
        if self.data_x is None or self.data_y is None:
            QMessageBox.warning(self, "Увага", "Спочатку завантажте дані!")
            return

        x_train, x_test, y_train, y_test = train_test_split(
            self.data_x, self.data_y, test_size=self.test_size_spin.value(), random_state=42
        )

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=self.epochs_spin.value(),
            random_state=42
        )
        model.fit(x_train_scaled, y_train)

        self.y_test = y_test
        self.y_pred = model.predict(x_test_scaled)

        mse = mean_squared_error(y_test, self.y_pred)
        r2 = r2_score(y_test, self.y_pred)

        ax = self.figure.axes[0] if self.figure.axes else self.figure.add_subplot(111)
        ax.clear()
        ax.plot(model.loss_curve_, color="purple", linewidth=2, label="Крива втрат (MSE)")
        ax.set_title("Навчання Глибинної Нейромережі (DNN Loss)")
        ax.set_xlabel("Епохи (Epochs)")
        ax.set_ylabel("Похибка (Loss)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        self.canvas.draw()

        self.log_output.setPlainText(
            f"Глибинна Нейронна Мережа (DNN) завершена!\n"
            f"Mean Squared Error (MSE): {mse:.4f}\n"
            f"Коефіцієнт детермінації (R² Score): {r2:.4f}\n"
            f"Кількість ітерацій: {len(model.loss_curve_)}"
        )

    def plot_predictions(self, title):
        ax = self.figure.axes[0] if self.figure.axes else self.figure.add_subplot(111)
        ax.clear()
        ax.scatter(self.y_test, self.y_pred, color="blue", alpha=0.7, label="Передбачення")
        ax.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], 'r--', label="Ідеальна залежність")
        ax.set_title(title)
        ax.set_xlabel("Реальні значення Y")
        ax.set_ylabel("Передбачені значення Y'")
        ax.grid(True, alpha=0.3)
        ax.legend()
        self.canvas.draw()

    def save_results(self):
        if self.y_pred is None:
            QMessageBox.warning(self, "Увага", "Немає результатів для збереження.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Зберегти звіт регресії", str(RESULTS_DIR / "regression_report.csv"), "CSV Files (*.csv)"
        )
        if path:
            df = pd.DataFrame({"Actual": self.y_test, "Predicted": self.y_pred, "Error": self.y_test - self.y_pred})
            df.to_csv(path, index=False)
            QMessageBox.information(self, "Збережено", f"Звіт збережено у {path}")


# =========================================================
# 🧬 ВІКНО ГЕНЕТИЧНИХ АЛГОРИТМІВ (Custom GA)
# =========================================================
class CustomGeneticAlgorithm:
    def __init__(self, num_genes, pop_size=20, generations=50, mutation_rate=0.1):
        self.num_genes = num_genes
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = np.random.uniform(-1.0, 1.0, (pop_size, num_genes))
        self.history = []

    def fitness_function(self, chromosome, X, y, is_classification=False):
        if is_classification:
            logits = np.dot(X, chromosome)
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -15, 15)))
            preds = (probs > 0.5).astype(int)
            acc = np.mean(preds == y)
            return acc + 1e-6
        else:
            preds = np.dot(X, chromosome)
            mse = np.mean((y - preds) ** 2)
            return 1.0 / (mse + 1e-6)

    def run(self, X, y, is_classification=False):
        self.history = []
        best_chrom = None
        best_fit = -1

        for gen in range(self.generations):
            fitnesses = np.array([self.fitness_function(c, X, y, is_classification) for c in self.population])

            idx_best = np.argmax(fitnesses)
            if fitnesses[idx_best] > best_fit:
                best_fit = fitnesses[idx_best]
                best_chrom = self.population[idx_best].copy()

            self.history.append(best_fit)

            probs = fitnesses / np.sum(fitnesses)
            selected_indices = np.random.choice(self.pop_size, size=self.pop_size, p=probs)
            parents = self.population[selected_indices]

            new_population = []
            for i in range(0, self.pop_size, 2):
                p1, p2 = parents[i], parents[(i + 1) % self.pop_size]
                point = np.random.randint(1, self.num_genes) if self.num_genes > 1 else 0
                child1 = np.concatenate([p1[:point], p2[point:]])
                child2 = np.concatenate([p2[:point], p1[point:]])
                new_population.extend([child1, child2])

            self.population = np.array(new_population[:self.pop_size])

            mutation_mask = np.random.rand(*self.population.shape) < self.mutation_rate
            noise = np.random.normal(0, 0.2, size=self.population.shape)
            self.population += mutation_mask * noise

        return best_chrom, best_fit


class GeneticAnalysisWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Lab 5 | Генетичний алгоритм (Custom GA)")
        self.resize(1000, 700)
        self.data_x = None
        self.data_y = None
        self.ga_history = []
        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)

        top_bar = QHBoxLayout()
        load_excel_btn = QPushButton("📁 Завантажити Excel/CSV")
        load_excel_btn.clicked.connect(self.load_dataset)
        load_npz_btn = QPushButton("📂 Завантажити датасет жестів (.npz)")
        load_npz_btn.clicked.connect(self.load_npz_dataset)
        self.data_info_label = QLabel("Дані не завантажено")

        top_bar.addWidget(load_excel_btn)
        top_bar.addWidget(load_npz_btn)
        top_bar.addWidget(self.data_info_label)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        main_split = QHBoxLayout()

        controls_box = QGroupBox("Параметри генетичного алгоритму")
        ctrl_layout = QFormLayout(controls_box)

        self.generations_spin = QSpinBox()
        self.generations_spin.setRange(5, 500)
        self.generations_spin.setValue(50)

        self.sol_per_pop_spin = QSpinBox()
        self.sol_per_pop_spin.setRange(4, 100)
        self.sol_per_pop_spin.setValue(20)

        self.mutation_percent_spin = QDoubleSpinBox()
        self.mutation_percent_spin.setRange(0.01, 1.0)
        self.mutation_percent_spin.setSingleStep(0.05)
        self.mutation_percent_spin.setValue(0.10)

        run_ga_cls_btn = QPushButton("🧬 1. Генетична Класифікація (Accuracy)")
        run_ga_cls_btn.clicked.connect(self.run_ga_classification)

        run_ga_reg_btn = QPushButton("⚙️ 2. Генетична Оптимізація (MSE Loss)")
        run_ga_reg_btn.clicked.connect(self.run_ga_regression)

        save_res_btn = QPushButton("💾 Зберегти звіт у файл")
        save_res_btn.clicked.connect(self.save_results)

        ctrl_layout.addRow("Кількість поколінь:", self.generations_spin)
        ctrl_layout.addRow("Розмір популяції:", self.sol_per_pop_spin)
        ctrl_layout.addRow("Ймовірність мутацій:", self.mutation_percent_spin)
        ctrl_layout.addRow(run_ga_cls_btn)
        ctrl_layout.addRow(run_ga_reg_btn)
        ctrl_layout.addRow(save_res_btn)

        main_split.addWidget(controls_box, 1)

        self.figure = Figure(figsize=(6, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        main_split.addWidget(self.canvas, 2)

        layout.addLayout(main_split)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

    def load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Завантажити файл даних", str(RESULTS_DIR), "Data Files (*.xlsx *.csv)"
        )
        if not path:
            return
        try:
            df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
            self.data_x = df.iloc[:, :-1].values
            self.data_y = df.iloc[:, -1].values
            self.data_info_label.setText(f"Завантажено: {len(self.data_x)} рядків")
            QMessageBox.information(self, "Успіх", "Дані завантажено!")
        except Exception as e:
            QMessageBox.critical(self, "Помилка", str(e))

    def load_npz_dataset(self):
        if not DATA_PATH.exists():
            QMessageBox.warning(self, "Помилка", "Датасет жестів не знайдено.")
            return
        data = np.load(DATA_PATH, allow_pickle=True)
        self.data_x = data["features"]
        encoder = LabelEncoder()
        self.data_y = (encoder.fit_transform(data["labels"]) > 0).astype(int)
        self.data_info_label.setText(f"Завантажено NPZ: {len(self.data_x)} прикладів")
        QMessageBox.information(self, "Успіх", "Жестовий датасет завантажено!")

    def run_ga_classification(self):
        if self.data_x is None or self.data_y is None:
            QMessageBox.warning(self, "Увага", "Спочатку завантажте дані!")
            return

        ga = CustomGeneticAlgorithm(
            num_genes=self.data_x.shape[1],
            pop_size=self.sol_per_pop_spin.value(),
            generations=self.generations_spin.value(),
            mutation_rate=self.mutation_percent_spin.value()
        )

        best_chrom, best_fit = ga.run(self.data_x, self.data_y, is_classification=True)
        self.ga_history = ga.history
        self.plot_fitness("Генетична класифікація (Accuracy Fitness)")
        self.log_output.setPlainText(
            f"Генетичну класифікацію завершено!\n"
            f"Найкраща пристосованість (Accuracy/Fitness): {best_fit:.4f}\n"
            f"Кількість хромосом (ваг): {len(best_chrom)}"
        )

    def run_ga_regression(self):
        if self.data_x is None or self.data_y is None:
            QMessageBox.warning(self, "Увага", "Спочатку завантажте дані!")
            return

        ga = CustomGeneticAlgorithm(
            num_genes=self.data_x.shape[1],
            pop_size=self.sol_per_pop_spin.value(),
            generations=self.generations_spin.value(),
            mutation_rate=self.mutation_percent_spin.value()
        )

        best_chrom, best_fit = ga.run(self.data_x, self.data_y, is_classification=False)
        self.ga_history = ga.history
        self.plot_fitness("Генетична оптимізація (1 / MSE Fitness)")
        self.log_output.setPlainText(
            f"Генетичну оптимізацію завершено!\n"
            f"Найкраща пристосованість (Fitness): {best_fit:.4f}\n"
            f"MSE Loss: {1.0 / best_fit:.6f}"
        )

    def plot_fitness(self, title):
        ax = self.figure.axes[0] if self.figure.axes else self.figure.add_subplot(111)
        ax.clear()
        ax.plot(self.ga_history, marker="o", color="blue", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Покоління (Generations)")
        ax.set_ylabel("Пристосованість (Fitness)")
        ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def save_results(self):
        if not self.ga_history:
            QMessageBox.warning(self, "Увага", "Немає результатів для збереження.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Зберегти звіт ГА", str(RESULTS_DIR / "genetic_report.txt"), "Text Files (*.txt)"
        )
        if path:
            df = pd.DataFrame({"Generation": range(len(self.ga_history)), "Fitness": self.ga_history})
            df.to_csv(path, index=False)
            QMessageBox.information(self, "Збережено", f"Звіт збережено у {path}")


# =========================================================
# 🧠 КЛАСИ НЕЧІТКОГО АНАЛІЗУ (Fuzzy Logic Models)
# =========================================================
class FuzzyCMeans:
    def __init__(self, n_clusters=3, m=2.0, max_iter=100, error=1e-5):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error
        self.u = None
        self.centers = None

    def fit(self, X):
        N, d = X.shape
        self.u = np.random.dirichlet(np.ones(self.n_clusters), size=N)

        for _ in range(self.max_iter):
            u_old = self.u.copy()
            um = self.u**self.m
            self.centers = np.dot(um.T, X) / np.sum(um, axis=0)[:, None]

            dist = np.zeros((N, self.n_clusters))
            for k in range(self.n_clusters):
                dist[:, k] = np.linalg.norm(X - self.centers[k], axis=1)

            dist = np.fmax(dist, 1e-10)
            inv_dist = 1.0 / dist
            inv_dist_m = inv_dist ** (2.0 / (self.m - 1.0))
            self.u = inv_dist_m / np.sum(inv_dist_m, axis=1, keepdims=True)

            if np.linalg.norm(self.u - u_old) < self.error:
                break


class FuzzyClassifier:
    def __init__(self, sigma=1.0):
        self.sigma = sigma
        self.classes_ = None
        self.centers_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centers_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])

    def predict_proba(self, X):
        dists = np.array([np.linalg.norm(X - center, axis=1) for center in self.centers_]).T
        memberships = np.exp(-((dists / self.sigma) ** 2))
        sum_m = np.sum(memberships, axis=1, keepdims=True)
        sum_m = np.where(sum_m == 0, 1e-10, sum_m)
        return memberships / sum_m

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class FuzzyAnalysisWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Lab 4 | Нечіткий аналіз даних (Fuzzy Logic)")
        self.resize(1000, 700)
        self.data_x = None
        self.data_y = None
        self.fcm_model = None
        self.fuzzy_clf = None
        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)

        top_bar = QHBoxLayout()
        load_excel_btn = QPushButton("📁 Завантажити Excel/CSV")
        load_excel_btn.clicked.connect(self.load_dataset)
        load_npz_btn = QPushButton("📂 Завантажити з 1-3 Лаби (.npz)")
        load_npz_btn.clicked.connect(self.load_npz_dataset)
        self.data_info_label = QLabel("Дані не завантажено")

        top_bar.addWidget(load_excel_btn)
        top_bar.addWidget(load_npz_btn)
        top_bar.addWidget(self.data_info_label)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        main_split = QHBoxLayout()

        controls_box = QGroupBox("Параметри нечітких моделей")
        ctrl_layout = QFormLayout(controls_box)

        self.cluster_count_spin = QSpinBox()
        self.cluster_count_spin.setRange(2, 10)
        self.cluster_count_spin.setValue(3)

        self.fuzziness_spin = QDoubleSpinBox()
        self.fuzziness_spin.setRange(1.1, 5.0)
        self.fuzziness_spin.setSingleStep(0.1)
        self.fuzziness_spin.setValue(2.0)

        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.1, 10.0)
        self.sigma_spin.setSingleStep(0.2)
        self.sigma_spin.setValue(1.5)

        run_fcm_btn = QPushButton("1️⃣ Нечітка кластеризація (FCM)")
        run_fcm_btn.clicked.connect(self.run_fcm)

        run_clf_btn = QPushButton("2️⃣ Нечітка класифікація")
        run_clf_btn.clicked.connect(self.run_fuzzy_classification)

        save_res_btn = QPushButton("💾 Зберегти результати у файл")
        save_res_btn.clicked.connect(self.save_results)

        ctrl_layout.addRow("Кількість кластерів (K):", self.cluster_count_spin)
        ctrl_layout.addRow("Експонента нечіткості (m):", self.fuzziness_spin)
        ctrl_layout.addRow("Сигма гаусіани (σ):", self.sigma_spin)
        ctrl_layout.addRow(run_fcm_btn)
        ctrl_layout.addRow(run_clf_btn)
        ctrl_layout.addRow(save_res_btn)

        main_split.addWidget(controls_box, 1)

        self.figure = Figure(figsize=(6, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        main_split.addWidget(self.canvas, 2)

        layout.addLayout(main_split)

        self.table = QTableWidget()
        layout.addWidget(self.table)

    def load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Завантажити файл наборів даних", str(RESULTS_DIR), "Data Files (*.xlsx *.csv)"
        )
        if not path:
            return
        try:
            df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
            self.data_x = df.iloc[:, :-1].values
            self.data_y = df.iloc[:, -1].values
            self.data_info_label.setText(f"Завантажено: {len(self.data_x)} рядків")
            QMessageBox.information(self, "Успіх", "Дані завантажено!")
        except Exception as e:
            QMessageBox.critical(self, "Помилка", str(e))

    def load_npz_dataset(self):
        if not DATA_PATH.exists():
            QMessageBox.warning(self, "Помилка", "Датасет жестів не знайдено.")
            return
        data = np.load(DATA_PATH, allow_pickle=True)
        self.data_x = data["features"]
        self.data_y = data["labels"]
        self.data_info_label.setText(f"Завантажено з NPZ: {len(self.data_x)} прикладів")
        QMessageBox.information(self, "Успіх", "Датасет завантажено!")

    def run_fcm(self):
        if self.data_x is None:
            return
        k = self.cluster_count_spin.value()
        m = self.fuzziness_spin.value()
        self.fcm_model = FuzzyCMeans(n_clusters=k, m=m)
        self.fcm_model.fit(self.data_x)

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(self.data_x)

        ax = self.figure.axes[0] if self.figure.axes else self.figure.add_subplot(111)
        ax.clear()
        hard_clusters = np.argmax(self.fcm_model.u, axis=1)
        ax.scatter(reduced[:, 0], reduced[:, 1], c=hard_clusters, cmap="tab10", alpha=0.7)

        centers_reduced = pca.transform(self.fcm_model.centers)
        ax.scatter(centers_reduced[:, 0], centers_reduced[:, 1], c="red", marker="X", s=150, label="Нечіткі центри")
        ax.set_title(f"Fuzzy C-Means (K={k}, m={m})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.canvas.draw()

        self.update_table(self.fcm_model.u, [f"Кластер {i+1}" for i in range(k)])

    def run_fuzzy_classification(self):
        if self.data_x is None or self.data_y is None:
            return
        sigma = self.sigma_spin.value()
        self.fuzzy_clf = FuzzyClassifier(sigma=sigma)
        self.fuzzy_clf.fit(self.data_x, self.data_y)

        probs = self.fuzzy_clf.predict_proba(self.data_x)

        ax = self.figure.axes[0] if self.figure.axes else self.figure.add_subplot(111)
        ax.clear()
        classes = self.fuzzy_clf.classes_
        for i, c in enumerate(classes):
            ax.plot(probs[:50, i], label=f"Належність до {c}")
        ax.set_title(f"Нечітка класифікація (σ={sigma})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.canvas.draw()

        self.update_table(probs, [f"Клас: {c}" for c in classes])

    def update_table(self, matrix, col_names):
        self.table.clear()
        self.table.setRowCount(min(100, len(matrix)))
        self.table.setColumnCount(len(col_names))
        self.table.setHorizontalHeaderLabels(col_names)

        for r in range(min(100, len(matrix))):
            for c in range(len(col_names)):
                self.table.setItem(r, c, QTableWidgetItem(f"{matrix[r, c]:.4f}"))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def save_results(self):
        if self.fcm_model is None and self.fuzzy_clf is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Зберегти звіт", str(RESULTS_DIR / "fuzzy_report.csv"), "CSV Files (*.csv)"
        )
        if path:
            matrix = self.fcm_model.u if self.fcm_model is not None else self.fuzzy_clf.predict_proba(self.data_x)
            pd.DataFrame(matrix).to_csv(path, index=False)
            QMessageBox.information(self, "Збережено", f"Файл збережено: {path}")


# =========================================================
# 🖥️ ОСНОВНИЙ ІНТЕРФЕЙС ТА ГОЛОВНЕ ВІКНО (Лаби 1-3)
# =========================================================
class GestureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lab 3 | Розпізнавання жестів")
        self.resize(1120, 760)
        self.camera = None
        self.mouse_control_active = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.gesture1_start_time = None
        self.gesture2_start_time = None
        self.gesture3_start_time = None
        self.gesture4_start_time = None
        self.right_palm_start_time = None

        self.required_hold_duration = 1.0

        self.cursor_filter_x = OneEuroFilter(mincutoff=0.2, beta=0.01, dcutoff=1.0)
        self.cursor_filter_y = OneEuroFilter(mincutoff=0.2, beta=0.01, dcutoff=1.0)

        self.countdown_timer = QTimer(self)
        self.countdown_timer.setInterval(1000)
        self.countdown_timer.timeout.connect(self._handle_countdown)
        self.is_counting_down = False
        self._feature_buffer = []

        if not HAND_MODEL_PATH.exists():
            raise FileNotFoundError(f"Не знайдено модель руки: {HAND_MODEL_PATH}.")

        shutil.copyfile(HAND_MODEL_PATH, RUNTIME_HAND_MODEL_PATH)
        base_options = python.BaseOptions(model_asset_path=str(RUNTIME_HAND_MODEL_PATH))
        self.hand_landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=0.65,
                min_hand_presence_confidence=0.65,
                min_tracking_confidence=0.65,
            )
        )

        self.current_features = None
        self.model = None
        self.encoder = LabelEncoder()
        self.samples = []
        self.labels = []
        self.countdown_val = 0

        self.build_ui()
        self.load_samples()
        self.load_model_if_exists()

    def build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        top = QHBoxLayout()
        self.video_label = QLabel("Камера вимкнена")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background:#16232b; color:#d9f1ec; font-size:20px;")
        top.addWidget(self.video_label, 3)

        side_widget = QWidget()
        side = QVBoxLayout(side_widget)
        side.setContentsMargins(5, 5, 5, 5)

        camera_box = QGroupBox("Камера")
        camera_layout = QVBoxLayout(camera_box)
        start_button = QPushButton("Запустити камеру")
        start_button.clicked.connect(self.start_camera)
        stop_button = QPushButton("Зупинити камеру")
        stop_button.clicked.connect(self.stop_camera)
        camera_layout.addWidget(start_button)
        camera_layout.addWidget(stop_button)
        side.addWidget(camera_box)

        actions_box = QGroupBox("Автоматизація і дії")
        actions_layout = QVBoxLayout(actions_box)
        self.enable_actions_checkbox = QCheckBox("Актувати гарячі дії (pyautogui)")
        self.enable_actions_checkbox.setChecked(False)
        self.enable_actions_checkbox.setStyleSheet("font-weight: bold; color: #2e7d32;")
        actions_layout.addWidget(self.enable_actions_checkbox)
        side.addWidget(actions_box)

        data_box = QGroupBox("Навчальні приклади")
        data_layout = QFormLayout(data_box)
        self.gesture_combo = QComboBox()
        self.gesture_combo.addItems(GESTURES)
        self.sample_count = QLabel("0 прикладів")
        capture_button = QPushButton("Зберегти поточний жест")
        capture_button.clicked.connect(lambda: self.capture_sample())

        delayed_capture_btn = QPushButton("Записати через 3 сек (Пробіл)")
        delayed_capture_btn.clicked.connect(self.start_delayed_capture)

        clear_button = QPushButton("Очистити старі приклади")
        clear_button.clicked.connect(self.clear_samples)
        load_dataset_button = QPushButton("Завантажити набір даних")
        load_dataset_button.clicked.connect(self.load_dataset)
        data_layout.addRow("Назва жесту:", self.gesture_combo)
        data_layout.addRow("Накопичено:", self.sample_count)
        data_layout.addRow(capture_button)
        data_layout.addRow(delayed_capture_btn)
        data_layout.addRow(clear_button)
        data_layout.addRow(load_dataset_button)
        side.addWidget(data_box)

        learning_box = QGroupBox("Класифікація і кластеризація")
        learning_layout = QVBoxLayout(learning_box)
        self.epochs = QSpinBox()
        self.epochs.setRange(50, 1000)
        self.epochs.setValue(200)
        self.cluster_count = QSpinBox()
        self.cluster_count.setRange(2, 20)
        self.cluster_count.setValue(3)
        learning_layout.addWidget(QLabel("Ітерації MLP:"))
        learning_layout.addWidget(self.epochs)
        learning_layout.addWidget(QLabel("Кількість кластерів K:"))
        learning_layout.addWidget(self.cluster_count)
        train_button = QPushButton("Навчити класифікатор")
        train_button.clicked.connect(self.train_classifier)
        cluster_button = QPushButton("Згрупувати пози KMeans")
        cluster_button.clicked.connect(self.open_clustering_window)
        learning_layout.addWidget(train_button)
        learning_layout.addWidget(cluster_button)
        side.addWidget(learning_box)

        model_box = QGroupBox("Модель")
        model_layout = QVBoxLayout(model_box)
        save_button = QPushButton("Зберегти модель")
        save_button.clicked.connect(self.save_model)
        load_button = QPushButton("Завантажити модель")
        load_button.clicked.connect(self.load_model)
        archive_button = QPushButton("Архівувати модель")
        archive_button.clicked.connect(self.archive_model)
        restore_button = QPushButton("Розархівувати модель")
        restore_button.clicked.connect(self.restore_model)
        report_button = QPushButton("Зберегти звіт")
        report_button.clicked.connect(self.save_report)
        model_layout.addWidget(save_button)
        model_layout.addWidget(load_button)
        model_layout.addWidget(archive_button)
        model_layout.addWidget(restore_button)
        model_layout.addWidget(report_button)
        side.addWidget(model_box)
        side.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side_widget)
        scroll.setMinimumWidth(320)

        top.addWidget(scroll, 1)
        layout.addLayout(top)

        self.status_label = QLabel("Статус: готово")
        self.result_label = QLabel("Розпізнаний жест: --")
        self.result_label.setStyleSheet("font-size:22px; font-weight:bold; color:#0e7772;")
        layout.addWidget(self.result_label)
        layout.addWidget(self.status_label)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        layout.addWidget(self.log)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.start_delayed_capture()
        else:
            super().keyPressEvent(event)

    def start_delayed_capture(self):
        if self.is_counting_down:
            return
        self.countdown_timer.stop()
        self.is_counting_down = True
        self._feature_buffer = []
        self.countdown_val = 3
        self.status_label.setText(f"Приготуйся! Запис через {self.countdown_val} сек...")
        self.countdown_timer.start()

    def _handle_countdown(self):
        self.countdown_val -= 1
        if self.countdown_val > 0:
            self.status_label.setText(f"Приготуйся! Запис через {self.countdown_val} сек...")
        else:
            self.countdown_timer.stop()
            self.is_counting_down = False
            if self._feature_buffer:
                averaged = np.mean(np.stack(self._feature_buffer), axis=0).astype(np.float32)
                self.capture_sample(averaged)
            else:
                self.capture_sample()
            self._feature_buffer = []

    def start_camera(self):
        if self.camera is not None:
            return
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None
            QMessageBox.critical(self, "Камера", "Не вдалося відкрити камеру.")
            return
        self.timer.start(30)
        self.status_label.setText("Статус: камера працює")

    def stop_camera(self):
        self.timer.stop()
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.video_label.setText("Камера вимкнена")
        self.status_label.setText("Статус: камера вимкнена")

    def extract_features(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.hand_landmarker.detect(image)
        if not result.hand_landmarks:
            return None, result
        return result.hand_landmarks, result

    @staticmethod
    def _relative_flat(landmarks, origin):
        values = []
        for landmark in landmarks:
            values.extend([landmark.x - origin.x, landmark.y - origin.y, landmark.z - origin.z])
        return np.array(values, dtype=np.float32)

    def combined_features(self, hands_landmarks, result):
        left_landmarks = None
        right_landmarks = None

        for index, landmarks in enumerate(hands_landmarks or []):
            if index >= len(result.handedness):
                continue
            raw_side = result.handedness[index][0].category_name
            actual_side = "Right" if raw_side == "Left" else "Left"
            if actual_side == "Left":
                left_landmarks = landmarks
            else:
                right_landmarks = landmarks

        if left_landmarks is None and right_landmarks is None:
            return None

        wrists = []
        if left_landmarks is not None:
            wrists.append(left_landmarks[0])
        if right_landmarks is not None:
            wrists.append(right_landmarks[0])

        origin = _SimplePoint(
            x=sum(w.x for w in wrists) / len(wrists),
            y=sum(w.y for w in wrists) / len(wrists),
            z=sum(w.z for w in wrists) / len(wrists),
        )

        all_points = list(left_landmarks or []) + list(right_landmarks or [])
        rel_all = self._relative_flat(all_points, origin).reshape(-1, 3)
        scale = max(np.linalg.norm(rel_all, axis=1).max(), 1e-6)

        left = np.zeros(FEATURES_PER_HAND, dtype=np.float32)
        right = np.zeros(FEATURES_PER_HAND, dtype=np.float32)
        if left_landmarks is not None:
            left = self._relative_flat(left_landmarks, origin) / scale
        if right_landmarks is not None:
            right = self._relative_flat(right_landmarks, origin) / scale

        return np.concatenate([left, right])

    @staticmethod
    def _select_hand_landmarks(result, desired_side):
        if not result or not result.hand_landmarks:
            return None
        for index, landmarks in enumerate(result.hand_landmarks):
            if index >= len(result.handedness):
                continue
            raw_side = result.handedness[index][0].category_name
            actual_side = "Right" if raw_side == "Left" else "Left"
            if actual_side == desired_side:
                return landmarks
        return None

    @staticmethod
    def _map_with_margin(value, margin=CURSOR_MARGIN):
        span = max(1.0 - 2 * margin, 1e-6)
        mapped = (value - margin) / span
        return min(1.0, max(0.0, mapped))

    def update_frame(self):
        if self.camera is None:
            return
        ok, frame = self.camera.read()
        if not ok:
            return

        hands_landmarks, result = self.extract_features(frame)
        self.current_features = (
            self.combined_features(hands_landmarks, result) if hands_landmarks else None
        )

        if self.is_counting_down and self.current_features is not None:
            self._feature_buffer.append(self.current_features.copy())

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                for landmark in hand_landmarks:
                    point = (
                        int(landmark.x * frame.shape[1]),
                        int(landmark.y * frame.shape[0]),
                    )
                    cv2.circle(frame, point, 5, (0, 220, 170), -1)

        frame = cv2.flip(frame, 1)

        if self.current_features is not None and self.model is not None:
            prediction_code = self.model.predict(self.current_features.reshape(1, -1))[0]
            prediction = self.encoder.inverse_transform([prediction_code])[0]
            probabilities = self.model.predict_proba(self.current_features.reshape(1, -1))[0]
            confidence = float(np.max(probabilities)) * 100
            self.result_label.setText(f"Розпізнано: {prediction} ({confidence:.1f}%)")

            if self.enable_actions_checkbox.isChecked():
                g1_name = "Ліва рука: Палець вгору"
                g2_name = "Ліва рука: Лодонь"
                g3_name = "Ліва рука: Кулак"
                g4_name = "Права рука: Серце"
                g_right_palm = "Права рука: Лодонь"

                if prediction == g_right_palm:
                    self.gesture1_start_time = None
                    self.gesture2_start_time = None
                    self.gesture3_start_time = None
                    self.gesture4_start_time = None
                    if self.right_palm_start_time is None:
                        self.right_palm_start_time = time.time()
                    elapsed = time.time() - self.right_palm_start_time
                    required_toggle_time = 3.0

                    if elapsed >= required_toggle_time:
                        self.mouse_control_active = not self.mouse_control_active
                        status = "ВКЛ 🖱️" if self.mouse_control_active else "ВИКЛ 🛑"
                        self.log.append(f"Права рука (3 сек) -> Режим мишки: {status}")
                        self.status_label.setText(f"✋ Перемикання мишки: {status}")
                        self.right_palm_start_time = None
                        self.cursor_filter_x = OneEuroFilter(mincutoff=1.2, beta=0.6, dcutoff=1.0)
                        self.cursor_filter_y = OneEuroFilter(mincutoff=1.2, beta=0.6, dcutoff=1.0)
                    else:
                        rem = required_toggle_time - elapsed
                        action_str = "ВИМКНЕННЯ" if self.mouse_control_active else "ВВІМКНЕННЯ"
                        self.status_label.setText(
                            f"✋ Утримайте праву долоню {rem:.1f} сек для {action_str} мишки..."
                        )
                elif not self.mouse_control_active:
                    self.right_palm_start_time = None
                    if prediction == g1_name:
                        self.gesture2_start_time = None
                        self.gesture3_start_time = None
                        self.gesture4_start_time = None
                        if self.gesture1_start_time is None:
                            self.gesture1_start_time = time.time()
                        elapsed = time.time() - self.gesture1_start_time
                        if elapsed >= self.required_hold_duration:
                            pyautogui.press("volumeup")
                            self.log.append("Утримано 1 сек -> Гучність +")
                            self.gesture1_start_time = time.time()
                        else:
                            rem = self.required_hold_duration - elapsed
                            self.status_label.setText(f"Гучність + через {rem:.1f} сек...")
                    elif prediction == g2_name:
                        self.gesture1_start_time = None
                        self.gesture3_start_time = None
                        self.gesture4_start_time = None
                        if self.gesture2_start_time is None:
                            self.gesture2_start_time = time.time()
                        elapsed = time.time() - self.gesture2_start_time
                        if elapsed >= self.required_hold_duration:
                            os.system("start tg://")
                            self.log.append("Утримано 1 сек -> Телеграм")
                            self.gesture2_start_time = time.time()
                        else:
                            rem = self.required_hold_duration - elapsed
                            self.status_label.setText(f"Відкриття Телеграм через {rem:.1f} сек...")
                    elif prediction == g3_name:
                        self.gesture1_start_time = None
                        self.gesture2_start_time = None
                        self.gesture4_start_time = None
                        if self.gesture3_start_time is None:
                            self.gesture3_start_time = time.time()
                        elapsed = time.time() - self.gesture3_start_time
                        if elapsed >= self.required_hold_duration:
                            pyautogui.press("space")
                            self.log.append("Утримано 1 сек -> Пауза / Плей")
                            self.gesture3_start_time = time.time()
                        else:
                            rem = self.required_hold_duration - elapsed
                            self.status_label.setText(f"Пауза через {rem:.1f} сек...")
                    elif prediction == g4_name:
                        self.gesture1_start_time = None
                        self.gesture2_start_time = None
                        self.gesture3_start_time = None
                        if self.gesture4_start_time is None:
                            self.gesture4_start_time = time.time()
                        elapsed = time.time() - self.gesture4_start_time
                        if elapsed >= self.required_hold_duration:
                            h, w, _ = frame.shape
                            draw_vector_heart(frame, center_x=w // 2, center_y=h // 2, scale=8)
                            self.log.append("Показано векторне ❤️")
                            self.status_label.setText("❤️ Векторне серце активовано!")
                        else:
                            rem = self.required_hold_duration - elapsed
                            self.status_label.setText(f"Серце через {rem:.1f} сек...")
                    else:
                        self.gesture1_start_time = None
                        self.gesture2_start_time = None
                        self.gesture3_start_time = None
                        self.gesture4_start_time = None
                else:
                    self.right_palm_start_time = None
            else:
                self.right_palm_start_time = None
                self.status_label.setText("Режим навчання: гарячі дії вимкнено")
        elif self.current_features is None:
            self.result_label.setText("Розпізнаний жест: руку не знайдено")
            self.right_palm_start_time = None

        if self.mouse_control_active:
            current_landmarks = self._select_hand_landmarks(result, "Right")
            if current_landmarks:
                index_tip = current_landmarks[8]
                screen_w, screen_h = pyautogui.size()
                mapped_x = self._map_with_margin(index_tip.x)
                mapped_y = self._map_with_margin(index_tip.y)
                raw_target_x = (1 - mapped_x) * (screen_w - 1)
                raw_target_y = mapped_y * (screen_h - 1)
                target_x = int(self.cursor_filter_x.filter(raw_target_x))
                target_y = int(self.cursor_filter_y.filter(raw_target_y))
                pyautogui.moveTo(target_x, target_y, _pause=False)
                self.status_label.setText(
                    "🖱️ Керування мишкою активне (права рука) — "
                    "покажіть праву долоню на 3 сек для виходу"
                )
            else:
                self.status_label.setText(
                    "🖱️ Мишка активна: покажіть ПРАВУ руку в кадр для керування курсором"
                )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(
            frame.data,
            frame.shape[1],
            frame.shape[0],
            frame.strides[0],
            QImage.Format.Format_RGB888,
        )
        self.video_label.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def capture_sample(self, features=None):
        feats = features if features is not None else self.current_features
        if feats is None:
            QMessageBox.warning(self, "Приклад", "Покажи руку в камеру.")
            return
        self.samples.append(feats.copy())
        self.labels.append(self.gesture_combo.currentText())
        self.save_samples()
        self.update_sample_count()
        self.log.append(f"Збережено приклад: {self.labels[-1]}")
        self.status_label.setText("Статус: приклад успішно збережено")

    def train_classifier(self):
        if len(set(self.labels)) < 2 or len(self.samples) < 10:
            QMessageBox.warning(self, "Навчання", "Потрібно щонайменше 10 прикладів і 2 різні жести.")
            return

        counts = {}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        thin_classes = [f"{name} ({count})" for name, count in counts.items() if count < MIN_SAMPLES_PER_CLASS]
        if thin_classes:
            proceed = QMessageBox.question(
                self,
                "Мало прикладів",
                "Замало прикладів (< {}) для класів:\n{}\n\nНавчати все одно?".format(
                    MIN_SAMPLES_PER_CLASS, "\n".join(thin_classes)
                ),
                QMessageBox.Yes | QMessageBox.No,
            )
            if proceed != QMessageBox.Yes:
                return

        x = np.array(self.samples)
        y = self.encoder.fit_transform(self.labels)

        try:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.25, random_state=42, stratify=y
            )
        except ValueError:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.25, random_state=42
            )

        self.model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=self.epochs.value(),
                early_stopping=True,
                random_state=42,
            ),
        )
        self.model.fit(x_train, y_train)
        accuracy = accuracy_score(y_test, self.model.predict(x_test))

        try:
            cv_scores = cross_val_score(self.model, x, y, cv=min(5, min(counts.values())))
            cv_text = f", CV: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})"
        except ValueError:
            cv_text = ""

        self.save_model_file()
        self.log.append(f"Класифікатор навчено. Точність тесту: {accuracy:.2%}{cv_text}")
        self.status_label.setText(f"Статус: модель навчена, accuracy {accuracy:.2%}")

    def open_clustering_window(self):
        if not hasattr(self, "clustering_window") or self.clustering_window is None:
            self.clustering_window = ClusteringWindow(self)
        self.clustering_window.show()
        self.clustering_window.raise_()
        self.clustering_window.activateWindow()

    def restore_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Розархівувати модель", str(ARCHIVES_DIR), "Архіви (*.zip)"
        )
        if not path:
            return
        try:
            with zipfile.ZipFile(path) as archive:
                model_names = [name for name in archive.namelist() if name.endswith(".joblib")]
                if not model_names:
                    raise ValueError("В архіві не знайдено модель .joblib")
                archive.extract(model_names[0], MODELS_DIR)
                extracted = MODELS_DIR / model_names[0]
                shutil.copyfile(extracted, MODEL_PATH)
                if extracted != MODEL_PATH and extracted.exists():
                    extracted.unlink()
            self.load_model_file(MODEL_PATH)
            self.log.append("Модель розархівовано та завантажено.")
        except (zipfile.BadZipFile, OSError, ValueError) as error:
            QMessageBox.critical(self, "Розархівація", str(error))

    def save_report(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Зберегти звіт", str(RESULTS_DIR / "gesture_report.txt"), "Text (*.txt)"
        )
        if not path:
            return
        report = [
            "Лабораторна робота №3: класифікація жестів",
            f"Класи: {', '.join(GESTURES)}",
            f"Навчальних прикладів: {len(self.samples)}",
            f"Розмір ознак: {FEATURES_PER_HAND * 2}",
            f"Модель навчена: {'так' if self.model is not None else 'ні'}",
            "Кластеризація: KMeans, результат зберігається у gesture_clusters.npz",
        ]
        Path(path).write_text("\n".join(report), encoding="utf-8")
        self.log.append(f"Звіт збережено: {path}")

    def save_samples(self):
        np.savez(DATA_PATH, features=np.array(self.samples), labels=np.array(self.labels))

    def load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Завантажити набір жестів", str(RESULTS_DIR), "Gesture dataset (*.npz)"
        )
        if not path:
            return
        try:
            data = np.load(path, allow_pickle=True)
            features = data["features"]
            labels = data["labels"].astype(str)
            if features.ndim != 2 or features.shape[1] != FEATURES_PER_HAND * 2:
                raise ValueError("Набір має містити ознаки лівої та правої руки.")
            valid_rows = np.isin(labels, GESTURES)
            if not np.any(valid_rows):
                raise ValueError("У наборі немає відомих класів жестів.")
            self.samples = list(features[valid_rows])
            self.labels = list(labels[valid_rows])
            self.save_samples()
            self.update_sample_count()
            self.log.append(f"Набір даних завантажено: {path}")
        except (OSError, KeyError, ValueError) as error:
            QMessageBox.critical(self, "Набір даних", str(error))

    def load_samples(self):
        if DATA_PATH.exists():
            data = np.load(DATA_PATH, allow_pickle=True)
            loaded_labels = data["labels"].astype(str)
            loaded_features = data["features"]
            valid_rows = (
                np.isin(loaded_labels, GESTURES)
                & (loaded_features.shape[1] == FEATURES_PER_HAND * 2)
            )
            self.samples = list(loaded_features[valid_rows])
            self.labels = list(loaded_labels[valid_rows])
            if len(self.samples) != len(loaded_labels):
                self.log.append("Старі приклади іншого формату пропущено. Очистіть дані перед новим записом.")
            self.update_sample_count()

    def clear_samples(self):
        answer = QMessageBox.question(
            self,
            "Очистити дані",
            "Видалити старі приклади та навчену модель?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.samples = []
        self.labels = []
        self.model = None
        self.encoder = LabelEncoder()
        if DATA_PATH.exists():
            DATA_PATH.unlink()
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        self.update_sample_count()
        self.result_label.setText("Розпізнаний жест: --")
        self.log.append("Старі приклади та модель очищено.")

    def update_sample_count(self):
        self.sample_count.setText(f"{len(self.samples)} прикладів")

    def save_model_file(self):
        joblib.dump({"model": self.model, "encoder": self.encoder}, MODEL_PATH)

    def save_model(self):
        if self.model is None:
            QMessageBox.warning(self, "Модель", "Спочатку навчіть класифікатор.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Зберегти модель", str(MODELS_DIR / "gesture_classifier.joblib"), "Model (*.joblib)")
        if path:
            joblib.dump({"model": self.model, "encoder": self.encoder}, path)
            self.log.append(f"Модель збережено: {path}")

    def load_model_if_exists(self):
        if MODEL_PATH.exists():
            self.load_model_file(MODEL_PATH)

    def load_model_file(self, path):
        bundle = joblib.load(path)
        loaded_encoder = bundle["encoder"]
        model = bundle["model"]
        n_features = getattr(model, "n_features_in_", None)
        if n_features is None and hasattr(model, "steps"):
            n_features = getattr(model.steps[-1][1], "n_features_in_", FEATURES_PER_HAND * 2)
        if n_features not in (None, FEATURES_PER_HAND * 2):
            self.log.append("Стара модель має формат однієї руки і не завантажена.")
            return
        if not set(loaded_encoder.classes_).issubset(set(GESTURES)):
            self.log.append("Стара модель має інші класи жестів і не завантажена.")
            return
        self.model = model
        self.encoder = loaded_encoder
        self.log.append(f"Модель завантажено: {Path(path).name}")

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Завантажити модель", str(MODELS_DIR), "Model (*.joblib)")
        if path:
            try:
                self.load_model_file(path)
            except Exception as error:
                QMessageBox.critical(self, "Модель", str(error))

    def archive_model(self):
        if not MODEL_PATH.exists():
            QMessageBox.warning(self, "Архів", "Спочатку навчіть і збережіть модель.")
            return
        archive_path = ARCHIVES_DIR / "gesture_classifier.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(MODEL_PATH, MODEL_PATH.name)
        self.log.append(f"Архів створено: {archive_path}")

    def closeEvent(self, event):
        self.stop_camera()
        self.countdown_timer.stop()
        self.hand_landmarker.close()
        event.accept()


class ClusteringWindow(QDialog):
    def __init__(self, classification_window):
        super().__init__(classification_window)
        self.classification_window = classification_window
        self.setWindowTitle("Lab 3 | Кластеризація поз руки")
        self.resize(850, 650)
        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Кількість кластерів K:"))
        self.cluster_count = QSpinBox()
        self.cluster_count.setRange(2, 20)
        self.cluster_count.setValue(3)
        controls.addWidget(self.cluster_count)
        run_button = QPushButton("Запустити кластеризацію")
        run_button.clicked.connect(self.run_clustering)
        save_button = QPushButton("Зберегти результат")
        save_button.clicked.connect(self.save_result)
        controls.addWidget(run_button)
        controls.addWidget(save_button)
        controls.addStretch()
        layout.addLayout(controls)

        self.figure = Figure(figsize=(8, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

    def run_clustering(self):
        samples = self.classification_window.samples
        if len(samples) < self.cluster_count.value():
            QMessageBox.warning(self, "Кластеризація", "Недостатньо навчальних прикладів.")
            return

        values = np.array(samples)
        model = KMeans(n_clusters=self.cluster_count.value(), n_init=10, random_state=42)
        clusters = model.fit_predict(values)
        centers = model.cluster_centers_

        np.savez(RESULTS_DIR / "gesture_clusters.npz", features=values, clusters=clusters, centers=centers)

        pca = PCA(n_components=2, random_state=42)
        reduced_values = pca.fit_transform(values)

        axis = self.figure.axes[0] if self.figure.axes else self.figure.add_subplot(111)
        axis.clear()
        axis.scatter(reduced_values[:, 0], reduced_values[:, 1], c=clusters, cmap="viridis", s=35, alpha=0.85)
        axis.set_title("2D Проекція кластерів поз руки (PCA)")
        axis.set_xlabel("Головна компонента 1")
        axis.set_ylabel("Головна компонента 2")
        axis.grid(alpha=0.25)

        self.canvas.draw_idle()

        distribution = np.bincount(clusters, minlength=self.cluster_count.value()).tolist()
        self.output.setPlainText(
            f"KMeans завершено\nКластерів: {self.cluster_count.value()}\n"
            f"Розподіл об'єктів: {distribution}\n"
            f"Пояснена дисперсія PCA: {sum(pca.explained_variance_ratio_):.2%}\n"
            f"Результат збережено: {RESULTS_DIR / 'gesture_clusters.npz'}"
        )

    def save_result(self):
        source = RESULTS_DIR / "gesture_clusters.npz"
        if not source.exists():
            QMessageBox.warning(self, "Результат", "Спочатку запустіть кластеризацію.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Зберегти кластеризацію", str(RESULTS_DIR / "gesture_clusters.npz"), "NumPy (*.npz)"
        )
        if path:
            shutil.copyfile(source, path)


# =========================================================
# 🚀 ГОЛОВНЕ СТАРТОВЕ ВІКНО КЕРУВАННЯ
# =========================================================
class StartWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lab 8 | Data Mining & Text Recognition (CNN)")
        self.resize(540, 500)
        self.classification_window = None
        self.fuzzy_window = None
        self.genetic_window = None
        self.func_window = None
        self.tree_window = None
        self.text_cnn_window = None

        layout = QVBoxLayout()
        title = QLabel("Інформаційна система Data Mining\nз модулями CNN та розпізнаванням тексту")
        title.setStyleSheet("font-size:20px; font-weight:bold; color:#1a365d;")
        layout.addWidget(title)

        classification_button = QPushButton("📷 Вікно розпізнавання жестів & мишки (Лаб 1-3)")
        classification_button.clicked.connect(self.open_classification)

        clustering_button = QPushButton("📊 Вікно кластеризації поз KMeans (Лаб 3)")
        clustering_button.clicked.connect(self.open_clustering)

        fuzzy_button = QPushButton("🌀 Вікно нечіткого аналізу Fuzzy Logic (Лаб 4)")
        fuzzy_button.clicked.connect(self.open_fuzzy)

        genetic_button = QPushButton("🧬 Вікно Генетичних Алгоритмів (Лаб 5)")
        genetic_button.clicked.connect(self.open_genetic)

        func_button = QPushButton("📈 Вікно Функціональних Залежностей / Регресії (Лаб 6)")
        func_button.clicked.connect(self.open_func)

        tree_button = QPushButton("🌳 Вікно Дерев Рішень / Decision Forests (Лаб 7)")
        tree_button.clicked.connect(self.open_tree)

        text_cnn_button = QPushButton("📝 Вікно розпізнавання тексту та згорткових мереж / CNN (Лаб 8)")
        text_cnn_button.setStyleSheet("font-weight:bold; background-color:#e2e8f0; padding:8px;")
        text_cnn_button.clicked.connect(self.open_text_cnn)

        layout.addWidget(classification_button)
        layout.addWidget(clustering_button)
        layout.addWidget(fuzzy_button)
        layout.addWidget(genetic_button)
        layout.addWidget(func_button)
        layout.addWidget(tree_button)
        layout.addWidget(text_cnn_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_classification(self):
        if self.classification_window is None:
            self.classification_window = GestureApp()
        self.classification_window.show()
        self.classification_window.raise_()
        self.classification_window.activateWindow()

    def open_clustering(self):
        self.open_classification()
        self.classification_window.open_clustering_window()

    def open_fuzzy(self):
        if self.fuzzy_window is None:
            self.fuzzy_window = FuzzyAnalysisWindow(self)
        self.fuzzy_window.show()
        self.fuzzy_window.raise_()
        self.fuzzy_window.activateWindow()

    def open_genetic(self):
        if self.genetic_window is None:
            self.genetic_window = GeneticAnalysisWindow(self)
        self.genetic_window.show()
        self.genetic_window.raise_()
        self.genetic_window.activateWindow()

    def open_func(self):
        if self.func_window is None:
            self.func_window = FunctionalDependencyWindow(self)
        self.func_window.show()
        self.func_window.raise_()
        self.func_window.activateWindow()

    def open_tree(self):
        if self.tree_window is None:
            self.tree_window = DecisionTreeWindow(self)
        self.tree_window.show()
        self.tree_window.raise_()
        self.tree_window.activateWindow()

    def open_text_cnn(self):
        if self.text_cnn_window is None:
            self.text_cnn_window = TextCnnWindow(self)
        self.text_cnn_window.show()
        self.text_cnn_window.raise_()
        self.text_cnn_window.activateWindow()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StartWindow()
    window.show()
    sys.exit(app.exec())