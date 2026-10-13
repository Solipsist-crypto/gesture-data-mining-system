# 🤖 Data Mining Desktop Suite & Real-Time Computer Vision

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PySide6](https://img.shields.io/badge/GUI-PySide6%2FQt6-green?logo=qt)
![MediaPipe](https://img.shields.io/badge/Vision-MediaPipe-orange?logo=google)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-blue?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Data Mining Desktop Suite** — це комплексний десктопний застосунок для інтелектуального аналізу даних, комп'ютерного зору та машинного навчання, побудований на базі **PySide6**, **MediaPipe**, **OpenCV** та **Scikit-Learn**. 

Проєкт об'єднує 8 аналітичних модулів: від розпізнавання жестів рук у реальному часі та керування курсором ПК до нечіткого аналізу (Fuzzy Logic), кастомних генетичних алгоритмів, дерев рішень, аналізу тональності тексту (NLP) та згорткового аналізу зображень (CNN).

---

## 🌟 Ключові Можливості та Модулі

### 📷 1–3. Комп'ютерний Зір та Жестове Керування ПК
* **Real-time Hand Tracking:** Детекція 21 ключової точки обох рук через `MediaPipe Tasks API`.
* **Автоматизація ПК:** Модуль автоматизації дій (`pyautogui`) з адаптивною фільтрацією руху `OneEuroFilter` для плавного позиціонування курсора миші без тремтіння рук.
* **Класифікація & Кластеризація:** Навчання `MLPClassifier` для розпізнавання складних жестів та візуалізація кластерів поз через `KMeans` + `PCA`.

### 🌀 4. Нечіткий Аналіз Даних (Fuzzy Logic)
* Алгоритм нечіткої кластеризації **Fuzzy C-Means (FCM)**.
* М'яка класифікація об'єктів на основі гаусових функцій належності $u(x)$.

### 🧬 5. Еволюційні та Генетичні Алгоритми (Custom GA)
* **Zero-Dependency Implementation:** Кастомний генетичний алгоритм на чистому **NumPy** для еволюційного пошуку та оптимізації ваг моделей (задачі класифікації та мінімізації похибки MSE).

### 📈 6. Функціональні Залежності та Регресія
* Множинна лінійна регресія та Глибинні Нейронні Мережі (**DNN / MLPRegressor**) для моделювання складних нелінійних залежностей у масивах даних.

### 🌳 7. Ансамблі Дерев Рішень (Decision Forests)
* Реалізація моделей **Random Forest** та **Gradient Boosting**.
* Візуалізація та аналіз важливості ознак (Feature Importances).

### 📝 8. Аналіз Тексту (NLP) та Згорткові Мережі (CNN)
* **Sentiment Analysis:** Векторизація тексту через `TF-IDF` (символьні n-грами) та логістична регресія для аналізу тональності відгуків/повідомлень.
* **Handwritten Digit Recognition:** Інтерактивний холст для малювання з автоматичним обрізанням, центруванням (8x8) та розпізнаванням рукописних цифр ($0$–$9$) за допомогою згорткової нейромережі (CNN/MLP).

---

## 🛠️ Технологічний Стек

* **Мова програмування:** Python 3.10+
* **Інтерфейс (GUI):** PySide6 (Qt6), Matplotlib (вбудовані дашборди)
* **Комп'ютерний зір:** MediaPipe, OpenCV, PyAutoGUI
* **Машинне навчання & NLP:** Scikit-Learn, NumPy, Pandas, Joblib
* **Архітектура:** Multi-Window Qt Application, NPZ Dataset Management

---

## 📂 Структура Проєкту

```text
├── archives/               # Архіви збережених моделей (.zip)
├── models/                 # Файли навчених моделей (.joblib)
├── results/                # Датасети (.npz, .csv) та згенеровані звіти
├── hand_landmarker.task    # Модель MediaPipe для детектування рук
├── main.py                 # Головна точка входу (StartWindow)
├── README.md               # Документація проєкту
└── requirements.txt        # Перелік залежностей Python