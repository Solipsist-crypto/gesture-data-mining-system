# 🖐️ Intelligent Data Mining & Gesture Control System (PySide6 + MediaPipe)

Інформаційна система для інтелектуального аналізу даних, розпізнавання жестів рук у реальному часі та автоматизації керування ПК. Проєкт об'єднує класичні алгоритми машинного навчання (MLP, KMeans), методи нечіткого аналізу (Fuzzy Logic) та модуль комп'ютерного зору.

---

## 🚀 Основний функціонал

* **📷 Розпізнавання жестів у реальному часі:** Трекінг 21 ключевої точки для обох рук через `MediaPipe Tasks API`.
* **🖱️ Жестове керування мишкою:**
  * Позиціонування курсора за кінчиком вказівного пальця правої руки.
  * Адаптивна фільтрація шуму та тремтіння за допомогою фільтра `OneEuroFilter`.
  * **Правий клік (ПКМ):** Швидкий щипок великого та вказівного пальців ($\text{Thumb} + \text{Index}$).
  * **Режим активації:** Утримання правої долоні протягом 3 секунд.
* **🤖 Класифікація та кластеризація (Data Mining):**
  * **MLPClassifier:** Багатошаровий персептрон для розпізнавання 7 типів жестів.
  * **KMeans:** Кластеризація поз рук із візуалізацією 2D-проекцій через PCA.
* **🌀 Модуль нечіткого аналізу (Fuzzy Logic):**
  * **Fuzzy C-Means (FCM):** Нечітка кластеризація з розрахунком матриці належності $u_{ij} \in [0, 1]$.
  * **Fuzzy Classifier:** Гаусова нечітка класифікація ($\sigma$), стійка до шуму та суперечливих даних.
  * Підтримка завантаження наборів даних у форматах `.xlsx`, `.csv` та `.npz`.
* **⚡ Гарячі дії та автоматизація (pyautogui):**
  * Регулювання гучності, відкриття Telegram, пауза/відтворення та малювання векторного серця.

---

## 🛠️ Технологічний стек

* **Python 3.10+**
* **GUI:** PySide6 (Qt for Python)
* **Computer Vision:** OpenCV, MediaPipe Tasks (`hand_landmarker.task`)
* **Machine Learning & Math:** scikit-learn, NumPy, Pandas, SciPy
* **Visualization:** Matplotlib
* **OS Automation:** PyAutoGUI

---

## 📦 Встановлення та запуск

### 1. Клонування репозиторію та налаштування середовища
```bash
git clone [https://github.com/your-username/gesture-data-mining.git](https://github.com/your-username/gesture-data-mining.git)
cd gesture-data-mining

# Створення та активація віртуального середовища
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate