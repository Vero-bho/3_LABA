import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



data = pd.read_csv("D:\\УЧЕБА\\ИИ\\3_LABA\\spam_or_not_spam.csv")
X_train, X_test, y_train, y_test = train_test_split(data["email"],
data["label"], test_size=0.2, random_state=42)

print("="*50)
print("АНАЛИЗ ДАННЫХ")
print("="*50)
print("Размер датасета:", data.shape)
print("Колонки в датасете:", data.columns.tolist())
print("\nПервые 5 строк:")
print(data.head())

# ПРОВЕРКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ
print(f"\n ПРОВЕРКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ:")
print(f"Пропуски в колонке 'email': {data['email'].isnull().sum()}")
print(f"Пропуски в колонке 'label': {data['label'].isnull().sum()}")
# УДАЛЕНИЕ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ
original_size = len(data)
data_clean = data.dropna(subset=['email', 'label'])
removed_count = original_size - len(data_clean)
print(f"\n ОЧИСТКА ДАННЫХ:")
print(f"Исходный размер: {original_size}")
print(f"Удалено строк с пропусками: {removed_count}")
print(f"Очищенный размер: {len(data_clean)}")
X_train, X_test, y_train, y_test = train_test_split(
data_clean['email'],
data_clean['label'],
test_size=0.2,
random_state=42,
stratify=data_clean['label'] # Сохраняем распределение классов
)
print(f"\n РАЗДЕЛЕНИЕ ДАННЫХ:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape} (спам: {y_train.sum()}, не спам: {len(y_train) - y_train.sum()})")
print(f"y_test: {y_test.shape} (спам: {y_test.sum()}, не спам: {len(y_test) - y_test.sum()})")

# Проверяем распределение меток
print(f"\n РАСПРЕДЕЛЕНИЕ МЕТОК:")
print(data_clean['label'].value_counts())
# Преобразуем текстовые данные в строковый тип
data_clean['email'] = data_clean['email'].fillna('').astype(str)


# Векторизация TF-IDF
print("\n" + "="*50)
print("ВЕКТОРИЗАЦИЯ ТЕКСТА (TF-IDF)")
print("="*50)
# Создание TF-IDF
vectorizer = TfidfVectorizer(
max_features=5000, # Ограничим до 5000 самых важных слов
stop_words='english',
lowercase=True,
strip_accents='unicode',
min_df=5, # Игнорировать слова, встречающиеся менее 5 раз
max_df=0.7, # Игнорировать слишком частые слова
ngram_range=(1, 2), # Учитывать не только слова, но и пары слов
token_pattern=r'(?u)\b[A-Za-z]{3,}\b' # Игнорировать короткие слова и символы
)
# Преобразование данных обучения и тестирования в функции TF-IDF
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)
print("Векторизация TF-IDF завершена")
print(f"Размер X_train_features: {X_train_features.shape}")
print(f"Размер X_test_features: {X_test_features.shape}")
print(f"Количество признаков (слов): {len(vectorizer.get_feature_names_out())}")
# Пример первых 10 признаков (слов)
print(f"Пример признаков: {vectorizer.get_feature_names_out()[:10]}")

# Создание и обучение модели логистической регрессии
model = LogisticRegression()
# Обучаем модель
model.fit(X_train_features, y_train)
print("Модель логистической регрессии обучена")


# Предсказания и оценка точности
y_pred = model.predict(X_test_features)
# Вычисляем точность
accuracy = accuracy_score(y_test, y_pred)
print("Точность:", accuracy)
# ОЦЕНКА МОДЕЛИ
print("\n" + "="*50)
print("ОЦЕНКА МОДЕЛИ")
print("="*50)
print(f"Точность модели: {accuracy:.4f}")
print("\n ОТЧЕТ ПО КЛАССИФИКАЦИИ:")
print(classification_report(y_test, y_pred))
# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Матрица ошибок')
plt.ylabel('Фактические значения')
plt.xlabel('Предсказанные значения')
plt.show()
input("\nНажмите Enter для завершения...")

# ВЫВОД КОНКРЕТНЫХ ПРЕДСКАЗАНИЙ
print("\n" + "="*50)
print("КОНКРЕТНЫЕ ПРЕДСКАЗАНИЯ - СПАМ ИЛИ НЕ СПАМ")
print("="*50)
# Показываем несколько примеров с предсказаниями
sample_size = min(10, len(X_test)) # Покажем первые 10 примеров
for i in range(sample_size):
    actual_label = "СПАМ" if y_test.iloc[i] == 1 else "НЕ СПАМ"
    predicted_label = "СПАМ" if y_pred[i] == 1 else "НЕ СПАМ"
    is_correct = "ВЕРНО" if y_test.iloc[i] == y_pred[i] else "ОШИБКА"
    print(f"\nПример {i+1}:")
    print(f"Текст: {X_test.iloc[i][:100]}...") # Показываем первые 100 символов
    print(f"Реальный класс: {actual_label}")
    print(f"Предсказанный класс: {predicted_label}")
    print(f"Результат: {is_correct}")
    print("-" * 60)
# ТЕСТИРОВАНИЕ НА НОВЫХ ПРИМЕРАХ
print("\n" + "="*50)
print("ТЕСТИРОВАНИЕ НА НОВЫХ ТЕКСТАХ")
print("="*50)
# Создаем тестовые примеры
test_emails = [
    "Win free money now! Click here to claim your prize!", # Должен быть спам
    "Hello, let's meet for lunch tomorrow at 12 PM.", # Должен быть не спам
    "URGENT: Your account will be suspended. Verify now!", # Должен быть спам
    "Hi John, thanks for your email. I'll get back to you soon." # Должен быть не спам
]
for i, email in enumerate(test_emails, 1):
    # Преобразуем текст в TF-IDF признаки
    email_vec = vectorizer.transform([email])
    # Делаем предсказание
    prediction = model.predict(email_vec)[0]
    # Получаем вероятности
    probability = model.predict_proba(email_vec)[0]
    result = "СПАМ" if prediction == 1 else "НЕ СПАМ"
    spam_prob = probability[1] # Вероятность что это спам
    print(f"\nТестовый пример {i}:")
    print(f"Текст: {email}")
    print(f"Предсказание: {result}")
    print(f"Вероятность спама: {spam_prob:.3f}")
if spam_prob > 0.7:
    print("ВЫСОКАЯ ВЕРОЯТНОСТЬ СПАМА!")
elif spam_prob < 0.3:
    print("ВЕРОЯТНО НЕ СПАМ")
else:
    print("НЕОПРЕДЕЛЕННО")
    print("-" * 60)
input("\nНажмите Enter для завершения...")