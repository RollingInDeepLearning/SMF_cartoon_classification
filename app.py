import streamlit as st
import joblib
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


MAPPING =  {'Cry babies magic tears': 0,
 'Enchantimals (Эншантималс)': 1,
 'My little pony': 2,
 'none': 3,
 'Акуленок': 4,
 'Барбоскины': 5,
 'Бременские музыканты': 6,
 'Буба': 7,
 'Бэтмен': 8,
 'Вспыш': 9,
 'Говорящий Том': 10,
 'Губка Боб': 11,
 'Енотки': 12,
 'ЖилаБыла Царевна': 13,
 'Зебра в клеточку': 14,
 'Котик Мормотик': 15,
 'Кошечки собачки': 16,
 'Крутиксы': 17,
 'Кукутики': 18,
 'Лунтик': 19,
 'Малышарики': 20,
 'Маша и медведь': 21,
 'Мини-мишки': 22,
 'Ну_погоди каникулы': 23,
 'Оранжевая корова': 24,
 'Паровозики Чаттингтон': 25,
 'Пороро': 26,
 'Приключения Пети и Волка': 27,
 'Простоквашино': 28,
 'Свинка Пеппа': 29,
 'Симпсоны': 30,
 'Синий трактор': 31,
 'Смешарики': 32,
 'Сумка': 33,
 'Трансформеры': 34,
 'Фиксики': 35,
 'Финник': 36,
 'Царевны': 37,
 'Цветняшки': 38,
 'Чебурашка': 39,
 'Черепашки Ниндзя': 40,
 'Чик-Чирикино': 41,
 'Чуддики': 42,
 'Чучело-Мяучело': 43,
 'Щенячий патруль': 44}

def get_data_from_url(video_url):
    driver = webdriver.Chrome()
    try:
        driver.get(video_url)
        video_title = driver.title.replace(' - YouTube', '')
        channel_link_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'a.yt-simple-endpoint.style-scope.yt-formatted-string'))
        )
        channel_href = channel_link_element.get_attribute('href')
        driver.get(channel_href)
        canonical_link = driver.find_element(By.CSS_SELECTOR, 'link[rel="canonical"]')
        canonical_href = canonical_link.get_attribute('href')
        channel_id = canonical_href.split('/')[-1]
        st.info(f"Название видео: {video_title}")
        return f'{video_title} {channel_id}'
    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
        return None
    finally:
        driver.quit()

def download_model(url, save_path="model.joblib"):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        st.success(f"Модель успешно скачана и сохранена в {save_path}.")
    else:
        st.error(f"Ошибка при скачивании модели: {response.status_code}")

def load_model_and_vectorizer(model_link, vectorizer_path):
    model_path = "model.joblib"
    if not joblib.os.path.exists(model_path):
        st.info("Первый запуск модели, ожидайте окончания загрузки...")
        download_model(model_link)
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def classify_data(model, vectorizer, text):
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    return prediction, probabilities

# Основная часть приложения
st.title("Классификация видео по YouTube")

model_link = "https://huggingface.co/Rollingindeeplearning/rf_cartoon_classificator/resolve/main/best_model.joblib"
vectorizer_path = 'vectorizer.joblib'

link_url = st.text_input("Введите ссылку на видео:")
if st.button("Классифицировать"):
    data = get_data_from_url(link_url)
    if data:
        model, vectorizer = load_model_and_vectorizer(model_link, vectorizer_path)
        result_num, class_probabilities = classify_data(model, vectorizer, data)

        reverse_mapping = {v: k for k, v in MAPPING.items()}
        
        if result_num in reverse_mapping:
            result = reverse_mapping[result_num]
            st.write("Результат классификации:", result)
            st.write("Вероятность принадлежности к классу:", round(max(class_probabilities), 4))