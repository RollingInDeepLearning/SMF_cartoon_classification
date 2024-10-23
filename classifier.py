import joblib
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, make_scorer, f1_score



def get_data_from_url(video_url):
    # Настройка веб-драйвера (например, для Chrome)
    driver = webdriver.Chrome()
    
    try:
        driver.get(video_url)

        # Получение названия видео из тега <title>
        video_title = driver.title.replace(' - YouTube', '')
        
        # Получение ссылки на канал
        channel_link_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'a.yt-simple-endpoint.style-scope.yt-formatted-string'))
        )
        
        channel_href = channel_link_element.get_attribute('href')
        
        # Переход по ссылке на канал
        driver.get(channel_href)
        
        # Получение канонической ссылки
        canonical_link = driver.find_element(By.CSS_SELECTOR, 'link[rel="canonical"]')
        canonical_href = canonical_link.get_attribute('href')
        
        # Извлечение ID канала из канонической ссылки
        channel_id = canonical_href.split('/')[-1]  # Получаем последний элемент после '/'
        
        return f'{video_title} {channel_id}'

    except Exception as e:
        print("Ошибка:", str(e))
        return None, None

    finally:
        driver.quit()

def download_model(url, save_path = "model.joblib"):
    """
    Скачивает модель по указанному URL и сохраняет её по заданному пути.
    
    :param url: URL для скачивания модели
    :param save_path: Путь для сохранения загруженной модели
    """
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Модель успешно скачана и сохранена в {save_path}.")
    else:
        print(f"Ошибка при скачивании модели: {response.status_code}")


def load_model_and_vectorizer(model_link, vectorizer_path):
    """
    Загружает обученную модель и векторизатор из файлов.
    
    :param model_link: URL для скачивания модели
    :param vectorizer_path: Путь к файлу векторизатора
    :return: Загруженная модель и векторизатор
    """
    
    model_path = "model.joblib"
    
    # Проверяем, существует ли файл модели
    if not joblib.os.path.exists(model_path):
        download_model(model_link)   
        
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer
 
def classify_data(model, vectorizer, text):
    """
    Классифицирует данные с помощью загруженной модели.
    
    :param model: Загруженная модель
    :param vectorizer: Загруженный векторизатор
    :param text: Текст для классификации
    :return: Предсказание модели
    """
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    return prediction, probabilities

