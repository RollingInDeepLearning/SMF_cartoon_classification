import sys
from classifier import get_data_from_url, load_model_and_vectorizer, classify_data

MODEL_LINK = "https://huggingface.co/Rollingindeeplearning/rf_cartoon_classificator/resolve/main/best_model.joblib"
VECTORIZER_PATH = 'vectorizer.joblib'

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

def main(link_url):
    # Получение данных по ссылке (пока заглушка)
    data = get_data_from_url(link_url)
    
    # Загрузка модели и векторизатора

    model, vectorizer = load_model_and_vectorizer(MODEL_LINK, VECTORIZER_PATH)
    
    # Классификация данных
    result_num, class_probabilities = classify_data(model, vectorizer, data)

    reverse_mapping = {v: k for k, v in MAPPING.items()}

# Получаем название по значению
    if result_num in reverse_mapping:
        result = reverse_mapping[result_num]
    else:
        result = None
    
    # Вывод результата в консоль
    print("Результат классификации:", result)
    print("Вероятность принадлежности к классу:", round(max(class_probabilities),4))
    

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Использование: python main.py <video_url>")
        sys.exit(1)

    link_url = sys.argv[1]
    
    main(link_url)