import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random
import json

# Загрузка необходимых ресурсов для обработки текста
nltk.download('punkt')

# Инициализация стеммера для обработки слов
stemmer = LancasterStemmer()

# Загрузка данных из файла intents.json
with open('intents.json') as file:
	data = json.load(file)

# Инициализация списков для хранения обработанных слов, меток и документов
all_words = []
all_labels = []
docs = []


def process_intents():
	"""
	Обработка данных из файла intents.json и подготовка их для обучения нейронной сети.

	Returns:
	np.array: Массив обучающих данных.
	np.array: Массив выходных данных.
	"""
	global all_words
	global all_labels
	global docs

	for intent in data['intents']:
		for pattern in intent['patterns']:
			# Токенизация и стемминг слов
			words = [stemmer.stem(w) for w in nltk.word_tokenize(pattern) if w != "?"]
			all_words.extend(words)
			docs.append((words, intent["tag"]))

			# Добавление метки в список меток, если ее там нет
			if intent['tag'] not in all_labels:
				all_labels.append(intent["tag"])

	# Удаление дубликатов и сортировка слов и меток
	all_words = sorted(list(set(all_words)))
	all_labels = sorted(all_labels)

	# Инициализация списков для обучающих и выходных данных
	training = []
	output = []

	# Создание пустого вектора для выходных данных
	output_empty = [0] * len(all_labels)

	# Создание векторов "мешка слов" и выходных данных для каждого документа
	for words, tag in docs:
		bag = [1 if w in words else 0 for w in all_words]
		output_row = output_empty[:]
		output_row[all_labels.index(tag)] = 1

		training.append(bag)
		output.append(output_row)

	return np.array(training), np.array(output)


def build_model(input_size):
	"""
	Построение модели нейронной сети.

	Args:
	input_size (int): Размер входного вектора.

	Returns:
	Sequential: Построенная модель.
	"""
	model = Sequential()
	model.add(Dense(8, input_dim = input_size, activation = 'relu'))
	model.add(Dense(8, activation = 'relu'))
	model.add(Dense(len(all_labels), activation = 'softmax'))
	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	return model


def create_bag_of_words(s, all_words):
	"""
	Создание вектора "мешка слов" для входной строки.

	Args:
	s (str): Входная строка.
	all_words (list): Список всех уникальных слов.

	Returns:
	np.array: Вектор "мешка слов".
	"""
	bag = [1 if se in nltk.word_tokenize(s) else 0 for se in all_words]
	return np.array(bag)


def chat(bot_model):
	"""
	Взаимодействие с чат-ботом.

	Args:
	bot_model: Обученная модель чат-бота.
	"""
	print("The bot is prepared for chatting")
	while True:
		user_input = input("\nYou: ")
		if user_input.lower() == 'quit':
			break

		# Предсказание ответа на вопрос пользователя
		results = bot_model.predict(np.array([create_bag_of_words(user_input, all_words)]))
		results_index = np.argmax(results)
		tag = all_labels[results_index]

		# Вывод ответа бота на экран
		for tg in data['intents']:
			if tg['tag'] == tag:
				responses = tg['responses']
				print("Bot: " + random.choice(responses))


# Обработка данных и подготовка к обучению
training_data, output_data = process_intents()

# Построение и обучение модели нейронной сети
model = build_model(len(all_words))
model.fit(training_data, output_data, epochs = 1500, batch_size = 8, verbose = 1)

# Сохранение обученной модели
model.save("model.h5")

# Запуск чат-бота
chat(model)
