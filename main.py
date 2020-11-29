import random
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC


from Config import BOT_CONFIG


# <--- сюда вставляем новую классификацию день 2 --->

X_texts = []
y = []
for intent, intent_data in BOT_CONFIG['intents'].items():
    for example in intent_data['examples']:
        X_texts.append(example)
        y.append(intent)

vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
X = vectorizer.fit_transform(X_texts)
clf = LinearSVC().fit(X, y) # у LinearSVC нет Proba, поэтому берем LogisticRegr для оценки вероятности
clf_proba = SVC(probability=True).fit(X, y) # поменяли метод на SVC тк регрессия давала низкую вероятноть

def get_intent(question):
    question_vector = vectorizer.transform([question])
    intent = clf.predict(question_vector)[0]
    examples = BOT_CONFIG['intents'][intent]['examples']
    for example in examples: # делаем доп проверку именно интента на совпадение вводных (скопировали из низа)
        dist = nltk.edit_distance(question, example)
        dist_percentage = dist / len(example)
        if dist_percentage < 0.4:
            return intent

# <--- сюда вставляем новую классификацию --->

def get_answer_by_intent(intent):
    if intent in BOT_CONFIG['intents']:
        phrases = BOT_CONFIG['intents'][intent]['responses']
        return random.choice(phrases)

# <--- код из дня 3 --->

with open('/Users/e.rukavishnikova.asbis/Downloads/dialogues.txt') as f:
    content = f.read()

def filter_text(text):
    text = text.lower() # убираем зависимость от регистра с помощью lower
    text = [c for c in text if c in 'абвгдеёжзийклмнопрстуфхцчшщъьэюя- '] # создаем условия «только для текста»
    text =''.join(text) # объединяем
    return text

dialogues = [dialogue_line.split('\n') for dialogue_line in content.split('\n\n')]

questions = set()
qa_dataset = []
for replicas in dialogues:
    if len(replicas) < 2:
        continue

    question, answer = replicas[:2]
    question = filter_text(question[2:])
    answer = answer[2:]

    if question and question not in questions:
        questions.add(question)
        qa_dataset.append([question, answer])

qa_by_word_dataset = {}
for question, answer in qa_dataset:
    words = question.split(' ') # делим все фразы на отедльные слова
    for word in words: # создаем подмассив только с теми парами, где есть совпадения слов с вопросом
        if word not in qa_by_word_dataset:
            qa_by_word_dataset[word] = [] #если такого слова нет, то создаем пустой датасет
        qa_by_word_dataset[word].append((question, answer))

qa_by_word_dataset_filtered = {word: qa_list
                               for word, qa_list in qa_by_word_dataset.items()
                                if len(qa_list) < 1000}


def generate_answer_by_text(text):
    text = filter_text(text)
    words = text.split(' ') # делаим запрос на отдельные слова
    qa = []
    for word in words:
       if word in qa_by_word_dataset_filtered:
           qa += qa_by_word_dataset_filtered[word] # просим добавить слово в подмассив += qa
    qa = list(set(qa))[:1000] # создаем сет чтобы убрать повторы

    results = []
    for question, answer in qa: # теперь меняем с qa_dataset на qa – подмассив, который меньше — для скорости
        dist = nltk.edit_distance(question, text)
        dist_percentage = dist / len(question)
        results.append([dist_percentage, question, answer])

    if results: # проверяем, что есть хоть 1 совпадение
        dist_percentage, question, answer = min(results, key=lambda pair: pair[0])
        if dist_percentage < 0.3:
            return answer

# <--- код из дня 3 --->

def get_failure_phrase(): # в скобках идет то, что ф-я принмиает на вход, ничего — нормальный варинат
    phrases = BOT_CONFIG['failure_phrases']
    return random.choice(phrases)

stats = [0, 0, 0] # просим бота собирать статистику обращений в переменную
# Задаем функцию, которая и будет нашим ботом.
# Это основная архитектура приложения, в основном состоящая из последовательности условий «если — то»
def bot(question):
    # NLU – neuro lingual understanding
    # 1. Understand intent of a customer
    intent = get_intent(question)

    # Get answer
    # 1. Ready-made respond
    if intent:
        answer = get_answer_by_intent(intent)
        if answer:
            stats[0] += 1
            return answer

# 2. Generate spontanious respond (context-base)
    answer = generate_answer_by_text(question)
    if answer:
        stats[1] += 1
        return answer

# 3. Если оба подхода не сработали — возвращаем заглушку
    answer = get_failure_phrase()
    stats[2] += 1
    return answer


# Задаем цикл. Условие — возвращать написанное, пока не будет написано ключевое слово exit
# question = None # Необходимо объявить пустое значение, чтобы код коректно работал

# while question not in ['exit', 'выход']:
#    question = input()
#    answer = bot(question)
#    print(answer, stats)
#    print('ready')

# Добавляем пример кода простого эхо-бота, чистим от ненужного

from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    update.message.reply_text('Ну привет')


def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Бог в помощь')


def echo(update: Update, context: CallbackContext) -> None:
    """Echo the user message.""" # вносим коррективы в ф-ю эха, чтобы бот отвечал нашими сообщениями
    answer = bot(update.message.text)
    update.message.reply_text(answer)
    print(stats)
    print('– ', update.message.text)
    print('– ', answer)
    print()

def main():
    """Start the bot."""

    updater = Updater("1412671246:AAFKNUck-bWWgSdrvB4w2SGywSLr-l8pz7w", use_context=True)

    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    updater.start_polling()
    updater.idle()
main()