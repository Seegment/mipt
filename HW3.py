## Импортируем библиотеки
import pandas as pd 
#import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
dtypes = {
    'row_id': 'int32',
    'timestamp': 'int64',
    'user_id': 'int32',
    'content_id': 'int16',
    'content_type_id': 'int8',
    'task_container_id': 'int16',
    'user_answer': 'int8',
    'answered_correctly': 'int8',
    'prior_question_elapsed_time': 'float32',
    'prior_question_had_explanation': 'boolean'
}
Считываем необходимые файлы для дальшейней обработки плюс первичный анализ содержимого
# Главный Датасет
train = pd.read_csv('/Users/seegment/Desktop/Магистратура/Программирование на Python/Домашнее задание 3/train.csv', dtype=dtypes)
del dtypes
train.pop('row_id')
#вспомогательные df 
lectures = pd.read_csv('/Users/seegment/Desktop/Магистратура/Программирование на Python/Домашнее задание 3/lectures.csv')
questions = pd.read_csv('/Users/seegment/Desktop/Магистратура/Программирование на Python/Домашнее задание 3/questions.csv')
# Собираем первичную информацию о DF
train.info()
lectures.info()
questions.info()
# Делаем предварительный анализ всех таблиц (в разных блоках чтобы сохранить внешний вид таблиц)
train.head()

#row_id - Индекс строки
#timestamp - затраченное время
#user_id - Иденткфикатор пользователя 
#content_id - Код идентефикатора взаимодействия пользователя ???
#content_type_id - 0 в случае если вопрос; 1 в случае если лекция
#task_container_id - что-то вроде объединения для теста. То есть несколько вопросов могут быть в рамках одного данного id 
#user_answer - ответ пользователя (-1 в случае лекции)
#answered_correctly - правильный ответ на вопрос (-1 в случае лекции)
#prior_question_elapsed_time - 
#prior_question_had_explanation - увидел ли пользователь ответ пна предыдущий контейнер вопросов 


lectures.head()

# lecture_id - внешний ключ для DF - связывается с content_id (для тип content_type_id = 1) 
# tag - ?
# part - кластеризация
# type_of - краткое описание цели лекции
questions.head()

# question_id - внешний ключ для DF - связывается с content_id (для тип content_type_id = 0)
# bundle_id - код, 
# correct_answer - правильный ответ (но уже в цифре)
# part соответствующий раздел теста 
# tags - кластеризация вопросов 
Задача: Проанализировать как можно больше характеристик, влияющих на успеваемость студентов. 

Успеваемость - это процент, корректно выполненых студентом вопросов на протяжении всего учебного периода
# Для начала я хочу разделить лекции и вопросы, чтобы в дальнейшем анализировать с использованием дополнительных словарей 
train_questions = train[train['content_type_id'] == 0].merge(questions, left_on='content_id', right_on='question_id', how='left')
train_lectures = train[train['content_type_id'] == 1].merge(lectures, left_on='content_id', right_on='lecture_id', how='left')
# Также я создам признак, по которому сделаем дополнительную проверку на соответсвие справочника по правильным ответам с данными из исходной таблицы
train_questions['is_correct'] = (train_questions['user_answer'] == train_questions['correct_answer']).astype(int)
print('Количество различий: ',train_questions[train_questions['is_correct'] != train_questions['answered_correctly']].value_counts().to_string())
# Так как различий нет - убираем один из атрибутов
train_questions.pop('answered_correctly')
train_questions.pop('user_answer')
train_questions.pop('correct_answer')
1 Гипотеза:

Просмотр ответов на предыдущий блок вопросов помогает соорентироваться в текущем
# Анализ использования объяснений
explanation_stats = train_questions.copy()
# Как мы поняли, в 0 task_container_id в стобце prior_question_had_explanation проставляются пустые ячейки, поэтому уберем их для анализа
explanation_stats = explanation_stats.dropna(subset=['prior_question_had_explanation'])

# Делаем группировку по среднему от правильнвых ответов
explanation_stats = train_questions.groupby('prior_question_had_explanation')['is_correct'].mean()


# Визуализируем
sns.barplot(x=explanation_stats.index, y=explanation_stats.values)
plt.title("Зависимость успеваемости от наличия объяснений")
plt.show()
2 Гипотеза:

Правильность ответов зависит от просмотра лекций
# У лекций есть типы, но я не смог перевести  значение, поэтому будем считать, что они в равной степени влияют на успешность в прохождении теста 
train_lectures['type_of'].unique()
# Для начала создадим переменнцую, в которой запишем id человека и факт просмотренной определенной лекции 
#train_lectures_id = train_lectures.groupby('user_id')['task_container_id'].value_counts()
#train_lectures_id = train_lectures.groupby('user_id')['content_type_id'].value_counts()
# Подсчитаем количество лекций, просмотренных студентом
user_lecture_counts = train_lectures.groupby('user_id')['content_id'].count().reset_index()
user_lecture_counts.columns = ['user_id', 'lecture_count']
# Подсчитаем успеваемость студента 
# Группируем данные по user_id и считаем общее количество ответов и количество правильных ответов
user_correct_answers = train_questions.groupby('user_id')['is_correct'].agg(['count', 'sum']).reset_index()

# Переименовываем столбцы для ясности
user_correct_answers.columns = ['user_id', 'total_answers', 'correct_answers']

# Рассчитываем процент правильных ответов для каждого студента
user_correct_answers['correct_answer_percentage'] = (user_correct_answers['correct_answers'] / user_correct_answers['total_answers']) * 100

# Оставим только столбцы user_id и процент правильных ответов
user_correct_answers = user_correct_answers[['user_id', 'correct_answer_percentage']]
# Объединяем две таблицы по user_id
merged_data = pd.merge(user_lecture_counts, user_correct_answers, on='user_id', how='inner')

# Визуализация корреляции: количество лекций vs процент правильных ответов
plt.figure(figsize=(10,6))
sns.scatterplot(data=merged_data, x='lecture_count', y='correct_answer_percentage', alpha=0.5)
plt.title('Корреляция между количеством лекций и процентом правильных ответов')
plt.xlabel('Количество просмотренных лекций')
plt.ylabel('Процент правильных ответов (%)')
plt.show()

# Вычисляем коэффициент корреляции Пирсона между количеством лекций и процентом правильных ответов
correlation = merged_data['lecture_count'].corr(merged_data['correct_answer_percentage'])
print(f'Коэффициент корреляции между количеством лекций и процентом правильных ответов: {correlation:.2f}')
