import requests
import re
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from Model import Model


class AuthorsAttribution:

    # w_files - флаг, надо ли заносить данные в файлы.
    def __init__(self, w_files=True):
        self.__w_files = w_files
        self.data = {}
        self.tokenized_data = {}
        self.frequency_dict = {}

    # Чтение данных.
    # names - список имён авторов, которым соответствуют файлы с сылками на тексты.
    def read_data(self, names):
        self.data = {name: [] for name in names}

        for name in names:
            with open(name + '.txt', encoding='utf-8') as refs_file:
                refs_file.read(1)

                for line in refs_file:
                    r = requests.get(line.strip(' \n'))
                    texts = re.findall('<p class="article-block article-block-unstyled">([^>]*)</p>', r.text)
                    united_text = ''
                    for text in texts:
                        united_text += re.sub('&quot;', '"', text)
                    self.data[name].append(united_text)

                if self.__w_files:
                    output_file = open(name + '_texts.txt', 'w', encoding='utf-8')
                    for text in self.data[name]:
                        output_file.write(text)
                    output_file.close()

    # Токенизация текстов.
    def tokenize_data(self):
        self.tokenized_data = {}
        model = Model('russian-syntagrus-ud-2.3-181115.udpipe')
        for name in self.data:
            self.tokenized_data.update({name: []})
            for text in self.data[name]:
                sentences = model.tokenize(text)
                for s in sentences:
                    model.tag(s)
                    model.parse(s)
                self.tokenized_data[name].append(list(map(lambda el: el.split('\t'),
                                                          model.write(sentences, "conllu").split('\n'))))

            if self.__w_files:
                output_file = open(name + '_tokenized_texts.txt', 'w', encoding='utf-8')
                for text in self.tokenized_data[name]:
                    for word in text:
                        if len(word) > 1:
                            output_file.write(word[1] + '[' + word[3] + '] ')
                    output_file.write('\n\n')
                output_file.close()

    # Создание частотного словаря.
    # part - По какой части данных составлять словарь
    def __make_frequency_dict_part(self, part=1.0):
        frequency_dict = {}

        for name in self.tokenized_data:
            frequency = {}
            for i in range(int(len(self.tokenized_data[name]) * part)):
                text = self.tokenized_data[name][i]
                for word in text:
                    if len(word) > 1 and not word[3] == 'PUNCT':
                        try:
                            frequency[word[2]] += 1
                        except KeyError:
                            frequency.update({word[2]: 1})
            frequency_dict.update({name: sorted(frequency.items(), key=itemgetter(1), reverse=True)})

        return frequency_dict

    # Создание словаря частот частей речи в текстах.
    def __make_part_of_speech_dict(self):
        parts_of_speech = {}
        for name in self.tokenized_data:
            parts_of_speech.update({name: []})
            for text in self.tokenized_data[name]:
                attributes = np.zeros(7)

                for word in text:
                    if len(word) > 1:
                        if word[3] == 'NOUN':
                            attributes[1] += 1
                        elif word[3] == 'VERB':
                            attributes[2] += 1
                        elif word[3] == 'ADJ':
                            attributes[3] += 1
                        else:
                            if word[3] == 'ADP' or word[3] == 'CCONJ' or word[3] == 'PART':
                                attributes[4] += 1
                            if word[2] == 'в':
                                attributes[5] += 1
                            elif word[2] == 'не':
                                attributes[6] += 1
                    elif not word[0].find('sent_id') == -1:
                        attributes[0] = float(re.findall('# sent_id = ([0-9]+)', word[0])[0])

                parts_of_speech[name].append(attributes / len(text))

            if self.__w_files:
                output_file = open(name + '_parts_of_speech.txt', 'w', encoding='utf-8')
                total = np.zeros(7)
                for line in parts_of_speech[name]:
                    output_file.write('NOUNS: ' + str(line[1]) + '  VERBS: ' + str(line[2]) + '  ADJECTIVES: '
                                      + str(line[3]) + '  SERVICES: ' + str(line[4]) + '  В: ' + str(line[5])
                                      + '  НЕ: ' + str(line[6]) + '  SENTENCES: ' + str(line[0]) + '\n')
                    total += line

                total /= len(parts_of_speech[name])
                output_file.write('\nAVERAGE:\nNOUNS: ' + str(total[1]) + '  VERBS: ' + str(total[2]) + '  ADJECTIVES: '
                                  + str(total[3]) + '  SERVICES: ' + str(total[4]) + '  В: ' + str(total[5])
                                  + '  НЕ: ' + str(total[6]) + '  SENTENCES: ' + str(total[0]) + '\n')
                output_file.close()

        return parts_of_speech

    def __make_frequency_dict(self):
        self.frequency_dict = self.__make_frequency_dict_part()

        if self.__w_files:
            for name in self.frequency_dict:
                output_file = open(name + '_words.txt', 'w', encoding='utf-8')
                for word in self.frequency_dict[name]:
                    output_file.write(word[0] + ' ' + str(word[1]) + '\n')
                output_file.close()

    # Проверка закона Ципфа.
    def zipf_law_check(self):
        self.__make_frequency_dict()

        for name in self.frequency_dict:
            frequency = list(map(lambda el: el[1], self.frequency_dict[name]))

            plt.figure(figsize=(15, 10))
            plt.plot(np.arange(len(frequency)), frequency)
            plt.title('Words of ' + name, fontsize=20)
            plt.ylim((0, 100))
            plt.grid(ls=':')
            plt.show()

    def __classification(self, matrix, labels, depth=5):
        matrix = np.array(matrix)
        labels = np.array(labels)
        x_train, x_test, y_train, y_test = train_test_split(matrix, labels, random_state=42)

        dtc = DecisionTreeClassifier(max_depth=depth)
        dtc.fit(x_train, y_train)
        y_predict = dtc.predict(x_test)
        return accuracy_score(y_test, y_predict)

    # Атрибуция авторов первым способом:
    # По частоте встречи слов, наиболее частых у авторов в целом (100 самых частых от каждого).
    def authors_attribution_1(self):
        most_frequent_words = {}
        for words in list(map(lambda el: el[30:130], self.__make_frequency_dict_part(0.75).values())):
            most_frequent_words.update({word[0]: 0 for word in words})
        most_frequent_words = {el[1]: el[0] for el in enumerate(most_frequent_words.keys())}

        matrix = []
        labels = []
        author_index = 0
        for name in self.tokenized_data:
            for text in self.tokenized_data[name]:
                attributes = np.zeros(len(most_frequent_words))

                for word in text:
                    if len(word) > 1:
                        try:
                            attributes[most_frequent_words[word[2]]] += 1
                        except KeyError:
                            pass
                matrix.append(attributes / len(text))
                labels.append(author_index)

            author_index += 1

        return self.__classification(matrix, labels, 5)

    # Атрибуция авторов вторым способом:
    # По количеству слов в предложении, частоте различных частей речи, определённых предлогов и частиц.
    def authors_attribution_2(self):
        matrix = []
        labels = []
        author_index = 0
        attributes = self.__make_part_of_speech_dict()

        for name in attributes:
            for line in attributes[name]:
                matrix.append(line)
                labels.append(author_index)
            author_index += 1

        return self.__classification(matrix, labels, 3)

    # Атрибуция авторов вторым способом:
    # По частоте появления новых слов.
    def authors_attribution_3(self):
        matrix = []
        labels = []
        author_index = 0
        for name in self.tokenized_data:
            for text in self.tokenized_data[name]:
                attributes = np.zeros(110)
                words = {}
                words_num = 0
                i = 0

                for word in text:
                    if len(word) > 1 and not word[3] == 'PUNCT':
                        if not words.get(word[2]):
                            words.update({word[2]: 1})
                            attributes[i] = words_num
                            i += 1
                            if i == 110:
                                break
                        words_num += 1

                matrix.append(attributes[10:] / len(text))
                labels.append(author_index)

            author_index += 1

        return self.__classification(matrix, labels, 5)


aa = AuthorsAttribution()
aa.read_data(['borisenko', 'osteopat_v_moskve', 'kino_ot_glushkova', 'sputnikfm_lilya', 'topcinema'])
aa.tokenize_data()
aa.zipf_law_check()
print(aa.authors_attribution_1())
print(aa.authors_attribution_2())
print(aa.authors_attribution_3())
