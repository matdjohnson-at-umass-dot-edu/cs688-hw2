
import os
import re

import numpy as np
from PIL import Image


class DatasetHolder:

    def __init__(self):
        self.test_text = list()
        self.test_images = list()
        self.test_words = list()
        self.train_text = list()
        self.train_images = list()
        self.train_words = list()
        # from the assignment, letters are ‘etainoshrd’
        self.letter_index = ['e', 't', 'a', 'i', 'n', 'o', 's', 'h', 'r', 'd']

    def get_test_text(self):
        return_test_text = list()
        for test_text_entry in self.test_text:
            return_test_text.append(test_text_entry.copy())
        return return_test_text

    def get_test_images(self):
        return_test_images = list()
        for test_images_entry in self.test_images:
            return_test_images.append(test_images_entry.copy())
        return return_test_images

    def get_test_words(self):
        return_test_words = list()
        for test_word in self.test_words:
            return_test_words.append(str(test_word))
        return return_test_words

    def get_train_text(self):
        return_train_text = list()
        for train_text_entry in self.train_text:
            return_train_text.append(train_text_entry.copy())
        return return_train_text

    def get_train_images(self):
        return_train_images = list()
        for train_image_entry in self.train_images:
            return_train_images.append(train_image_entry.copy())
        return return_train_images

    def get_train_words(self):
        return_train_words = list()
        for train_word in self.train_words:
            return_train_words.append(str(train_word))
        return return_train_words

    def get_letter_index(self):
        return_letter_index = list()
        for letter in self.letter_index:
            return_letter_index.append(str(letter))
        return return_letter_index

    def read_file_contents(self):
        train_text_matcher = re.compile('train-img-([0-9]+).txt')
        train_img_matcher = re.compile('train-img-([0-9]+).png')
        test_text_matcher = re.compile('test-img-([0-9]+).txt')
        test_img_matcher = re.compile('test-img-([0-9]+).png')
        test_text_files = dict()
        test_image_files = dict()
        train_text_files = dict()
        train_image_files = dict()
        for file in os.listdir('./data'):
            train_text_match = train_text_matcher.match(file)
            train_img_match = train_img_matcher.match(file)
            test_text_match = test_text_matcher.match(file)
            test_img_match = test_img_matcher.match(file)
            if train_text_match:
                train_text_files[int(train_text_match.group(1))] = "./data/" + file
            if train_img_match:
                train_image_files[int(train_img_match.group(1))] = "./data/" + file
            if test_text_match:
                test_text_files[int(test_text_match.group(1))] = "./data/" + file
            if test_img_match:
                test_image_files[int(test_img_match.group(1))] = "./data/" + file
        train_text_file_indices = sorted(train_text_files)
        train_image_file_indices = sorted(train_image_files)
        test_text_file_indices = sorted(test_text_files)
        test_image_file_indices = sorted(test_image_files)
        for file_name_index in train_text_file_indices:
            self.train_text.append(self.text_file_content_to_numpy_array(train_text_files[file_name_index]))
        for file_name_index in train_image_file_indices:
            self.train_images.append(self.image_file_to_numpy_arrray(train_image_files[file_name_index]))
        for file_name_index in test_text_file_indices:
            self.test_text.append(self.text_file_content_to_numpy_array(test_text_files[file_name_index]))
        for file_name_index in test_image_file_indices:
            self.test_images.append(self.image_file_to_numpy_arrray(test_image_files[file_name_index]))
        train_words_file_object = open('./data/train-words.txt', 'r')
        for word in train_words_file_object:
            self.train_words.append(word.strip())
        test_words_file_object = open('./data/test-words.txt', 'r')
        for word in test_words_file_object:
            self.test_words.append(word.strip())

    def text_file_content_to_numpy_array(self, file_name):
        text_file_object = open(file_name, 'r')
        file_lines = list()
        for line in text_file_object:
            file_lines.append(line.strip().split(' '))
        numpy_array = np.array(file_lines, dtype=np.float128)
        return numpy_array

    def image_file_to_numpy_arrray(self, file_name):
        image_object = Image.open(file_name)
        numpy_array = np.asarray(image_object, dtype=np.float128)
        return numpy_array

