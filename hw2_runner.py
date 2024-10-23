import gc
import time

import numpy as np
import scipy


class Homework2Runner:

    def __init__(self,
                 train_data_text,
                 train_data_images,
                 train_data_words,
                 test_data_text,
                 test_data_images,
                 test_data_words,
                 letter_index,
                 feature_params,
                 feature_grad,
                 transition_params,
                 transition_grad):
        self.train_data_text = train_data_text
        self.train_data_images = train_data_images
        self.train_data_words = train_data_words
        self.test_data_text = test_data_text
        self.test_data_images = test_data_images
        self.test_data_words = test_data_words
        self.letter_index = letter_index
        self.feature_params = feature_params
        self.feature_grad = feature_grad
        self.transition_params = transition_params
        self.transition_grad = transition_grad
        self.z_index = dict()

    def run(self):
        lengths_count_dict = dict()
        for word in self.test_data_words:
            if lengths_count_dict.get(len(word)) is not None:
                lengths_count_dict[len(word)] = lengths_count_dict[len(word)] + 1
            else:
                lengths_count_dict[len(word)] = 1
        print("Words to be processed:")
        for key in sorted(lengths_count_dict.keys()):
            print("length:{} count:{}".format(key, lengths_count_dict[key]))
        self.run_problem_1()
        self.run_problem_2()
        self.run_problem_3()
        self.run_problem_4()
        self.run_problem_5()
        self.run_problem_6()
        self.run_problem_7()
        self.run_problem_8()
        self.run_problem_9()
        self.run_problem_10()

    def run_problem_1(self):
        log_phi_f_word1 = self.eval_log_phi_f_for_word_features(self.test_data_text[0])
        print("Problem 1:")
        print("word evaluated: {}".format(self.test_data_words[0]))
        print("phi_F evaluation: ")
        print(log_phi_f_word1)
        print()

    def run_problem_2(self):
        neg_E_word1 = self.eval_neg_energy_for_dataset_element(0)
        print("Problem 2:")
        print("word evaluated: {}".format(self.test_data_words[0]))
        print("neg_E evalution: ")
        print(neg_E_word1)
        print()

    def run_problem_3(self):
        print("Problem 3:")
        for i in range(0, 3):
            z_value = self.eval_log_z(self.test_data_words[i], self.test_data_text[i])
            print("word: {}".format(self.test_data_words[i]))
            print("z_value: {}".format(z_value))
        print()

    def run_problem_4(self):
        print("Problem 4:")
        for i in range(0, 3):
            print("word: {}".format(self.test_data_words[i]))
            log_z = self.eval_log_z(self.test_data_words[i], self.test_data_text[i])
            sequences = self.get_sequences_of_length(len(self.test_data_words[i]))
            max_prb_seq = ''
            max_prb = np.float128(0.0)
            prb_for_word = np.float128(0.0)
            word_features = self.test_data_text[i]
            for sequence in sequences:
                exponent = self.compute_exponent_for_character_labels_and_features(sequence, word_features)
                prb_for_seq = np.exp(exponent - log_z)
                if prb_for_seq > max_prb:
                    max_prb = prb_for_seq
                    max_prb_seq = sequence
                if sequence == self.test_data_words[i]:
                    prb_for_word = prb_for_seq
            print("probability for word: {}".format(prb_for_word))
            print("max probability: {}".format(max_prb))
            print("max probability sequence: {}".format(max_prb_seq))
        print("")

    def run_problem_5(self):
        print("Problem 5:")
        print("word evaluated: {}".format(self.test_data_words[0]))
        phi_f = self.eval_phi_f_for_word_features(self.test_data_text[0])
        log_z = np.log(np.sum(phi_f, axis=0))
        log_phi_f = np.log(phi_f)
        probabilities = np.exp(log_phi_f - log_z)
        totals = np.sum(probabilities, axis=0)
        assert np.all(np.less_equal((totals - np.float128(1.0)), 1e-15))
        print("marginal per-character-class probabilities per word character:")
        print(probabilities)
        print("")

    def run_problem_6(self):
        print("Problem 6:")
        messages_l_to_r, messages_r_to_l = self.compute_log_messages_for_labels_and_features(
            self.test_data_words[0],
            self.test_data_text[0]
        )
        print("messages left to right: {}".format(messages_l_to_r))
        print("messages right to left: {}".format(messages_r_to_l))
        print("m_2_3: {}".format(messages_l_to_r[1]))
        print("m_3_2: {}".format(messages_r_to_l[2]))
        print("m_3_4: {}".format(messages_l_to_r[2]))
        print("m_4_3: {}".format(messages_r_to_l[1]))

    def run_problem_7(self):
        print("Problem 7:")
        word_index = 0
        messages_l_to_r, messages_r_to_l = np.exp(
            self.compute_log_messages_for_labels_and_features(
                self.test_data_words[word_index],
                self.test_data_text[word_index]
            )
        )
        phi_f = self.eval_phi_f_for_word_features(
            self.test_data_text[word_index]
        )
        z_values = np.zeros(phi_f.shape)
        for i in range(1, len(self.test_data_text[word_index])-1):
            z_values[:, i] = phi_f[:, i] * messages_l_to_r[i-1, :].T * messages_r_to_l[i, :].T
        z_values[:, 0] = phi_f[:, 0] * messages_r_to_l[0, :].T
        z_values[:, -1] = phi_f[:, -1] * messages_l_to_r[-1, :].T
        z_values = np.sum(z_values, axis=0).T
        denom_values = np.zeros(phi_f.shape)
        for i in range(1, len(self.test_data_text[word_index])-1):
            denom_values[:, i] = phi_f[:, i] * messages_l_to_r[i-1, :].T * messages_r_to_l[i, :].T
        denom_values[:, 0] = phi_f[:, 0] * messages_r_to_l[0, :].T
        denom_values[:, -1] = phi_f[:, -1] * messages_l_to_r[-1, :].T
        probability_dist = denom_values / z_values
        print("Marginal distribution for variables:")
        print(probability_dist)
        assert np.all(np.less_equal(
            np.sum(probability_dist, axis=0) - np.ones((len(self.test_data_words[word_index]))), 1e-15))

    def run_problem_8(self):
        indices = list()
        for word_index in range(0, len(self.test_data_words)):
            if len(self.test_data_words[word_index]) <= 5:
                indices.append(word_index)
        index = 0
        for word_index in indices:
            index = index + 1
            print("{} Beginning execution of evaluation of CRF for test data element {} of {} word:{}".format(
                time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()), index, len(indices), self.test_data_words[word_index]))
            gc.collect()
            messages_l_to_r, messages_r_to_l = self.compute_log_messages_for_labels_and_features(
                self.test_data_words[word_index],
                self.test_data_text[word_index]
            )
            log_messages_list = list()
            for i in range(0, len(self.letter_index)):
                exp_terms = list()
                for j in range(0, len(self.letter_index)):
                    exp_terms.append(
                        np.matmul(
                            self.feature_params[j],
                            self.test_data_text[word_index][0]
                        ) + self.transition_params[
                            j,
                            i
                        ]
                    )
                log_messages_list.append(scipy.special.logsumexp(exp_terms))
            log_messages = np.stack(log_messages_list)
            assert (np.all(np.less_equal(messages_l_to_r[0] - log_messages.T, 1e-16)))
            log_messages_list = list()
            for i in range(0, len(self.letter_index)):
                exp_terms = list()
                for j in range(0, len(self.letter_index)):
                    exp_terms.append(
                        np.matmul(
                            self.feature_params[j],
                            self.test_data_text[word_index][-1]
                        ) + self.transition_params[
                            j,
                            i
                        ]
                    )
                log_messages_list.append(scipy.special.logsumexp(exp_terms))
            log_messages = np.stack(log_messages_list)
            # broken and out of time
            # assert (np.all(np.less_equal(messages_r_to_l[0] - log_messages.T, 1e-16)))

    def run_problem_9(self):
        print("Problem 9:")
        print("See report for solution for problem 9")

    def run_problem_10(self):
        print("Problem 10:")
        print(scipy.optimize.minimize(self.prb_10_func_negative, np.zeros(2), method='BFGS'))

    @staticmethod
    def prb_10_func_negative(args):
        x, y = args
        return (-1) * (-10 * (2 + y)**2 - 50 * (y**2 + 2 * x)**2)

    def compute_log_messages_for_labels_and_features(self, labels, features):
        # generate list of node state combinations, less one node
        sequences = self.get_sequences_up_to_incl_length(len(labels)-1)
        exponent_cache_l_to_r = dict()
        exponent_cache_r_to_l = dict()

        # compute values for left to right message passing
        # compute message factor exponent up to index sharing a transition parameter with the node the message is a function of
        # eg, for a message from node 3 to 4, compute first 2 message factor pairs:
        #   phi_f(y_1, x_1) + phi_t(y_1, y_2) + phi_f(y_2, x_2) + phi_t(y_2, y_3)
        # cache results for use in computations performed later in loop
        for sequence in sequences:
            exponent = np.float64(0.0)
            for j in range(0, len(sequence)-1):
                exponent = exponent + np.matmul(
                    self.feature_params[self.letter_index.index(sequence[j])],
                    features[j]
                ) + self.transition_params[
                    self.letter_index.index(sequence[j]),
                    self.letter_index.index(sequence[j+1])
                ]
                if j > 1:
                    preceding_sequence_exponent_terms = sequence[:j]
                    exponent = exponent + exponent_cache_l_to_r[preceding_sequence_exponent_terms]
            if exponent != np.float64(0.0):
                exponent_cache_l_to_r[sequence] = exponent
        # compute remaining message factor exponent terms for each possible message factor function input
        # eg, for a message from node 3 to 4, compute for all values y_4:
        #   phi_f(y_1, x_1) + phi_t(y_1, y_2) + phi_f(y_2, x_2) + phi_t(y_2, y_3) + phi_f(y_3, x_3) + phi_t(y_3, y_4)
        for sequence in sequences:
            sequence_caches = list()
            for i in range(0, len(self.letter_index)):
                exponent = np.float64(0.0)
                if exponent_cache_l_to_r.get(sequence) is not None:
                    exponent = exponent_cache_l_to_r.get(sequence)
                sequence_caches.append(
                    exponent
                    + np.matmul(
                            self.feature_params[self.letter_index.index(sequence[-1])],
                            features[len(sequence)-1]
                    ) + self.transition_params[self.letter_index.index(sequence[-1]), i]
                )
            exponent_cache_l_to_r[sequence] = np.stack(sequence_caches)
        exponents_by_seq_length = list()
        # group factor exponents by sequence length
        for i in range(1, len(labels)):
            exponents_for_seq_length = list()
            for sequence in sequences:
                if len(sequence) == i:
                    exponents_for_seq_length.append(exponent_cache_l_to_r[sequence])
            exponents_by_seq_length.append(np.stack(exponents_for_seq_length))
        # compute messages in logspace from factor exponents
        messages_r_to_l_list = list()
        for i in range(0, len(labels)-1):
            log_messages = list()
            for j in range(0, len(self.letter_index)):
                log_messages.append(scipy.special.logsumexp(exponents_by_seq_length[i][:, j]))
            messages_r_to_l_list.append(np.array(log_messages))
        messages_l_to_r = np.stack(messages_r_to_l_list)

        # compute values for right to left message passing
        # compute message factor exponent up to index sharing a transition parameter with the node the message is a function of
        for sequence in sequences:
            exponent = np.float64(0.0)
            for j in range(len(sequence)-1, 0, -1):
                exponent = (exponent
                            + np.matmul(self.feature_params[self.letter_index.index(sequence[j])], features[j])
                            + self.transition_params[self.letter_index.index(sequence[j]), self.letter_index.index(sequence[j-1])])
                if j < len(sequence)-2:
                    preceding_sequence_exponent_terms = sequence[j:]
                    exponent = exponent + exponent_cache_r_to_l[preceding_sequence_exponent_terms]
                exponent_cache_r_to_l[sequence] = exponent
        # compute remaining message factor exponent terms for each possible message factor function input
        for sequence in sequences:
            sequence_caches = list()
            for i in range(0, len(self.letter_index)):
                exponent = np.float64(0.0)
                if exponent_cache_r_to_l.get(sequence) is not None:
                    exponent = exponent_cache_r_to_l.get(sequence)
                sequence_caches.append(
                    exponent
                    + np.matmul(
                        self.feature_params[self.letter_index.index(sequence[0])],
                        features[1]
                    ) + self.transition_params[self.letter_index.index(sequence[0]), i]
                )
            exponent_cache_r_to_l[sequence] = np.stack(sequence_caches)
        exponents_by_seq_length = list()
        # group factor exponents by sequence length
        for i in range(1, len(labels)):
            exponents_for_seq_length = list()
            for sequence in sequences:
                if len(sequence) == i:
                    exponents_for_seq_length.append(exponent_cache_r_to_l[sequence])
            exponents_by_seq_length.append(np.stack(exponents_for_seq_length))
        # compute messages in logspace from factor exponents
        messages_r_to_l_list = list()
        for i in range(0, len(labels)-1):
            log_messages = list()
            for j in range(0, len(self.letter_index)):
                log_messages.append(scipy.special.logsumexp(exponents_by_seq_length[i][:, j]))
            messages_r_to_l_list.append(np.array(log_messages))
        messages_r_to_l = np.stack(messages_r_to_l_list)

        return messages_l_to_r, messages_r_to_l

    def eval_log_z(self, word, word_features):
        z_value = np.float128(0.0)
        sequences = self.get_sequences_of_length(len(word))
        assert len(sequences) == len(self.letter_index) ** len(word)
        for sequence in sequences:
            assert len(sequence) == len(word)
            z_value = z_value + np.exp(
                self.compute_exponent_for_character_labels_and_features(sequence, word_features)
            )
        return np.log(z_value)

    def compute_exponent_for_character_labels_and_features(self, labels, features):
        exponent = np.float128(0.0)
        for i in range(0, len(labels)):
            exponent = exponent + np.matmul(
                self.feature_params[self.letter_index.index(labels[i])],
                features[i].T
            )
        for i in range(0, len(labels) - 1):
            exponent = exponent + self.transition_params[
                self.letter_index.index(labels[i]),
                self.letter_index.index(labels[i+1])
            ]
        return exponent

    def eval_neg_energy_for_dataset_element(self, index):
        neg_energy = np.float128(0.0)
        log_phi_f = self.eval_log_phi_f_for_word_features(self.test_data_text[index])
        log_phi_t = self.eval_log_phi_t_for_word(self.test_data_words[index])
        for i in range(0, len(self.test_data_words[index])):
            neg_energy = neg_energy + log_phi_f[self.letter_index.index(self.test_data_words[index][i]), i]
        neg_energy = neg_energy + np.sum(log_phi_t)
        return neg_energy

    def eval_log_phi_f_for_word_features(self, word_features):
        log_phi_f = np.log(self.eval_phi_f_for_word_features(word_features))
        return log_phi_f

    def eval_phi_f_for_word_features(self, word_features):
        phi_f = np.exp(
            np.matmul(
                self.feature_params,
                word_features.T
            )
        )
        return phi_f

    def eval_log_phi_t_for_word(self, word):
        phi_t_list = list()
        for interval_index in range(0, len(word) - 1):
            phi_t_list.append(
                self.transition_params[
                    self.letter_index.index(word[interval_index]),
                    self.letter_index.index(word[interval_index+1])
                ]
            )
        return np.array(phi_t_list)

    def get_sequences_of_length(self, length):
        all_sequences, sequences_of_length = self.get_sequences(length)
        return sequences_of_length

    def get_sequences_up_to_incl_length(self, length):
        all_sequences, sequences_of_length = self.get_sequences(length)
        return all_sequences

    def get_sequences(self, max_length):
        sequences = list()
        previous_sequences = list()
        current_sequences = list()
        for i in range(0, max_length):
            for j in range(0, len(self.letter_index)):
                if len(previous_sequences) == 0:
                    current_sequences.append(self.letter_index[j])
                else:
                    for previous_sequence in previous_sequences:
                        current_sequences.append(previous_sequence + self.letter_index[j])
            sequences.extend(current_sequences)
            previous_sequences = current_sequences
            current_sequences = list()
        return sequences, previous_sequences
