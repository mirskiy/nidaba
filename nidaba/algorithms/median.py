# -*- coding: utf-8 -*-
"""
nidaba.algorithms.median
~~~~~~~~~~~~~~~~~~~~~~~~

Implementation of various algorithms to calculate approximate generalized
median strings on recognition results.
"""

from __future__ import unicode_literals, print_function, absolute_import
from __future__ import division

import codecs
import numpy as np
import operator
import unicodedata
import itertools
import math
import collections

from nidaba.nidabaexceptions import NidabaAlgorithmException
from nidaba.algorithms.string import full_edit_distance, edit_distance

# stolen from kraken
class ocr_record(object):
    """ 
    A record object containing the recognition result of a single line
    """
    def __init__(self, prediction, cuts, confidences):
        self.prediction = prediction
        self.cuts = cuts
        self.confidences = confidences

    def __len__(self):
        return len(self.prediction)

    def __str__(self):
        return self.prediction

    def __iter__(self):
        self.idx = -1
        return self

    def __add__(self, other):
        new = ocr_record(self.prediction, self.cuts, self.confidences)
        if isinstance(other, collections.Iterable):
            new.prediction += other[0]
            new.cuts.extend(other[1])
            new.confidences.extend(other[2])
        elif isinstance(other, ocr_record):
            new.prediction += other.prediction
            new.cuts += other.cuts
            new.confidences += other.confidences
        return new

    def next(self):
        if self.idx + 1 < len(self):
            self.idx += 1
            return (self.prediction[self.idx], self.cuts[self.idx],
                    self.confidences[self.idx])
        else:
            raise StopIteration

    def __getitem__(self, key):
        if isinstance(key, slice):
            return ocr_record(self.prediction.__getitem__(key),
                              self.cuts.__getitem__(key),
                              self.confidences.__getitem__(key))
        elif isinstance(key, int):
            if key < 0:
                key += len(self)
            if key >= len(self):
                raise IndexError('Index (%d) is out of range' % key)
            return (self.prediction[key], self.cuts[key],
                    self.confidences[key])
        else:
            raise TypeError('Invalid argument type')

class confidence_weighted_edit_distance(object):
    """
    Edit distance function using symbol-based confidence weighted edit
    distance to an unweighted input string adapted from [0].

    [0] Barat, Cécile, et al. "Weighted symbols-based edit distance for
    string-structured image classification." Machine Learning and Knowledge
    Discovery in Databases. Springer Berlin Heidelberg, 2010. 72-86.
    """
    def __init__(self, strings):
        pass

    def distance(self, string, record):
        def cost(tok1, tok2, *args, **kwargs):
            if tok1[0] != tok2[0]:
                return max(tok1[2], tok2[2])
            else:
                return abs(tok2[2] - tok1[2])
        m = full_edit_distance(string, record, ins_func=cost, del_func=cost, sub_func=cost)[0]
        return m[-1][-1]

class naive_edit_distance(object):
    """
    Edit distance function using the unweighted Levenshtein distance measure.
    """
    def __init__(self, strings):
        pass

    def distance(self, string, record):
        return edit_distance(string.prediction, record.prediction)


def approximate_median(strings, distance_class=naive_edit_distance):
    """
    Calculates an approximate generalized median string using the greedy
    algorithm described in [0].

    Args:
        strings (list): A list ocr_records.

    Returns:
        A unicode string representing the generalized median string.

    [0]  F. Casacuberta and M. de Antonio. A greedy algorithm for computing
    approximate median strings. In VII Simposium Nacional de Reconocimiento de
    Formas y Analisis de Imagenes., pages 193–198, april 1997.
    """
    alphabet = set()
    for x in strings:
        alphabet.update(set(x.prediction))
    median = ocr_record(u'', [], [])
    mediandist = [np.inf]
    maxlen = len(max(strings, key=len))
    distance = distance_class(strings)
    while True:
        mm = np.inf
        for j in alphabet:
            confs = []
            # A new symbol may correspond to all matching symbols in the input
            # strings truncated to the length of the prefix + symbol. As
            # matching potential new symbols based on the location on the page
            # increases the input alphabet inordinately and requires an
            # additional distance measure we just calculate the mean of all
            # confidences so lower-than-average confidences in the input
            # strings are still penalized.
            for s in strings:
                confs += [x[2] for x in s[:len(median)+1] if x[0] == j]
            if not confs:
                confs = [100]
            ma = median + (j, [None], [np.mean(confs)])

            m = 0
            t = 0
            # XXX: horribly inefficient. Running the algorithm as intended
            # calculates the cost once for s[:len(ma)], keeps the matrix around
            # and then calculates t. Also the matrix is kept between
            # iterations. As we have no knowledge of the operation of the
            # distance function there is no other way for now.
            for s in strings:
                m += distance.distance(ma, s[:len(ma)])
                t += distance.distance(ma, s)
            if m < mm:
                mm = m
                b_ma = ma
        mediandist.append(t)
        median = b_ma
        if len(median) > maxlen and mediandist[-1] > mediandist[-2]:
            return median[:-1]


def improve_median(candidate, strings, distance_class=naive_edit_distance):
    """
    Tries to improve an approximate generalized median string using iterative
    systematic perturbation as described in [0]

    Args:
        candidate (unicode): Approximate generalized median string
        strings (list): A list of lists of tuples (s, c) where s is a unicode
                        object and c is an integer confidence attached to it.

    Returns:
        A unicode string representing the approximate median string.

    [0] Martínez-Hinarejos, Carlos D., Alfons Juan, and Francisco Casacuberta.
    Use of median string for classification. Pattern Recognition, 2000.
    Proceedings. 15th International Conference on. Vol. 2. IEEE, 2000.
    """
    alphabet = set()
    for x in strings:
        alphabet.update(set(x.prediction))
    i = 0
    distance = distance_class(strings)
    min_dist = (candidate, sum([distance.distance(candidate, s) for s in strings]))
    while True:
        for sym in alphabet:
            # substitution
            sub = candidate[:i] + sym + candidate[i+1:]
            dist = sum([distance.distance(sub, s) for s in strings]) 
            min_dist = (sub, dist) if dist < min_dist[1] else min_dist
            # insertion
            ins = candidate[:i] + sym + candidate[i:]
            dist = sum([distance.distance(ins, s) for s in strings]) 
            min_dist = (ins, dist) if dist < min_dist[1] else min_dist
        # deletion
        _del = candidate[i+1:]
        dist = sum([distance.distance(_del, s) for s in strings])
        if dist < min_dist[1]:
            min_dist = (_del, dist)
            i -= 1
        i += 1
        if i == len(min_dist[0]):
            return min_dist[0]
