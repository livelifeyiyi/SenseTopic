#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as Math
import pylab as Plot
import argparse
import tsne
import sys
import codecs
reload(sys)
sys.setdefaultencoding('utf-8')


def plot_vectors():
    from PIL import Image
    image = Image.open('../tsne/figure_1000_zoom.png')
    image.show()
    image.format
# plot_vectors()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', action='store', dest='labelfile', required=True,
                        help='Path of label file')
    parser.add_argument('-v', action='store', dest='vectorFile', required=True,
                        help='Embedding vector')
    parser.add_argument('-d', action='store', type=int, dest='demension',
                        required=True, help='Demension of vector')
    parser.add_argument('-p', action='store', type=int, dest='perplexity',
                        default=20, help='Perplexity, usually between 20 to 50')
    r = parser.parse_args()

    X = Math.loadtxt(r.vectorFile, dtype='float', delimiter= ' ', usecols=range(200))

    with codecs.open(r.labelfile, mode='r', encoding='utf-8') as f:
        labels = f.read().upper().splitlines()
        Y = tsne.tsne(X, 2, r.demension, r.perplexity)
        fig, ax = Plot.subplots()
        ax.scatter(Y[:, 0], Y[:, 1], 20)

        for i, txt in enumerate(labels):
            ax.annotate(txt, (Y[:, 0][i], Y[:, 1][i]))

        Plot.show()
