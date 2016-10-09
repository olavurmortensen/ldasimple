#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Author-topic model.
"""

import pdb
from pdb import set_trace as st

import logging
import numpy
import numbers

from gensim import utils, matutils
from gensim.models.ldamodel import dirichlet_expectation, get_random_state
from gensim.models import LdaModel
from gensim.models.hdpmodel import log_normalize  # For efficient normalization of variational parameters.
from six.moves import xrange

from pprint import pprint

# log(sum(exp(x))) that tries to avoid overflow
try:
    # try importing from here if older scipy is installed
    from scipy.maxentropy import logsumexp
except ImportError:
    # maxentropy has been removed in recent releases, logsumexp now in misc
    from scipy.misc import logsumexp

logger = logging.getLogger('gensim.models.atmodel')


class LdaSimple(LdaModel):
    """
    """

    def __init__(self, corpus=None, num_topics=100, id2word=None,
            threshold=0.001, iterations=10, alpha=None, eta=None,
            eval_every=1, random_state=None):

        if alpha is None:
            alpha = 1.0 / num_topics
        if eta is None:
            eta = 1.0 / num_topics

        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise ValueError('at least one of corpus/id2word must be specified, to establish input space dimensionality')

        if self.id2word is None:
            logger.warning("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        elif len(self.id2word) > 0:
            self.num_terms = 1 + max(self.id2word.keys())
        else:
            self.num_terms = 0

        if self.num_terms == 0:
            raise ValueError("cannot compute LDA over an empty collection (no terms)")
        
        logger.info('Vocabulary consists of %d words.', self.num_terms)

        self.corpus = corpus
        self.iterations = iterations
        self.num_topics = num_topics
        self.threshold = threshold
        self.alpha = alpha
        self.eta = eta
        self.num_docs = len(corpus)
        self.eval_every = eval_every
        self.random_state = random_state

        self.random_state = get_random_state(random_state)

        if corpus is not None:
            self.inference(corpus)

    def inference(self, corpus=None):
        if corpus is None:
            corpus = self.corpus

        logger.info('Starting inference. Training on %d documents.', len(corpus))

        # Initial value of gamma and lambda.
        var_gamma = self.random_state.gamma(100., 1. / 100.,
                (self.num_docs, self.num_topics))
        var_lambda = self.random_state.gamma(100., 1. / 100.,
                (self.num_topics, self.num_terms))

        var_phi = numpy.zeros((self.num_docs, self.num_terms, self.num_topics))

        Elogtheta = dirichlet_expectation(var_gamma)
        Elogbeta = dirichlet_expectation(var_lambda)
        expElogbeta = numpy.exp(Elogbeta)
        expElogtheta = numpy.exp(Elogtheta)
        likelihood = self.eval_likelihood(Elogtheta, Elogbeta)
        logger.info('Likelihood: %.3e', likelihood)
        for iteration in xrange(self.iterations):
            st()
            # Update phi.
            for d, doc in enumerate(corpus):
                ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
                cts = numpy.array([cnt for _, cnt in doc])  # Word counts.
                for v in ids:
                    for k in xrange(self.num_topics):
                        var_phi[d, v, k] = expElogtheta[d, k] * expElogbeta[k, v]
                        # var_phi[d, v, k] = numpy.exp(Elogtheta[d, k] + Elogbeta[k, v])
                    # Normalize phi.
                    (log_var_phi_dv, _) = log_normalize(var_phi[d, v, :])
                    var_phi[d, v, :] = numpy.exp(log_var_phi_dv)

            # Update gamma.
            for d, doc in enumerate(corpus):
                ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
                cts = numpy.array([cnt for _, cnt in doc])  # Word counts.
                for k in xrange(self.num_topics):
                    var_gamma[d, k] = 0.0
                    var_gamma[d, k] += self.alpha
                    for vi, v in enumerate(ids):
                        var_gamma[d, k] += cts[vi] * var_phi[d, v, k]

            # Update lambda.
            for k in xrange(self.num_topics):
                for v in xrange(self.num_terms):
                    var_lambda[k, v] = 0.0
                    var_lambda[k, v] += self.eta
                    for d, doc in enumerate(corpus):
                        # Get the count of v in doc. If v is not in doc, return 0.
                        cnt = dict(doc).get(v, 0)
                        var_lambda[k, v] += cnt * var_phi[d, v, k]

            logger.info('All variables updated.')

            Elogtheta = dirichlet_expectation(var_gamma)
            Elogbeta = dirichlet_expectation(var_lambda)
            expElogbeta = numpy.exp(Elogbeta)
            expElogtheta = numpy.exp(Elogtheta)

            # Print topics:
            self.var_lambda = var_lambda
            # pprint(self.show_topics())

            # Evaluate likelihood.
            if (iteration + 1) % self.eval_every == 0:
                prev_likelihood = likelihood
                likelihood = self.eval_likelihood(Elogtheta, Elogbeta)
                logger.info('Likelihood: %.3e', likelihood)
                #if numpy.abs(likelihood - prev_likelihood) / abs(prev_likelihood) < self.threshold:
                #break
        # End of update loop (iterations).

        return var_gamma, var_lambda

    def eval_likelihood(self, Elogtheta, Elogbeta, doc_ids=None):
        """
        Note that this is not strictly speaking a likelihood.

        Compute the expectation of the log conditional likelihood of the data,

            E_q[log p(w_d | theta, beta, A_d)],

        where p(w_d | theta, beta, A_d) is the log conditional likelihood of the data.
        """

        if doc_ids is None:
            docs = self.corpus
        else:
            docs = [self.corpus[d] for d in doc_ids]

        # TODO: check that this is correct.
        likelihood = 0.0
        for d, doc in enumerate(docs):
            ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
            cts = numpy.array([cnt for _, cnt in doc])  # Word counts.
            likelihood_d = 0.0
            for vi, v in enumerate(ids):
                for k in xrange(self.num_topics):
                    likelihood_d += numpy.log(cts[vi]) + Elogtheta[d, k] + Elogbeta[k, v]
            likelihood += likelihood_d

        return likelihood

    # Overriding LdaModel.get_topic_terms.
    def get_topic_terms(self, topicid, topn=10):
        """
        Return a list of `(word_id, probability)` 2-tuples for the most
        probable words in topic `topicid`.
        Only return 2-tuples for the topn most probable words (ignore the rest).
        """
        topic = self.var_lambda[topicid, :]
        topic = topic / topic.sum()  # normalize to probability distribution
        bestn = matutils.argsort(topic, topn, reverse=True)
        return [(id, topic[id]) for id in bestn]







