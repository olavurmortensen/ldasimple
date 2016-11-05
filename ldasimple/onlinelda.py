#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from scipy.special import gammaln
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


class OnlineLda(LdaModel):
    """
    """

    def __init__(self, corpus=None, num_topics=100, id2word=None, passes=1,
            threshold=0.001, iterations=10, alpha=None, eta=None,
            offset=1.0, decay=0.5, eval_every=1, random_state=None):

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
        self.passes = passes
        self.num_topics = num_topics
        self.threshold = threshold
        self.alpha = alpha
        self.eta = eta
        self.offset = offset
        self.decay = decay
        self.num_docs = len(corpus)
        self.eval_every = eval_every
        self.random_state = random_state

        self.random_state = get_random_state(random_state)

        if corpus is not None:
            self.inference(corpus)

    def rho(self, t):
        return pow(self.offset + t, -self.decay)

    def inference(self, corpus=None):
        if corpus is None:
            corpus = self.corpus

        logger.info('Starting inference. Training on %d documents.', len(corpus))

        # Initial value of gamma and lambda.
        var_gamma = self.random_state.gamma(100., 1. / 100.,
                (self.num_docs, self.num_topics))
        var_lambda = self.random_state.gamma(100., 1. / 100.,
                (self.num_topics, self.num_terms))
        
        tilde_lambda = var_lambda.copy()

        self.var_lambda = var_lambda
        self.var_gamma = var_gamma

        Elogtheta = dirichlet_expectation(var_gamma)
        Elogbeta = dirichlet_expectation(var_lambda)
        expElogbeta = numpy.exp(Elogbeta)
        expElogtheta = numpy.exp(Elogtheta)

        word_bound = self.word_bound(Elogtheta, Elogbeta)
        theta_bound = self.theta_bound(Elogtheta)
        beta_bound = self.beta_bound(Elogbeta)
        bound = word_bound + theta_bound + beta_bound
        corpus_words = sum(cnt for document in corpus for _, cnt in document)
        perwordbound = bound / corpus_words
        logger.info('Per-word-bound: %.3e. Total bound: %.3e. Word bound: %.3e. theta bound: %.3e. beta bound: %.3e.', perwordbound, bound, word_bound, theta_bound, beta_bound)
        for _pass in xrange(self.passes):
            converged = 0
            for d, doc in enumerate(corpus):
                ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
                cts = numpy.array([cnt for _, cnt in doc])  # Word counts.

                var_phi = numpy.zeros((self.num_terms, self.num_topics))

                for iteration in xrange(self.iterations):
                    lastgamma = var_gamma.copy()

                    # Update phi.
                    for v in ids:
                        for k in xrange(self.num_topics):
                            var_phi[v, k] = expElogtheta[d, k] * expElogbeta[k, v]
                            # var_phi[d, v, k] = numpy.exp(Elogtheta[d, k] + Elogbeta[k, v])
                        # Normalize phi.
                        var_phi[v, :] = var_phi[v, :] / (var_phi[v, :].sum() + 1e-100)

                    # Update gamma.
                    for k in xrange(self.num_topics):
                        var_gamma[d, k] = 0.0
                        var_gamma[d, k] += self.alpha
                        for vi, v in enumerate(ids):
                            var_gamma[d, k] += cts[vi] * var_phi[v, k]

                    Elogtheta = dirichlet_expectation(var_gamma)
                    expElogtheta = numpy.exp(Elogtheta)

                    if iteration > 0:
                        meanchange = numpy.mean(abs(var_gamma - lastgamma))
                        if meanchange < self.threshold:
                            converged += 1
                            break

                # End of update loop (iterations).

                # Update lambda.
                for k in xrange(self.num_topics):
                    #for v in xrange(self.num_terms):
                    for vi, v in enumerate(ids):
                        tilde_lambda[k, v] = self.eta
                        # Get the count of v in doc. If v is not in doc, return 0.
                        #cnt = dict(doc).get(v, 0)
                        tilde_lambda[k, v] += self.num_docs * cts[vi] * var_phi[v, k]

                rhot = self.rho(iteration)
                var_lambda[:, ids] = (1 - rhot) * var_lambda[:, ids] + rhot * tilde_lambda[:, ids]

                Elogbeta = dirichlet_expectation(var_lambda)
                expElogbeta = numpy.exp(Elogbeta)

                self.var_lambda = var_lambda
                self.var_gamma = var_gamma

                # Print topics:
                # pprint(self.show_topics())

                if self.eval_every and d % self.eval_every == 0:
                    word_bound = self.word_bound(Elogtheta, Elogbeta)
                    theta_bound = self.theta_bound(Elogtheta)
                    beta_bound = self.beta_bound(Elogbeta)
                    bound = word_bound + theta_bound + beta_bound
                    perwordbound = bound / corpus_words
                    logger.info('Per-word-bound: %.3e. Total bound: %.3e. Word bound: %.3e. theta bound: %.3e. beta bound: %.3e.', perwordbound, bound, word_bound, theta_bound, beta_bound)

            logger.info('Converged documents: %d/%d.', converged, self.num_docs)

        if not self.eval_every:
            word_bound = self.word_bound(Elogtheta, Elogbeta)
            theta_bound = self.theta_bound(Elogtheta)
            beta_bound = self.beta_bound(Elogbeta)
            bound = word_bound + theta_bound + beta_bound
            perwordbound = bound / corpus_words
            logger.info('Per-word-bound: %.3e. Total bound: %.3e. Word bound: %.3e. theta bound: %.3e. beta bound: %.3e.', perwordbound, bound, word_bound, theta_bound, beta_bound)

        return var_gamma, var_lambda

    def word_bound(self, Elogtheta, Elogbeta, doc_ids=None):
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

        bound = 0.0
        for d, doc in enumerate(docs):
            ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
            cts = numpy.array([cnt for _, cnt in doc])  # Word counts.
            bound_d = 0.0
            for vi, v in enumerate(ids):
                bound_d += cts[vi] * logsumexp(Elogtheta[d, :] + Elogbeta[:, v])
            bound += bound_d

            # Above is the same as:
            #Elogthetad = Elogtheta[d, :]
            #likelihood += numpy.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, id]) for id, cnt in doc)

        return bound

    def theta_bound(self, Elogtheta, doc_ids=None):

        if doc_ids is None:
            docs = self.corpus
        else:
            docs = [self.corpus[d] for d in doc_ids]

        bound = 0.0
        for d in xrange(len(docs)):
            var_gamma_d = self.var_gamma[d, :]
            Elogtheta_d = Elogtheta[d, :]
            # E[log p(theta | alpha) - log q(theta | gamma)]; assumes alpha is a vector
            bound += numpy.sum((self.alpha - var_gamma_d) * Elogtheta_d)
            bound += numpy.sum(gammaln(var_gamma_d) - gammaln(self.alpha))
            bound += gammaln(numpy.sum(self.alpha)) - gammaln(numpy.sum(var_gamma_d))

        return bound

    def beta_bound(self, Elogbeta):
        bound = 0.0
        bound += numpy.sum((self.eta - self.var_lambda) * Elogbeta)
        bound += numpy.sum(gammaln(self.var_lambda) - gammaln(self.eta))
        bound += numpy.sum(gammaln(numpy.sum(self.eta)) - gammaln(numpy.sum(self.var_lambda, 1)))

        return bound

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







