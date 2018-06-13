#!/usr/bin/env python3.6
# -*-Python-*-

import asyncio
from collections import defaultdict
import copy
from functools import partial
from glob import glob
import gzip
import math
import os
import pickle
import re

import adaptive
import toolz


def save(fname, data, compress=True):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    _open = gzip.open if compress else open
    with _open(fname, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(self, fname, compress=True):
    _open = gzip.open if compress else open
    with _open(fname, 'rb') as f:
        return pickle.load(f)


class Learner1D(adaptive.Learner1D):

    def save(self, fname, compress=True):
        save(fname, self.data, compress)

    def load(self, fname, compress=True):
        try:
            self.data = load(self, fname, compress)
        except FileNotFoundError:
            pass


class Learner2D(adaptive.Learner2D):

    def save(self, fname, compress=True):
        save(fname, self.data, compress)

    def load(self, fname, compress=True):
        try:
            self.data = load(self, fname, compress)
            self.refresh_stack()
        except FileNotFoundError:
            pass

    def refresh_stack(self):
        # Remove points from stack if they already exist
        for point in copy.copy(self._stack):
            if point in self.data:
                self._stack.pop(point)


class AverageLearner(adaptive.AverageLearner):

    def save(self, fname, compress=True):
        data = (self.data, self.npoints, self.sum_f, self.sum_f_sq)
        save(fname, data, compress)

    def load(self, fname, compress=True):
        try:
            data = load(self, fname, compress)
            self.data, self.npoints, self.sum_f, self.sum_f_sq = data
        except FileNotFoundError:
            pass


def get_fname(i, fname_pattern=None):
    fname_pattern = 'data_learner_{}.pickle' or fname_pattern
    return fname_pattern.format(f'{i:05d}')


class BalancingLearner(adaptive.BalancingLearner):

    def save(self, folder, fname_pattern=None, compress=True):
        for i, learner in enumerate(self.learners):
            fname = get_fname(i, fname_pattern)
            learner.save(os.path.join(folder, fname), compress=compress)

    def load(self, folder, fname_pattern=None, compress=True):
        for i, learner in enumerate(self.learners):
            fname = get_fname(i, fname_pattern)
            learner.load(os.path.join(folder, fname), compress=compress)


class Runner(adaptive.Runner):

    async def _periodic_saver(self, save_kwargs, interval):
        while self.status() == 'running':
            await asyncio.sleep(interval)
            self.learner.save(**save_kwargs)

    def start_periodic_saver(self, save_kwargs, interval=3600):
        saving_coro = self._periodic_saver(save_kwargs, interval)
        return self.ioloop.create_task(saving_coro)


###################################################
# Running multiple runners, each on its own core. #
###################################################

def run_learner_in_ipyparallel_client(learner, goal, interval, save_kwargs, client_kwargs):
    import ipyparallel
    import zmq
    import adaptive
    import asyncio

    client = ipyparallel.Client(context=zmq.Context(), **client_kwargs)
    client[:].use_cloudpickle()
    loop = asyncio.new_event_loop()
    runner = Runner(learner, executor=client, goal=goal, ioloop=loop)
    save_task = runner.start_periodic_saver(save_kwargs, interval)
    loop.run_until_complete(runner.task)
    return learner


default_client_kwargs = dict(profile='pbs', timeout=300, hostname='hpc05')
default_save_kwargs = dict(fname_pattern=None, folder='tmp-{}', interval=3600)


def split_learners_in_executor(learners, executor, ncores, goal=None, interval,
                               save_kwargs=default_save_kwargs,
                               client_kwargs=default_client_kwargs):
    if goal is None:
        if interval == 0:
            raise Exception('Turn on periodic saving if there is no goal.')
        goal = lambda l: False

    futs = []
    for i, _learners in enumerate(split(learners, ncores)):
        learner = BalancingLearner(_learners)
        save_kwargs['fname_pattern'] = f"{i:05d}_" + save_kwargs['fname_pattern']
        fut = executor.submit(run_learner_in_ipyparallel_client, learner,
                              goal, interval, save_kwargs, client_kwargs)
        futs.append(fut)
    return futs


def combine_learners_from_folders(learners, file_pattern='tmp-*/*',
                                  save_folder=None, save_fname_pattern=None):
    fnames = sorted(glob(file_pattern), key=alphanum_key)
    assert len(fnames) == len(learners)
    for learner, fname in zip(learners, fnames):
        learner.load(*os.path.split(fname))

    if save_folder is not None:
        BalancingLearner(learners).save(save_folder, save_fname_pattern)


######################
# General functions. #
######################

def split(lst, n_parts):
    n = math.ceil(len(lst) / n_parts)
    return toolz.partition_all(n, lst)


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    keys = []
    for _s in re.split('([0-9]+)', s):
        try:
            keys.append(int(_s))
        except:
            keys.append(_s)
    return keys

