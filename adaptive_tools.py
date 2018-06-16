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


def load(fname=None, compress=True):
    _open = gzip.open if compress else open
    with _open(fname, 'rb') as f:
        return pickle.load(f)


def get_fname(learner, fname):
    if fname is not None:
        return fname
    elif hasattr(learner, 'fname'):
        return learner.fname
    else:
        raise Exception('No `fname` supplied.')


class Learner1D(adaptive.Learner1D):

    def save(self, fname=None, compress=True):
        fname = get_fname(self, fname)
        save(fname, self.data, compress)

    def load(self, fname=None, compress=True):
        fname = get_fname(self, fname)
        try:
            self.data = load(fname, compress)
        except (FileNotFoundError, EOFError):
            pass


class Learner2D(adaptive.Learner2D):

    def save(self, fname=None, compress=True):
        fname = get_fname(self, fname)
        save(fname, self.data, compress)

    def load(self, fname=None, compress=True):
        fname = get_fname(self, fname)
        try:
            self.data = load(fname, compress)
            self.refresh_stack()
        except (FileNotFoundError, EOFError):
            pass

    def refresh_stack(self):
        # Remove points from stack if they already exist
        for point in copy.copy(self._stack):
            if point in self.data:
                self._stack.pop(point)


class AverageLearner(adaptive.AverageLearner):

    def save(self, fname=None, compress=True):
        fname = get_fname(self, fname)
        data = (self.data, self.npoints, self.sum_f, self.sum_f_sq)
        save(fname, data, compress)

    def load(self, fname=None, compress=True):
        fname = get_fname(self, fname)
        try:
            data = load(fname, compress)
            self.data, self.npoints, self.sum_f, self.sum_f_sq = data
        except (FileNotFoundError, EOFError):
            pass


class BalancingLearner(adaptive.BalancingLearner):

    def save(self, folder, compress=True):
        for i, learner in enumerate(self.learners):
            fname = get_fname(learner, fname=None)
            learner.save(os.path.join(folder, fname), compress=compress)

    def load(self, folder, compress=True):
        for i, learner in enumerate(self.learners):
            fname = get_fname(learner, fname=None)
            learner.load(os.path.join(folder, fname), compress=compress)


class Runner(adaptive.Runner):

    async def _periodic_saver(self, save_kwargs, interval):
        while self.status() == 'running':
            await asyncio.sleep(interval)
            self.learner.save(**save_kwargs)

    def start_periodic_saver(self, save_kwargs, interval):
        saving_coro = self._periodic_saver(save_kwargs, interval)
        return self.ioloop.create_task(saving_coro)


###################################################
# Running multiple runners, each on its own core. #
###################################################

def run_learner_in_ipyparallel_client(learner, goal, interval, save_kwargs, client_kwargs, targets=None):
    import ipyparallel
    import zmq
    import adaptive
    import asyncio

    client = ipyparallel.Client(context=zmq.Context(), **client_kwargs)
    client[:].use_cloudpickle()
    loop = asyncio.new_event_loop()
    runner = Runner(learner, executor=client.executor(targets), goal=goal, ioloop=loop)
    save_task = runner.start_periodic_saver(save_kwargs, interval)
    loop.run_until_complete(runner.task)
    return learner


default_client_kwargs = dict(profile='pbs', timeout=300, hostname='hpc05')
default_save_kwargs = dict(folder='tmp-{}')


def runners_in_executor(learners, client, nrunners, goal=None, interval=3600,
                        save_kwargs=default_save_kwargs,
                        client_kwargs=default_client_kwargs):
    if goal is None and interval == 0:
        raise Exception('Turn on periodic saving if there is no goal.')

    futs = []
    split_leaners = split(learners, nrunners)
    split_targets = split(range(nrunners, len(client)), nrunners)
    for i, (_learners, targets) in enumerate(zip(split_leaners, split_targets)):
        learner = BalancingLearner(_learners)
        fut = client[i].executor.submit(
            run_learner_in_ipyparallel_client, learner,
            goal, interval, save_kwargs, client_kwargs, targets
        )
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
