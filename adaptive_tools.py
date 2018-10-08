#!/usr/bin/env python3.6
# -*-Python-*-

import abc
import asyncio
from contextlib import suppress
import copy
import gzip
import math
import os
import pickle

import adaptive
import toolz


def save(fname, data, compress=True):
    fname = os.path.expanduser(fname)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    _open = gzip.open if compress else open
    with _open(fname, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(fname=None, compress=True):
    fname = os.path.expanduser(fname)
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


class SaverMixin(abc.ABC):

    @abc.abstractmethod
    def get_data(self):
        pass

    @abc.abstractmethod
    def set_data(self):
        pass

    def save(self, fname=None, compress=True):
        fname = get_fname(self, fname)
        data = self.get_data()
        save(fname, data, compress)

    def load(self, fname=None, compress=True):
        fname = get_fname(self, fname)
        with suppress(FileNotFoundError, EOFError):
            data = load(fname, compress)
            self.set_data(data)
            self.refresh()

    def refresh(self):
        pass


class Learner1D(SaverMixin, adaptive.Learner1D):

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data

    def refresh(self):
        self.tell_many(*zip(*self.data.items()))


class Learner2D(SaverMixin, adaptive.Learner2D):

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data

    def refresh(self):
        # Remove points from stack if they already exist
        for point in copy.copy(self._stack):
            if point in self.data:
                self._stack.pop(point)


class AverageLearner(SaverMixin, adaptive.AverageLearner):

    def get_data(self):
        return (self.data, self.npoints, self.sum_f, self.sum_f_sq)

    def set_data(self, data):
        self.data, self.npoints, self.sum_f, self.sum_f_sq = data


class DataSaver(SaverMixin, adaptive.DataSaver):
    def get_data(self):
        return self.learner.get_data(), self.extra_data

    def set_data(self, data):
        learner_data, self.extra_data = data
        self.learner.set_data(learner_data)

    def refresh(self):
        self.learner.refresh()


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
    client.shutdown(targets)
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


def split(lst, n_parts):
    n = math.ceil(len(lst) / n_parts)
    return toolz.partition_all(n, lst)
