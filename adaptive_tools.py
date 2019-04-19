#!/usr/bin/env python3.6
# -*-Python-*-

import asyncio
import math

import adaptive
import toolz


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
    runner = adaptive.Runner(learner, executor=client.executor(targets), goal=goal, ioloop=loop)
    save_task = runner.start_periodic_saving(save_kwargs, interval)
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
