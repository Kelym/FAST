import torch
from torch import optim

import os
import os.path
import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse
import pprint; pp = pprint.PrettyPrinter(indent=4)

import utils
from utils import read_vocab, Tokenizer, vocab_pad_idx, timeSince, try_cuda
from utils import filter_param, module_grad, colorize
from utils import NumpyEncoder
from env import R2RBatch, ImageFeatures
from model import EncoderLSTM, AttnDecoderLSTM
from model import SpeakerEncoderLSTM, DotScorer
from model import SimpleCandReranker
from follower import Seq2SeqAgent
from scorer import Scorer
from make_speaker import make_speaker
import eval

import train
from vocab import SUBTRAIN_VOCAB, TRAINVAL_VOCAB, TRAIN_VOCAB

learning_rate = 0.0001
weight_decay = 0.0005

def get_model_prefix(args, image_feature_list):
    model_prefix = train.get_model_prefix(args, image_feature_list)
    model_prefix.replace('follower','search',1)
    return model_prefix

def _train(args, train_env, agent, optimizers,
          n_iters, log_every=train.log_every, val_envs=None):
    ''' Train on training set, validating on both seen and unseen. '''

    if val_envs is None: val_envs = {}

    print('Training with %s feedback' % args.feedback_method)

    data_log = defaultdict(list)
    start = time.time()

    split_string = "-".join(train_env.splits)

    def make_path(n_iter):
        return os.path.join(
            args.SNAPSHOT_DIR, '%s_%s_iter_%d' % (
                get_model_prefix(args, train_env.image_features_list),
                split_string, n_iter))

    best_metrics = {}
    last_model_saved = {}
    for idx in range(0, n_iters, log_every):
        agent.env = train_env
        if hasattr(agent, 'speaker') and agent.speaker:
            agent.speaker.env = train_env

        interval = min(log_every, n_iters-idx)
        iter = idx + interval
        data_log['iteration'].append(iter)

        # Train for log_every interval
        agent.train(optimizers, interval, feedback=args.feedback_method)
        train_losses = np.array(agent.losses)
        train_loss_avg = np.average(train_losses)
        data_log['train loss'].append(train_loss_avg)
        loss_str = 'train loss: %.4f' % train_loss_avg

        ce_loss = np.average(agent.ce_losses)
        pm_loss = np.average(agent.pm_losses)
        loss_str += ' ce {:.3f}|pm {:.3f}'.format(ce_loss,pm_loss)

        save_log = []
        # Run validation
        for env_name, (val_env, evaluator) in sorted(val_envs.items()):
            agent.env = val_env
            if hasattr(agent, 'speaker') and agent.speaker:
                agent.speaker.env = val_env

            agent.results_path = '%s%s_%s_iter_%d.json' % (
                args.RESULT_DIR, get_model_prefix(
                    args, train_env.image_features_list),
                env_name, iter)

            # Get validation loss under the same conditions as training
            agent.test(use_dropout=True, feedback=args.feedback_method,
                       allow_cheat=True)
            val_losses = np.array(agent.losses)
            val_loss_avg = np.average(val_losses)
            data_log['%s loss' % env_name].append(val_loss_avg)
            loss_str += ', %s loss: %.4f' % (env_name, val_loss_avg)

            ce_loss = np.average(agent.ce_losses)
            pm_loss = np.average(agent.pm_losses)
            data_log['%s ce' % env_name].append(ce_loss)
            data_log['%s pm' % env_name].append(pm_loss)
            loss_str += ' ce {:.3f}|pm {:.3f}'.format(ce_loss,pm_loss)

            # Get validation distance from goal under evaluation conditions
            agent.test(use_dropout=False, feedback='argmax')
            if not args.no_save:
                agent.write_results()
            score_summary, _ = evaluator.score_results(agent.results)

            for metric, val in sorted(score_summary.items()):
                data_log['%s %s' % (env_name, metric)].append(val)
                if metric in ['success_rate']:
                    loss_str += ', %s: %.3f' % (metric, val)

                    key = (env_name, metric)
                    if key not in best_metrics or best_metrics[key] < val:
                        best_metrics[key] = val
                        if not args.no_save:
                            model_path = make_path(iter) + "_%s-%s=%.3f" % (
                                env_name, metric, val)
                            save_log.append(
                                "new best, saved model to %s" % model_path)
                            agent.save(model_path)
                            if key in last_model_saved:
                                for old_model_path in last_model_saved[key]:
                                    if os.path.isfile(old_model_path):
                                        os.remove(old_model_path)
                            #last_model_saved[key] = [agent.results_path] +\
                            last_model_saved[key] = [] +\
                                list(agent.modules_paths(model_path))

        print(('%s (%d %d%%) %s' % (
            timeSince(start, float(iter)/n_iters),
            iter, float(iter)/n_iters*100, loss_str)))
        for s in save_log:
            print(colorize(s))

        if not args.no_save:
            if save_every and iter % save_every == 0:
                agent.save(make_path(iter))

            df = pd.DataFrame(data_log)
            df.set_index('iteration')
            df_path = '%s/trainsearch_%s_%s_log.csv' % (
                args.PLOT_DIR, get_model_prefix(
                    args, train_env.image_features_list), split_string)
            df.to_csv(df_path)

def train_val(args, agent, train_env, val_envs):
    ''' Train on the training set, and validate on seen and unseen splits. '''

    m_dict = {
            'follower': [agent.encoder,agent.decoder],
            'pm': [agent.prog_monitor],
            'follower+pm': [agent.encoder, agent.decoder, agent.prog_monitor],
            'all': agent.modules()
        }
    if agent.scorer:
        m_dict['scorer_all'] = agent.scorer.modules()
        m_dict['scorer_scorer'] = [agent.scorer.scorer]

    if agent.bt_button:
        m_dict['bt_button'] = [agent.bt_button]

    optimizers = [optim.Adam(filter_param(m), lr=learning_rate,
        weight_decay=weight_decay) for m in m_dict[args.grad] if len(filter_param(m))]

    if args.use_pretraining:
        _train(args, pretrain_env, agent, optimizers,
              args.n_pretrain_iters, val_envs=val_envs)

    _train(args, train_env, agent, optimizers,
          args.n_iters, val_envs=val_envs)

def sweep_gamma(args, agent, val_envs, gamma_space):
    ''' Train on training set, validating on both seen and unseen. '''

    print('Sweeping gamma, loss under %s feedback' % args.feedback_method)

    data_log = defaultdict(list)
    start = time.time()

    split_string = "-".join(list(val_envs.keys()))

    def make_path(gamma):
        return os.path.join(
            args.SNAPSHOT_DIR, '%s_gamma_%.3f' % (
                get_model_prefix(args, val_env.image_features_list),
                gamma))

    best_metrics = {}
    for idx,gamma in enumerate(gamma_space):
        agent.scorer.gamma = gamma
        data_log['gamma'].append(gamma)
        loss_str = ''

        save_log = []
        # Run validation
        for env_name, (val_env, evaluator) in sorted(val_envs.items()):
            print("evaluating on {}".format(env_name))
            agent.env = val_env
            if hasattr(agent, 'speaker') and agent.speaker:
                agent.speaker.env = val_env
            agent.results_path = '%s%s_%s_gamma_%.3f.json' % (
                args.RESULT_DIR, get_model_prefix(
                    args, val_env.image_features_list), env_name, gamma)

            # Get validation distance from goal under evaluation conditions
            agent.test(use_dropout=False, feedback='argmax')
            if not args.no_save:
                agent.write_results()
            score_summary, _ = evaluator.score_results(agent.results)
            pp.pprint(score_summary)

            for metric, val in sorted(score_summary.items()):
                data_log['%s %s' % (env_name, metric)].append(val)
                if metric in ['success_rate']:
                    loss_str += ', %s: %.3f' % (metric, val)

                    key = (env_name, metric)
                    if key not in best_metrics or best_metrics[key] < val:
                        save_log.append("new best _%s-%s=%.3f" % (
                                env_name, metric, val))

        idx = idx+1
        print(('%s (%.3f %d%%) %s' % (
            timeSince(start, float(idx)/len(gamma_space)),
            gamma, float(idx)/len(gamma_space)*100, loss_str)))
        for s in save_log:
            print(s)

        df = pd.DataFrame(data_log)
        df.set_index('gamma')
        df_path = '%s%s_%s_log.csv' % (
            args.PLOT_DIR, get_model_prefix(
            args, val_env.image_features_list), split_string)
        df.to_csv(df_path)

def eval_gamma(args, agent, train_env, val_envs):
    gamma_space = np.linspace(0.0, 1.0, num=20)
    gamma_space = [float(g)/100 for g in range(0,101,5)]
    sweep_gamma(args, agent, val_envs, gamma_space)

def run_search(args, agent, train_env, val_envs):
    for env_name, (val_env, evaluator) in sorted(val_envs.items()):
        print("evaluating on {}".format(env_name))
        agent.env = val_env
        if hasattr(agent, 'speaker') and agent.speaker:
            agent.speaker.env = val_env
        agent.results_path = '%s/%s_%s.json' % (
            args.RESULT_DIR, get_model_prefix(
                args, val_env.image_features_list), env_name)
        agent.records_path = '%s/%s_%s_records.json' % (
            args.RESULT_DIR, get_model_prefix(
                args, val_env.image_features_list), env_name)

        agent.test(use_dropout=False, feedback='argmax')
        if not args.no_save:
            agent.write_test_results()
            agent.write_results(results=agent.clean_results, results_path='%s/%s_clean.json'%(args.RESULT_DIR, env_name))
            with open(agent.records_path, 'w') as f:
                json.dump(agent.records, f)
        score_summary, _ = evaluator.score_results(agent.results)
        pp.pprint(score_summary)

def cache(args, agent, train_env, val_envs):
    if train_env is not None:
        cache_env_name = ['train'] + list(val_envs.keys())
        cache_env = [train_env] + [v[0] for v in val_envs.values()]
    else:
        cache_env_name = list(val_envs.keys())
        cache_env = [v[0] for v in val_envs.values()]

    print(cache_env_name)
    for env_name, env in zip(cache_env_name,cache_env):
        #if env_name is not 'val_unseen': continue
        agent.env = env
        if agent.speaker: agent.speaker.env = env
        print("Generating candidates for", env_name)
        agent.cache_search_candidates()
        if not args.no_save:
            with open('cache_{}{}{}{}.json'.format(env_name,'_debug' if args.debug else '', args.max_episode_len, args.early_stop),'w') as outfile:
                json.dump(agent.cache_candidates, outfile, cls=NumpyEncoder)
            with open('search_{}{}{}{}.json'.format(env_name,'_debug' if args.debug else '', args.max_episode_len, args.early_stop),'w') as outfile:
                json.dump(agent.cache_search, outfile, cls=NumpyEncoder)
        score_summary, _ = eval.Evaluation(env.splits).score_results(agent.results)
        pp.pprint(score_summary)

def make_arg_parser():
    parser = train.make_arg_parser()
    parser.add_argument("--max_episode_len", type=int, default=40)
    parser.add_argument("--gamma", type=float, default=0.21)
    parser.add_argument("--mean", action='store_true')
    parser.add_argument("--logit", action='store_true')
    parser.add_argument("--early_stop", action='store_true')
    parser.add_argument("--revisit", action='store_true')
    parser.add_argument("--inject_stop", action='store_true')
    parser.add_argument("--load_reranker", type=str, default='')
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--beam", action='store_true')
    parser.add_argument("--load_speaker", type=str,
            default='./tasks/R2R/experiments/release/speaker_final_release')
    parser.add_argument("--job", choices=['search','sweep','train','cache','test'],default='search')
    return parser

def setup_agent_envs(args):
    if args.job == 'train' or args.job == 'cache':
        train_splits = ['train']
    else:
        train_splits = []
    return train.train_setup(args, train_splits)

def test(args, agent, val_envs):
    test_env = val_envs['test'][0]
    test_env.notTest = False

    agent.env = test_env
    if agent.speaker:
        agent.speaker.env = test_env
    agent.results_path = '%stest.json' % (args.RESULT_DIR)
    agent.test(use_dropout=False, feedback='argmax')
    agent.write_test_results()
    print("finished testing. recommended to save the trajectory again")
    import pdb;pdb.set_trace()

def main(args):
    if args.job == 'test':
        args.use_test_set = True
        args.use_pretraining = False

    # Train a goal button
    #if args.job == 'train' and args.scorer is False:
    #    print(colorize('we need a scorer'))
    #    args.scorer = True

    if args.use_pretraining:
        agent, train_env, val_envs, pretrain_env = setup_agent_envs(args)
    else:
        agent, train_env, val_envs = setup_agent_envs(args)

    agent.search = True
    agent.search_logit = args.logit
    agent.search_mean = args.mean
    agent.search_early_stop = args.early_stop
    agent.episode_len = args.max_episode_len
    agent.gamma = args.gamma
    agent.revisit = args.revisit

    if args.load_reranker != '':
        agent.reranker = try_cuda(SimpleCandReranker(28))
        agent.reranker.load_state_dict(torch.load(args.load_reranker))
    agent.inject_stop = args.inject_stop
    agent.K = args.K
    agent.beam = args.beam


    # Load speaker
    if args.load_speaker is not '':
        speaker = make_speaker(args)
        speaker.load(args.load_speaker)
        agent.speaker = speaker

    if args.job == 'search':
        agent.episode_len = args.max_episode_len
        agent.gamma = args.gamma
        print('gamma', args.gamma, 'ep_len', args.ep_len)
        run_search(args, agent, train_env, val_envs)
    elif args.job == 'sweep':
        for gamma in [float(g)/100 for g in range(0,101,5)]:
            for ep_len in [40]:
                agent.episode_len = ep_len
                agent.gamma = gamma
                print('gamma', gamma, 'ep_len', ep_len)
        #eval_gamma(args, agent, train_env, val_envs)
    elif args.job == 'cache':
        cache(args, agent, train_env, val_envs)
    elif args.job == 'train':
        train_val(args, agent, train_env, val_envs)
    elif args.job == 'test':
        test(args, agent, val_envs)
    else:
        print("no job specified")

if __name__ == "__main__":
    utils.run(make_arg_parser(), main)
