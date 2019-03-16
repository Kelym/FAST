import torch
from torch import optim

import os
import os.path
import time
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse

import utils
from utils import read_vocab, Tokenizer, vocab_pad_idx, timeSince, try_cuda
from utils import module_grad, colorize, filter_param
from env import R2RBatch, ImageFeatures
from model import TransformerEncoder, EncoderLSTM, AttnDecoderLSTM, CogroundDecoderLSTM, ProgressMonitor, DeviationMonitor
from model import SpeakerEncoderLSTM, DotScorer
from follower import Seq2SeqAgent
from scorer import Scorer
import eval

from vocab import SUBTRAIN_VOCAB, TRAINVAL_VOCAB, TRAIN_VOCAB

MAX_INPUT_LENGTH = 80 # TODO make this an argument

max_episode_len = 10
glove_path = 'tasks/R2R/data/train_glove.npy'
action_embedding_size = 2048+128
hidden_size = 512
dropout_ratio = 0.5
learning_rate = 0.0001
weight_decay = 0.0005
FEATURE_SIZE = 2048+128
log_every = 100
save_every = 10000


def get_model_prefix(args, image_feature_list):
    image_feature_name = "+".join(
        [featurizer.get_name() for featurizer in image_feature_list])
    nn = ('{}{}{}{}{}{}'.format(
            ('_ts' if args.transformer else ''),
            ('_sc' if args.scorer else ''),
            ('_mh' if args.num_head > 1 else ''),
            ('_cg' if args.coground else ''),
            ('_pm' if args.prog_monitor else ''),
            ('_sa' if args.soft_align else ''),
            ))
    model_prefix = 'follower{}_{}_{}_{}heads'.format(
        nn, args.feedback_method, image_feature_name, args.num_head)
    if args.use_train_subset:
        model_prefix = 'trainsub_' + model_prefix
    if args.bidirectional:
        model_prefix = model_prefix + "_bidirectional"
    if args.use_pretraining:
        model_prefix = model_prefix.replace(
            'follower', 'follower_with_pretraining', 1)
    return model_prefix


def eval_model(agent, results_path, use_dropout, feedback, allow_cheat=False):
    agent.results_path = results_path
    agent.test(
        use_dropout=use_dropout, feedback=feedback, allow_cheat=allow_cheat)


def train(args, train_env, agent, optimizers, n_iters, log_every=log_every, val_envs=None):
    ''' Train on training set, validating on both seen and unseen. '''

    if val_envs is None:
        val_envs = {}

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

        interval = min(log_every, n_iters-idx)
        iter = idx + interval
        data_log['iteration'].append(iter)
        loss_str = ''

        # Train for log_every interval
        env_name = 'train'
        agent.train(optimizers, interval, feedback=args.feedback_method)
        _loss_str, losses = agent.get_loss_info()
        loss_str += env_name + ' ' + _loss_str
        for k,v in losses.items():
            data_log['%s %s' % (env_name,k)].append(v)

        save_log = []
        # Run validation
        for env_name, (val_env, evaluator) in sorted(val_envs.items()):
            agent.env = val_env
            # Get validation loss under the same conditions as training
            agent.test(use_dropout=True, feedback=args.feedback_method,
                       allow_cheat=True)
            _loss_str, losses = agent.get_loss_info()
            loss_str += ', ' + env_name + ' ' + _loss_str
            for k,v in losses.items():
                data_log['%s %s' % (env_name,k)].append(v)

            agent.results_path = '%s/%s_%s_iter_%d.json' % (
                args.RESULT_DIR, get_model_prefix(
                    args, train_env.image_features_list),
                env_name, iter)

            # Get validation distance from goal under evaluation conditions
            agent.test(use_dropout=False, feedback='argmax')

            print("evaluating on {}".format(env_name))
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
                            agent.write_results()
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
            df_path = '%s/%s_%s_log.csv' % (
                args.PLOT_DIR, get_model_prefix(
                    args, train_env.image_features_list), split_string)
            df.to_csv(df_path)

def setup(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def make_more_train_env(args, train_vocab_path, train_splits):
    setup(args.seed)
    image_features_list = ImageFeatures.from_args(args)
    vocab = read_vocab(train_vocab_path)
    tok = Tokenizer(vocab=vocab)
    train_env = R2RBatch(image_features_list, batch_size=args.batch_size,
                         splits=train_splits, tokenizer=tok)
    return train_env


def make_scorer(args):
    bidirectional = args.bidirectional
    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    feature_size = FEATURE_SIZE
    traj_encoder = try_cuda(SpeakerEncoderLSTM(action_embedding_size, feature_size,
                                          enc_hidden_size, dropout_ratio, bidirectional=args.bidirectional))
    scorer_module = try_cuda(DotScorer(enc_hidden_size, enc_hidden_size))
    scorer = Scorer(scorer_module, traj_encoder)
    if args.load_scorer is not '':
        scorer.load(args.load_scorer)
        print(colorize('load scorer traj '+ args.load_scorer))
    elif args.load_traj_encoder is not '':
        scorer.load_traj_encoder(args.load_traj_encoder)
        print(colorize('load traj encoder '+ args.load_traj_encoder))
    return scorer


def make_follower(args, vocab):
    enc_hidden_size = hidden_size//2 if args.bidirectional else hidden_size
    glove = np.load(glove_path) if args.use_glove else None
    feature_size = FEATURE_SIZE
    Encoder = TransformerEncoder if args.transformer else EncoderLSTM
    Decoder = CogroundDecoderLSTM if args.coground else AttnDecoderLSTM
    word_embedding_size = 256 if args.coground else 300
    encoder = try_cuda(Encoder(
        len(vocab), word_embedding_size, enc_hidden_size, vocab_pad_idx,
        dropout_ratio, bidirectional=args.bidirectional, glove=glove))
    decoder = try_cuda(Decoder(
        action_embedding_size, hidden_size, dropout_ratio,
        feature_size=feature_size, num_head=args.num_head))
    prog_monitor = try_cuda(ProgressMonitor(action_embedding_size,
                            hidden_size)) if args.prog_monitor else None
    bt_button = try_cuda(BacktrackButton()) if args.bt_button else None
    dev_monitor = try_cuda(DeviationMonitor(action_embedding_size,
                            hidden_size)) if args.dev_monitor else None

    agent = Seq2SeqAgent(
        None, "", encoder, decoder, max_episode_len,
        max_instruction_length=MAX_INPUT_LENGTH,
        attn_only_verb=args.attn_only_verb)
    agent.prog_monitor = prog_monitor
    agent.dev_monitor = dev_monitor
    agent.bt_button = bt_button
    agent.soft_align = args.soft_align

    if args.scorer:
        agent.scorer = make_scorer(args)

    if args.load_follower is not '':
        scorer_exists = os.path.isfile(args.load_follower + '_scorer_enc')
        agent.load(args.load_follower, load_scorer=(args.load_scorer is '' and scorer_exists))
        print(colorize('load follower '+ args.load_follower))

    return agent

def make_env_and_models(args, train_vocab_path, train_splits, test_splits):
    setup(args.seed)
    image_features_list = ImageFeatures.from_args(args)
    vocab = read_vocab(train_vocab_path)
    tok = Tokenizer(vocab=vocab)
    train_env = R2RBatch(image_features_list, batch_size=args.batch_size,
                         splits=train_splits, tokenizer=tok) if len(train_splits) > 0 else None
    test_envs = {
        split: (R2RBatch(image_features_list, batch_size=args.batch_size,
                         splits=[split], tokenizer=tok),
                eval.Evaluation([split]))
        for split in test_splits}

    agent = make_follower(args, vocab)
    agent.env = train_env

    return train_env, test_envs, agent


def train_setup(args, train_splits=['train']):
    # val_splits = ['train_subset', 'val_seen', 'val_unseen']
    val_splits = ['val_seen', 'val_unseen']
    #val_splits = ['val_unseen']
    if args.use_test_set:
        val_splits = ['test']
    if args.debug:
        log_every = 5
        args.n_iters = 10
        train_splits = val_splits = ['val_seen']

    vocab = TRAIN_VOCAB

    if args.use_train_subset:
        train_splits = ['sub_' + split for split in train_splits]
        val_splits = ['sub_' + split for split in val_splits]
        vocab = SUBTRAIN_VOCAB

    train_env, val_envs, agent = make_env_and_models(
        args, vocab, train_splits, val_splits)

    if args.use_pretraining:
        pretrain_splits = args.pretrain_splits
        assert len(pretrain_splits) > 0, \
            'must specify at least one pretrain split'
        pretrain_env = make_more_train_env(
            args, vocab, pretrain_splits)

    if args.use_pretraining:
        return agent, train_env, val_envs, pretrain_env
    else:
        return agent, train_env, val_envs

# Test set prediction will be handled separately
# def test_setup(args):
#     train_env, test_envs, encoder, decoder = make_env_and_models(
#         args, TRAINVAL_VOCAB, ['train', 'val_seen', 'val_unseen'], ['test'])
#     agent = Seq2SeqAgent(
#         None, "", encoder, decoder, max_episode_len,
#         max_instruction_length=MAX_INPUT_LENGTH)
#     return agent, train_env, test_envs


def train_val(args):
    ''' Train on the training set, and validate on seen and unseen splits. '''
    if args.use_pretraining:
        agent, train_env, val_envs, pretrain_env = train_setup(args)
    else:
        agent, train_env, val_envs = train_setup(args)

    m_dict = {
            'follower': [agent.encoder,agent.decoder],
            'pm': [agent.prog_monitor],
            'follower+pm': [agent.encoder, agent.decoder, agent.prog_monitor],
            'all': agent.modules()
        }
    if agent.scorer:
        m_dict['scorer_all'] = agent.scorer.modules()
        m_dict['scorer_scorer'] = [agent.scorer.scorer]

    optimizers = [optim.Adam(filter_param(m), lr=learning_rate,
        weight_decay=weight_decay) for m in m_dict[args.grad] if len(filter_param(m))]

    if args.use_pretraining:
        train(args, pretrain_env, agent, optimizers,
              args.n_pretrain_iters, val_envs=val_envs)

    train(args, train_env, agent, optimizers,
          args.n_iters, val_envs=val_envs)

# Test set prediction will be handled separately
# def test_submission(args):
#     ''' Train on combined training and validation sets, and generate test
#     submission. '''
#     agent, train_env, test_envs = test_setup(args)
#     train(args, train_env, agent)
#
#     test_env = test_envs['test']
#     agent.env = test_env
#
#     agent.results_path = '%s/%s_%s_iter_%d.json' % (
#         args.RESULT_DIR, get_model_prefix(args, train_env.image_features_list),
#         'test', args.n_iters)
#     agent.test(use_dropout=False, feedback='argmax')
#     if not args.no_save:
#         agent.write_results()


def make_arg_parser():
    parser = argparse.ArgumentParser()
    ImageFeatures.add_args(parser)
    parser.add_argument("--load_scorer", type=str, default='')
    parser.add_argument("--load_follower", type=str, default='')
    parser.add_argument("--load_traj_encoder", type=str, default='')
    parser.add_argument( "--feedback_method",
            choices=["sample", "teacher", "sample1step","sample2step","sample3step","teacher+sample","recover"], default="sample")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--bidirectional", action='store_true')
    parser.add_argument("--transformer", action='store_true')
    parser.add_argument("--scorer", action='store_true')
    parser.add_argument("--coground", action='store_false')
    parser.add_argument("--prog_monitor", action='store_false')
    parser.add_argument("--dev_monitor", action='store_true')
    parser.add_argument("--bt_button", action='store_true')
    parser.add_argument("--soft_align", action='store_true')
    parser.add_argument("--n_iters", type=int, default=20000)
    parser.add_argument("--num_head", type=int, default=1)
    parser.add_argument("--use_pretraining", action='store_true')
    parser.add_argument("--grad", type=str, default='all')
    parser.add_argument("--pretrain_splits", nargs="+", default=[])
    parser.add_argument("--n_pretrain_iters", type=int, default=50000)
    parser.add_argument("--no_save", action='store_true')
    parser.add_argument("--use_glove", action='store_true')
    parser.add_argument("--attn_only_verb", action='store_true')
    parser.add_argument("--use_train_subset", action='store_true',
        help="use a subset of the original train data for validation")
    parser.add_argument("--use_test_set", action='store_true')
    parser.add_argument("--seed", type=int, default=1)
    return parser


if __name__ == "__main__":
    utils.run(make_arg_parser(), train_val)
