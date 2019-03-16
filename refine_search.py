import pprint; pp = pprint.PrettyPrinter(indent=2)
from env import R2RBatch, ImageFeatures
from utils import Tokenizer, read_vocab, DotDict
from vocab import TRAINVAL_VOCAB, TRAIN_VOCAB
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import numpy as np
from train import make_follower

args = DotDict({
    'image_feature_type': ["mean_pooled"],
    'image_feature_datasets': ["imagenet"],
    'bidirectional': False,
    'use_glove': False,
    'transformer': False,
    'coground': True,
    'num_head': 1,
    'prog_monitor': True,
    'dev_monitor': False,
    'attn_only_verb': False,
    'soft_align': False,
    'scorer': None,
    'load_follower': 'tasks/R2R/experiments/pretrain_cgPm_pertraj/snapshots/follower_cg_pm_sample2step_imagenet_mean_pooled_1heads_train_iter_1900_val_unseen-success_rate=0.478'
})

image_features_list = ImageFeatures.from_args(args)
vocab = read_vocab(TRAIN_VOCAB)
tok = Tokenizer(vocab)
env = R2RBatch(image_features_list, batch_size=256, splits=['train','val_seen','val_unseen'],tokenizer=tok)
env.batch = env.data

from eval import Evaluation
test_envs = {split: (R2RBatch(image_features_list, batch_size=64,splits=[split], tokenizer=tok), Evaluation([split])) for split in ['val_unseen']}

agent = make_follower(args, vocab)

def average(_l):
    return float(sum(_l)) / len(_l)

def load_data(filenames):
    all_data = []
    for fn in filenames:
        with open(fn,'r') as f:
            train_file = json.loads(f.read())
        train_instrs = list(train_file.keys())
        train_data = {}

        for instr_id in train_instrs:
            path_id = int(instr_id.split('_')[0])
            path = env.gt[path_id]['path']
            scanId = env.gt[path_id]['scan']
            new_data = {
                'instr_id': instr_id,
                'gt': env.gt[path_id],
                'goal_viewpointId': env.gt[path_id]['path'][-1],
                'gold_len': len(env.gt[path_id]['path']),
                'expansions': [],
                'world_states': [],
                'actions': [],
                'fathers': [],
                'distance': [],
            }

            _is_gold = []
            ac_counts = []
            bads = []
            deviations = []

            redd = set()

            for i, candidate in enumerate(train_file[instr_id]):
                world_state, action, father, _, _ = candidate

                if (father, action) in redd:
                    continue
                redd.add((father, action)) # A bug in the cache script might append extra

                viewpointId = world_state[1]
                new_data['expansions'].append(candidate)
                new_data['world_states'].append(world_state)
                new_data['actions'].append(action)
                new_data['fathers'].append(father)
                new_data['distance'].append(env.distances[scanId][viewpointId][new_data['goal_viewpointId']])

                ac_counts.append(0 if i == 0 else ac_counts[father] + 1)
                _is_gold.append(i == 0 or
                        (_is_gold[father] and
                            ac_counts[-1] <= new_data['gold_len'] and
                                viewpointId == path[ac_counts[-1]-1]))

                bads.append(0 if _is_gold[-1] else bads[father] + 1)

                _dev = len(env.paths[scanId][viewpointId][path[0]]) - 1
                for i in path:
                    if len(env.paths[scanId][viewpointId][i]) - 1 < _dev:
                        _dev = len(env.paths[scanId][viewpointId][i]) - 1
                deviations.append(_dev)

            # _is_gold checks if father is gold, the proposed action might lead to deviation
            is_gold = [False] * len(new_data['fathers'])
            for i, _is in enumerate(_is_gold):
                if i > 0 and _is:
                    is_gold[new_data['fathers'][i]] = True
            new_data['is_gold'] = is_gold

            new_data['golden_end'] = -1
            for i, _is in enumerate(_is_gold):
                if _is and new_data['actions'][i] == 0 and ac_counts[i] == new_data['gold_len']:
                    new_data['golden_end'] = i

            new_data['ac_counts'] = ac_counts
            new_data['bad'] = bads
            train_data[instr_id] = new_data

        print(fn)
        print('on_track',average([sum(d['is_gold']) for d in train_data.values()]))
        print('on_track ratio',
                average([average(d['is_gold']) for d in train_data.values()]))
        print('oracle',average([any([dis < 3.0 for dis in d['distance']]) for d in train_data.values()]))
        all_data.append(train_data)

    return all_data

[train_data, val_unseen] = load_data(['search_train40True.json', 'search_val_unseen40True.json'])

####

batch_labels = []
valid_points = 0

for training_point in train_data.values():
    labels = training_point['is_gold']
    counts = training_point['ac_counts']
    cand_len = len(labels)
    choice = 1
    x_1 = []
    x_2 = []
    if choice == 1:
        for i in range(cand_len):
            for j in range(cand_len):
                if labels[i] and not labels[j]:
                    x_1.append(i)
                    x_2.append(j)
                    valid_points += 1
    batch_labels.append((x_1, x_2))

print('valid points', valid_points)

###

from utils import filter_param
m_dict = {
    'follower': [agent.encoder, agent.decoder],
}
optimizers = [optim.Adam(filter_param(m), lr=0.0001, weight_decay=0.0005) for m in m_dict['follower'] if len(filter_param(m))]

###

def eval(test_envs, agent):
    for env_name, (val_env, evaluator) in test_envs.items():
        agent.env = val_env
        if hasattr(agent, 'speaker') and agent.speaker:
            agent.speaker.env = val_env
        agent.search = True
        agent.search_logit = True
        agent.search_mean = False
        agent.search_early_stop = True
        agent.episode_len = 40
        agent.gamma = 0
        [m.eval() for m in agent.modules()]
        agent.test(use_dropout=False)
        score_summary, _ = evaluator.score_results(agent.results)
        pp.pprint(score_summary)
    [m.train() for m in m_dict['follower']]

x_1 = []
x_2 = []
agent.load(args.load_follower)
[o.zero_grad() for o in optimizers]
[m.zero_grad() for m in agent.modules()]
ce_loss = 0
pm_loss = 0

batch_size = 64
max_cand_size = 256
ce_criterion = nn.CrossEntropyLoss(ignore_index=-1)
pm_criterion = nn.MSELoss()

agent.env = env
agent.search = True
agent.search_logit = True
agent.search_mean = False
agent.search_early_stop = True
agent.episode_len = 20
agent.gamma = 0
agent.revisit = False
agent.inject_stop = True
agent.K = 10

[m.train() for m in m_dict['follower']]

for epoch in range(30):
    epoch_loss = 0
    for i, (instr_id, cand) in enumerate(train_data.items()):

        cand_len = len(cand['expansions'])
        if cand_len > max_cand_size: cand_len = max_cand_size

        last_obs = env.observe(cand['world_states'][:cand_len], instr_id=instr_id)
        seq, seq_mask, seq_lengths = agent._proc_batch([last_obs[0]])
        ctx, prev_h, prev_c = agent.encoder(seq, seq_lengths)

        new_world_states = env.step(cand['world_states'][:cand_len], cand['actions'][:cand_len], last_obs)
        new_obs = env.observe(new_world_states, instr_id=instr_id)
        new_obs[0] = last_obs[0]

        _a = agent._action_variable(last_obs)[0]
        last_a_t = _a[np.arange(0, cand_len), cand['actions'][:cand_len]].detach()
        last_a_t[0] = agent.decoder.u_begin.view(-1).detach()

        all_u_t, is_valid, _ = agent._action_variable(new_obs)
        all_u_t = all_u_t.detach()
        is_valid = is_valid.byte()
        valid_acs = is_valid.sum(dim=1)

        teacher_ac = agent._teacher_action(new_obs, [False] * cand_len)

        hs = []
        cs = []
        logits = []
        sum_logits = torch.zeros(cand_len).cuda()

        gold_idxes = []
        god_idxes = []

        for t in range(0, cand_len):
            if t > 0:
                dad = cand['fathers'][t]
                prev_h = hs[dad]
                prev_c = cs[dad]

            _h, _c, _tground, _vground, _talpha, _logit, _valpha = \
                    agent.decoder(last_a_t[t:t+1], all_u_t[t:t+1,:valid_acs[t]],
                        None, prev_h, prev_c, ctx, seq_mask)
            hs.append(_h)
            cs.append(_c)
            logits.append(_logit)

            if t > 0:
                sum_logits[t] = sum_logits[dad] + logits[dad][0,cand['actions'][t]]

            pm_score = agent.prog_monitor(prev_h, _c, _vground, _talpha)
            pm_target,_ = agent._progress_target([new_obs[t]], [False], pm_score)

            # Loss
            # choice 1: all actions loss, not just sampling
            #ce_loss += ce_criterion(_logit, teacher_ac[t:t+1])

            if t== 0 or (dad == god[0] and cand['actions'][t] == god[1]):
                # The current branch follows the argmax route
                god_idxes.append(t)
                god = (t, torch.argmax(_logit[0]))

                ce_loss += ce_criterion(_logit, teacher_ac[t:t+1])
                pm_loss += pm_criterion(pm_score, pm_target)

            if cand['is_gold']:
                gold_idxes.append(t)
                teacher_ce += ce_criterion(_logit, teacher_ac[t:t+1])
                teacher_pm += pm_criterion(pm_score, pm_target)

        '''
        # Use the pre-computed pair
        l0 = []
        l1 = []
        for j in range(len(batch_labels[i][0])):
            if batch_labels[i][0][j] < cand_len and batch_labels[i][1][j] < cand_len:
                l0.append(batch_labels[i][0][j])
                l1.append(batch_labels[i][1][j])
        '''
        _len = min(len(gold_idxes),len(god_idxes))
        l0 = gold_idxes[:_len]
        l1 = god_idxes[:_len]

        if len(l1) > 0:
            x_1.append(sum_logits[l0])
            x_2.append(sum_logits[l1])

        if i%batch_size == 0 and len(x_1):
            x1 = torch.cat(x_1, 0)
            x2 = torch.cat(x_2, 0)
            s = x1-x2
            rank_loss = F.relu(1.0 - (s)).mean() # max margin pairwise
            #rank_loss = (-s + torch.log(1 + torch.exp(s))).mean() # RankNet
            ce_loss /= batch_size
            pm_loss /= batch_size
            teacher_ce /= batch_size
            teacher_pm /= batch_size
            loss = ce_loss + pm_loss
            print(i, rank_loss.item(), ce_loss.item(), pm_loss.item(),
                    teacher_ce.item(), teacher_pm.item())
            if i / batch_size == 10:
                eval(test_envs, agent)
            loss.backward()
            epoch_loss += loss.item()
            [o.step() for o in optimizers]

            x_1 = []
            x_2 = []
            ce_loss = 0
            rank_loss = 0
            pm_loss = 0
            teacher_ce = 0
            teacher_pm = 0

            [o.zero_grad() for o in optimizers]
            torch.cuda.empty_cache()


    from datetime import datetime
    fn = datetime.now().strftime('%d-%H-%M')
    agent.save('tasks/R2R/experiments/search_refined_agent/latest')
    agent.save('tasks/R2R/experiments/search_refined_agent/' + fn)

print('Finished Training')
