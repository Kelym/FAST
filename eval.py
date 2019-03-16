''' Evaluation of agent trajectories '''

import os
import json
from collections import defaultdict
import networkx as nx
import numpy as np
import pprint; pp = pprint.PrettyPrinter(indent=4)  # NoQA

from env import R2RBatch, ImageFeatures
import utils
from utils import load_datasets, load_nav_graphs
from follower import BaseAgent

import train

from collections import namedtuple

EvalResult = namedtuple(
    "EvalResult", "nav_error, oracle_error, trajectory_steps, "
                  "trajectory_length, success, oracle_success, spl")


class Evaluation(object):
    ''' Results submission format:
        [{'instr_id': string,
          'trajectory':[(viewpoint_id, heading_rads, elevation_rads),]}] '''

    def __init__(self, splits):
        self.error_margin = 3.0
        self.splits = splits
        self.gt = {}
        self.instr_ids = []
        self.scans = []
        self.instructions = {}
        for item in load_datasets(splits):
            self.gt[item['path_id']] = item
            self.scans.append(item['scan'])
            self.instr_ids += [
                '%d_%d' % (item['path_id'], i) for i in range(3)]
            for j,instruction in enumerate(item['instructions']):
                self.instructions['%d_%d' % (item['path_id'], j)] = instruction
        self.scans = set(self.scans)
        self.instr_ids = set(self.instr_ids)
        self.graphs = load_nav_graphs(self.scans)
        self.distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def _score_item(self, instr_id, path):
        ''' Calculate error based on the final position in trajectory, and also
            the closest position (oracle stopping rule). '''
        gt = self.gt[int(instr_id.split('_')[0])]
        start = gt['path'][0]
        assert start == path[0][0], \
            'Result trajectories should include the start position'
        goal = gt['path'][-1]
        final_position = path[-1][0]
        nearest_position = self._get_nearest(gt['scan'], goal, path)
        nav_error = self.distances[gt['scan']][final_position][goal]
        oracle_error = self.distances[gt['scan']][nearest_position][goal]
        trajectory_steps = len(path)-1
        trajectory_length = 0  # Work out the length of the path in meters
        prev = path[0]
        for curr in path[1:]:
            trajectory_length += self.distances[gt['scan']][prev[0]][curr[0]]
            prev = curr

        success = nav_error < self.error_margin
        # check for type errors
        # assert success == True or success == False
        # check for type errors
        oracle_success = oracle_error < self.error_margin
        # assert oracle_success == True or oracle_success == False

        sp_length = 0
        prev = gt['path'][0]
        sp_length = self.distances[gt['scan']][gt['path'][0]][gt['path'][-1]]
        spl = 0.0 if nav_error >= self.error_margin else \
                (float(sp_length) / max(trajectory_length,sp_length))

        return EvalResult(nav_error=nav_error, oracle_error=oracle_error,
                          trajectory_steps=trajectory_steps,
                          trajectory_length=trajectory_length, success=success,
                          oracle_success=oracle_success,
                          spl=spl)

    def score_results(self, results):
        # results should be a dictionary mapping instr_ids to dictionaries,
        # with each dictionary containing (at least) a 'trajectory' field
        # return a dict with key being a evaluation metric
        self.scores = defaultdict(list)
        model_scores = []
        instr_ids = set(self.instr_ids)

        instr_count = 0
        for instr_id, result in results.items():
            if instr_id in instr_ids:
                instr_count += 1
                instr_ids.remove(instr_id)
                eval_result = self._score_item(instr_id, result['trajectory'])

                self.scores['nav_errors'].append(eval_result.nav_error)
                self.scores['oracle_errors'].append(eval_result.oracle_error)
                self.scores['trajectory_steps'].append(
                    eval_result.trajectory_steps)
                self.scores['trajectory_lengths'].append(
                    eval_result.trajectory_length)
                self.scores['success'].append(eval_result.success)
                self.scores['oracle_success'].append(
                    eval_result.oracle_success)
                self.scores['spl'].append(eval_result.spl)
                if 'score' in result:
                    model_scores.append(result['score'])

        assert len(instr_ids) == 0, \
            'Missing %d of %d instruction ids from %s' % (
                len(instr_ids), len(self.instr_ids), ",".join(self.splits))

        assert len(self.scores['nav_errors']) == len(self.instr_ids)
        score_summary = {
            'nav_error': np.average(self.scores['nav_errors']),
            'oracle_error': np.average(self.scores['oracle_errors']),
            'steps': np.average(self.scores['trajectory_steps']),
            'lengths': np.average(self.scores['trajectory_lengths']),
            'success_rate': float(
                sum(self.scores['success']) / len(self.scores['success'])),
            'oracle_rate': float(sum(self.scores['oracle_success'])
                                 / len(self.scores['oracle_success'])),
            'spl': float(sum(self.scores['spl'])) / len(self.scores['spl'])
        }
        if len(model_scores) > 0:
            assert len(model_scores) == instr_count
            score_summary['model_score'] = np.average(model_scores)

        num_successes = len(
            [i for i in self.scores['nav_errors'] if i < self.error_margin])
        # score_summary['success_rate'] = float(num_successes)/float(len(self.scores['nav_errors']))  # NoQA
        assert float(num_successes) / float(len(self.scores['nav_errors'])) == score_summary['success_rate']  # NoQA
        oracle_successes = len(
            [i for i in self.scores['oracle_errors'] if i < self.error_margin])
        assert float(oracle_successes) / float(len(self.scores['oracle_errors'])) == score_summary['oracle_rate']  # NoQA
        # score_summary['oracle_rate'] = float(oracle_successes) / float(len(self.scores['oracle_errors']))  # NoQA
        return score_summary, self.scores

    def score_file(self, output_file):
        ''' Evaluate each agent trajectory based on how close it got to the
        goal location '''
        with open(output_file) as f:
            return self.score_results(json.load(f))

    def score_test_file(self, output_file):
        with open(output_file) as f:
            _d = json.load(f)
            results = {}
            for item in _d:
                results[item['instr_id']] = item
            return self.score_results(results)

    def _path_segments(self,path):
        segs = []
        for i in range(len(path)-1):
            a = min(path[i], path[i+1])
            b = max(path[i], path[i+1])
            segs.append((a,b))
        return set(segs)

    def _inspect(self, instr_id, traj, eval_result):
        results = {'instr_id': instr_id}

        full_path = [p[0] for p in traj]
        path = [full_path[0]]
        for _vpt in full_path:
            if _vpt != path[-1]:
                path.append(_vpt)
        results['path'] = path

        plen = len(path)
        gt = self.gt[int(instr_id.split('_')[0])]
        gt_path = gt['path']
        glen = len(gt_path)

        # 1. No.X starts deviation
        _diff = 0
        while path[_diff] == gt_path[_diff]:
            _diff += 1
            if _diff == plen or _diff == glen: break
        if _diff == plen and _diff == glen:
            _diff = -1 # mark "no deviation"
        results['ontrack'] = _diff

        # 2. Percentage starts deviation
        _percent_diff = float(_diff) / plen
        results['%ontrack'] = _percent_diff

        # 3. # of path segments on gt_path
        psegs = self._path_segments(path)
        gsegs = self._path_segments(gt_path)
        _shared_segs = len(psegs & gsegs)
        results['good_segments'] = _shared_segs

        # 4. % of gt_segment in rollout
        _s_r = float(_shared_segs) / len(psegs)
        results['good/rollout'] = _s_r

        # 5. % of gt_segment got covered
        _s_g = float(_shared_segs) / len(gsegs)
        results['good/gt'] = _s_g

        results['success'] = eval_result.success

        return results

    def inspect_results(self, results):
        inspection = defaultdict(list)
        evals = []
        instr_ids = set(self.instr_ids)
        instr_count = 0
        skipped_count = 0

        if type(results) is list:
            _res = results
            results = {}
            for item in _res:
                results[item['instr_id']] = item

        for instr_id, result in results.items():
            if instr_id in instr_ids:
                instr_count += 1
                instr_ids.remove(instr_id)
                eval_result = self._score_item(instr_id, result['trajectory'])
                evals.append(eval_result)
                res = self._inspect(instr_id, result['trajectory'], eval_result)
                for k,v in res.items():
                    inspection[k].append(v)
            else:
                skipped_count += 1
        print('Inspected', instr_count)
        print('Skipped', skipped_count)
        return inspection,evals


def eval_simple_agents(args):
    ''' Run simple baselines on each split. '''
    img_features = ImageFeatures.from_args(args)
    for split in ['train', 'val_seen', 'val_unseen', 'test']:
        env = R2RBatch(img_features, batch_size=1, splits=[split])
        ev = Evaluation([split])

        for agent_type in ['Stop', 'Shortest', 'Random']:
            outfile = '%s%s_%s_agent.json' % (
                train.RESULT_DIR, split, agent_type.lower())
            agent = BaseAgent.get_agent(agent_type)(env, outfile)
            agent.test()
            agent.write_results()
            score_summary, _ = ev.score_file(outfile)
            print('\n%s' % agent_type)
            pp.pprint(score_summary)


def eval_seq2seq():
    ''' Eval sequence to sequence models on val splits (iteration selected from
    training error) '''
    outfiles = [
        train.RESULT_DIR + 'seq2seq_teacher_imagenet_%s_iter_5000.json',
        train.RESULT_DIR + 'seq2seq_sample_imagenet_%s_iter_20000.json'
    ]
    for outfile in outfiles:
        for split in ['val_seen', 'val_unseen']:
            ev = Evaluation([split])
            score_summary, _ = ev.score_file(outfile % split)
            print('\n%s' % outfile)
            pp.pprint(score_summary)

def eval_outfiles(outfolder):
    splits = ['val_seen','val_unseen']
    for _f in os.listdir(outfolder):
        outfile = os.path.join(outfolder,_f)
        _splits = []
        for s in splits:
            if s in outfile:
                _splits.append(s)
        ev = Evaluation(_splits)
        score_summary, _ = ev.score_file(outfile)
        print('\n', outfile)
        pp.pprint(score_summary)


if __name__ == '__main__':
    from train import make_arg_parser
    utils.run(make_arg_parser(), eval_simple_agents)
    # eval_seq2seq()
