import numpy as np
import json
import random
import os
import sys


class Raw_data:
    def __init__(self, data_path=None, set_name=None, rank_cut=100000):
        self.data_path = data_path
        self.set_name = set_name
        self.feature_path = data_path + 'filter/'
        settings = json.load(open(self.data_path + 'settings.json'))
        self.embed_size = settings['embed_size']
        self.rel_split = settings['rel_split']
        self.rel_level = settings['rel_level']
        self.relevance_type = settings['relevance_type']
        self.rank_list_size = rank_cut if rank_cut < settings['rank_cutoff'] else settings['rank_cutoff']  # k_max

        context_feature_setting = json.load(open(data_path + 'context_feature_setting.json'))
        self.selected_fid = context_feature_setting['selected_fid']
        self.selected_feature_num = context_feature_setting['rel_part_len']
        self.context_feature_len = context_feature_setting['context_feature_len']

        json_fin = open(data_path + 'click_weight_setting.json')
        click_weight_json = json.load(json_fin)
        self.weights = np.array(click_weight_json['weight'])
        self.eta = click_weight_json['eta']
        self.epsilon = click_weight_json['epsilon']
        json_fin.close()

        ## documnet features
        self.features = []  # d_max*emb_size
        feature_fin = open(self.feature_path + set_name + '.txt')
        for line in feature_fin:
            arr = line.strip().split(' ')
            self.features.append([0.0 for _ in range(self.embed_size)])
            for x in arr[2:]:
                arr2 = x.split(':')
                self.features[-1][int(arr2[0]) - 1] = float(arr2[1])
        feature_fin.close()
        self.item_num = len(self.features)
        self.item_field_M = len(self.features[-1])
        self.features = np.array(self.features)
        print('the total number of documents is ' + str(self.item_num), self.item_field_M)


        ##q_max
        self.qids = []
        flatten_qids = []
        self.dids = []
        flatten_dids = []
        flatten_poss = []
        self.clicks = []
        flatten_clicks = []
        self.bids = []
        flatten_bids = []
        self.len_list = []
        self.relavance_labels = []
        flatten_rls = []
        self.propensity = []
        flatten_prop = []
        self.query_features = []
        self.doc_features = []
        flatten_user_features = []
        flatten_item_features = []

        if data_path is None:
            print('error')
            return
        rep = 0
        for ranker_name in ['ranker1', 'ranker2']:
            init_list_fin = open(self.data_path + ranker_name + '/' +set_name + '/' + set_name + '.init_list')
            init_list = init_list_fin.readlines()
            for i, line in enumerate(init_list):
                arr = line.strip().split(' ')
                self.qids.append(i + rep * len(init_list))
                self.dids.append([int(x) for x in arr[1:][:self.rank_list_size]])
                self.len_list.append(len(arr[1:]))
                flatten_qids += [self.qids[-1] ]* len(arr[1:])
                flatten_dids += self.dids[-1]
                # for x in arr[1:][:self.rank_list_size]:
                #     flatten_user_features.append(np.array(self.query_features)[i])
                #     flatten_item_features.append(np.array(self.features)[int(x)])
                flatten_poss += [i for i in range(len(arr[1:]))]
            init_list_fin.close()
            rep += 1
            print(ranker_name, 'dids ', len(flatten_dids))

            # print('Reading ' + self.set_name + ' features...', end=' ')
            context_feature_fin = open(
                self.data_path + ranker_name + '/' + self.set_name + '/' + self.set_name + '.query_context_feature')
            for line in context_feature_fin:
                arr = line.strip().split(' ')
                self.query_features.append([float(x) for x in arr])
            context_feature_fin.close()
            print(ranker_name, 'query feature ', len(self.query_features))

            doc_context_feature_fin = open(
                self.data_path + ranker_name + '/' + self.set_name + '/' + self.set_name + '.doc_context_feature')
            for line in doc_context_feature_fin:
                arr = line.strip().split(' ')
                self.doc_features.append([float(x) for x in arr])
            doc_context_feature_fin.close()
            print(ranker_name, 'doc feature ', len(self.doc_features))

            gold_weight_fin = open(self.data_path + ranker_name + '/' + set_name + '/' + set_name + '.weights')
            for line in gold_weight_fin:
                self.relavance_labels.append([float(x) for x in line.strip().split(' ')[1:][:self.rank_list_size]])
                flatten_rls += self.relavance_labels[-1]
            gold_weight_fin.close()
            print(ranker_name, 'weights ', len(flatten_rls))

            click_fin = open(self.data_path + ranker_name + '/' + set_name + '/' + set_name + '.click')
            for line in click_fin:
                self.clicks.append([int(x) for x in line.strip().split(' ')[1:]])
                flatten_clicks += self.clicks[-1]
            # for x in line.strip().split(' ')[1:]:
            #     self.clicks.append(int(x))
            click_fin.close()
            print(ranker_name, 'clicks ', len(flatten_clicks))

            bid_fin = open(self.data_path + ranker_name + '/' + set_name + '/' + set_name + '.bid')
            for line in bid_fin:
                self.bids.append([float(x) for x in line.strip().split(' ')[1:]])
                flatten_bids += self.bids[-1]
            # for x in line.strip().split(' ')[1:]:
            #     self.clicks.append(int(x))
            bid_fin.close()
            print(ranker_name, 'bids ', len(flatten_bids))

            exam_fin = open(self.data_path + ranker_name + '/' + set_name + '/' + set_name + '.examination')
            for line in exam_fin:
                self.propensity.append([float(x) for x in line.strip().split(' ')[1:]])
                flatten_prop += self.propensity[-1]
            exam_fin.close()
            print(ranker_name, 'exams ', len(flatten_prop))


        self.flatten_dids = flatten_dids
        self.flatten_poss = flatten_poss
        self.doc_features = np.array(self.doc_features)

        self.data = np.ones([len(flatten_dids), 7], dtype=np.int)  # (qid, did, pos, exam, click)
        self.data[:, 0] = np.array(flatten_qids)
        self.data[:, 1] = np.array(flatten_dids)
        self.data[:, 2] = np.array(flatten_poss)
        self.data[:, 3] = np.array(flatten_prop)
        self.data[:, 4] = np.array(flatten_clicks)
        self.data[:, 5] = np.array(flatten_rls)
        self.data[:, 6] = np.array(flatten_bids)

        self.user_num = len(self.len_list)
        self.user_field_M = len(self.query_features[-1])
        self.query_features = np.array(self.query_features)
        print('the total number of quries is ' + str(self.user_num), self.user_field_M)
        self.pos_neg_ratio = np.sum(self.data[:, 4])/(self.data.shape[0] - np.sum(self.data[:, 4]))
        print('pos_neg_ratio is {}'.format(self.pos_neg_ratio))


def read_data(data_path, set_name, rank_cut=100000):
    data = Raw_data(data_path, set_name, rank_cut)
    return data



def generate_ranklist_by_scores(data, rerank_scores):
    if len(rerank_scores) != len(data.initial_list):
        raise ValueError("Rerank ranklists number must be equal to the initial list,"
                         " %d != %d." % (len(rerank_scores)), len(data.initial_list))
    qid_list_map = {}
    for i in range(len(data.qids)):
        scores = rerank_scores[i]
        rerank_list = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        if len(rerank_list) != len(data.initial_list[i]):
            raise ValueError("Rerank ranklists length must be equal to the gold list,"
                             " %d != %d." % (len(rerank_scores[i]), len(data.initial_list[i])))
        # remove duplicate and organize rerank list
        index_list = []
        index_set = set()
        for j in rerank_list:
            # idx = len(rerank_lists[i]) - 1 - j if reverse_input else j
            idx = j
            if idx not in index_set:
                index_set.add(idx)
                index_list.append(idx)
        for idx in range(len(rerank_list)):
            if idx not in index_set:
                index_list.append(idx)
        # get new ranking list
        qid = data.qids[i]
        did_list = []
        for idx in index_list:
            ni = data.initial_list[i][idx]
            ns = scores[idx]
            if ni >= 0:
                did_list.append((data.dids[ni], ns))
        qid_list_map[qid] = did_list
    return qid_list_map


def output_ranklist(data, rerank_scores, output_path, file_name='test'):
    qid_list_map = generate_ranklist_by_scores(data, rerank_scores)
    fout = open(output_path + file_name + '.ranklist', 'w')
    for qid in data.qids:
        for i in range(len(qid_list_map[qid])):
            fout.write(qid + ' Q0 ' + qid_list_map[qid][i][0] + ' ' + str(i + 1)
                       + ' ' + str(qid_list_map[qid][i][1]) + ' RankLSTM\n')
    fout.close()


def test():
    DATA_PATH = './data/Yahoo3/'
    # DATA_PATH2 = sys.argv[2] + '/'
    # for name in ['train', 'valid', 'test']:
    #     model = Raw_data(data_path=DATA_PATH2, data_path2=DATA_PATH2, set_name=name)

    model = Raw_data(data_path=DATA_PATH, set_name='test')
    print(model.len_list[0], model.len_list[1], model.len_list[2])
    for i in range(20):
        print(model.clicks[i])
        print(' ')


# python ./data_utils.py data/final_data1 data/final_data2


if __name__ == '__main__':
    test()
