import  sys
import os
import math
import numpy as np
import json
import matplotlib.pyplot as plt


class ClickModel:
    def __init__(self, data_path=None, ranker='ranker1', set_name='train', eta=0.7, epsilon=0.1, rank_cut=1000,
                 feature_path='./data/Yahoo/filter/', rel_type = 'DLA'):
        self.data_path = data_path
        self.ranker = ranker
        self.eta = eta
        self.epsilon = epsilon
        self.rank_list_size = rank_cut
        self.qids = []
        self.clicks = []
        self.relevance = []
        self.examination = []
        self.query_context_feature = []
        self.doc_context_feature = []
        self.set_name = set_name
        if data_path is None:
            self.embed_size = 0
            return

        # load setting and context_feature setting
        settings = json.load(open(data_path + 'settings.json'))
        self.embed_size = settings['embed_size']
        self.rank_list_size = rank_cut if rank_cut < settings['rank_cutoff'] else settings['rank_cutoff']
        self.context_feature_type = settings['context_feature_type']
        self.rel_split = settings['rel_split']
        self.rel_level = settings['rel_level']
        self.relevance_type = settings['relevance_type']

        context_feature_setting = json.load(open(data_path + 'context_feature_setting.json'))
        self.selected_fid = context_feature_setting['selected_fid']
        self.selected_feature_num = context_feature_setting['rel_part_len']
        self.context_feature_len = context_feature_setting['context_feature_len']
        print('Set selected_fid: ', self.selected_fid)


        if not os.path.exists(self.data_path + 'click_weight_setting.json'):
            # generate w
            self.weights = np.random.uniform(low=-self.eta, high=self.eta, size=10)
            self.weights -= np.mean(self.weights)
            print('Generate Weights: ', self.weights)
            click_weight = {'weight': self.weights.tolist(), 'weight_num': 10, "eta": self.eta, "epsilon": self.epsilon}
            json_fout = open(self.data_path + 'click_weight_setting.json', 'w')
            json.dump(click_weight, json_fout)
            json_fout.close()
        else:
            json_fin = open(self.data_path + 'click_weight_setting.json')
            click_weight_json = json.load(json_fin)
            self.weights = np.array(click_weight_json['weight'])
            print("old weight: ", self.weights)
            json_fin.close()

        # generate clicks(query)
        self.read_context_feature()
        self.read_rel()
        self.clicks = []
        self.examination = []

    def generate_clicks(self):
        if self.context_feature_type == 'query':
            for i in range(len(self.relevance)):
                self.clicks.append([])
                self.examination.append([])
                x = np.array(self.query_context_feature[i])
                for k in range(len(self.relevance[i])):
                    exam_prob = 1 / math.pow(k + 1, max(np.dot(self.weights, x) + 1, 0))
                    self.examination[-1].append(exam_prob)
                    if self.relevance_type == 'PBM':
                        if self.relevance[i][k]:
                            click = 1 if np.random.rand() < exam_prob else 0
                        else:
                            click = 1 if np.random.rand() < self.epsilon else 0
                    else:
                        click = 1 if np.random.rand() < exam_prob*(self.epsilon+(1-self.epsilon)*self.relevance[i][k]) else 0
                    self.clicks[-1].append(click)
        else:
            doc_idx = 0
            for i in range(len(self.relevance)):
                self.clicks.append([])
                self.examination.append([])
                for k in range(len(self.relevance[i])):
                    # print(len(self.relevance), i, len(self.relevance[i]), k, doc_idx)
                    x = np.array(self.doc_context_feature[doc_idx])
                    doc_idx += 1
                    exam_prob = 1 / math.pow(k + 1, max(np.dot(self.weights, x) + 1, 0))
                    self.examination[-1].append(exam_prob)
                    if self.relevance_type == 'PBM':
                        if self.relevance[i][k]:
                            click = 1 if np.random.rand() < exam_prob else 0
                        else:
                            click = 1 if np.random.rand() < self.epsilon else 0
                    else:
                        click = 1 if np.random.rand() < exam_prob*(self.epsilon+(1-self.epsilon)*self.relevance[i][k]) else 0
                    self.clicks[-1].append(click)

        self.save()
        self.plotExam()


    def get_click_prob(self, doc_feature, pos, rel):
        rel = (np.power(2, rel) - 1) / (pow(2, 5) - 1)
        selected_feature = doc_feature[:, self.selected_fid]
        exp = np.dot(selected_feature, self.weights) + 1
        exp[exp < 0] = 0
        exam_prob = 1 / np.power(pos + 1, exp)
        click_prob = exam_prob * (self.epsilon + (1 - self.epsilon) * rel)

        return click_prob, rel, exam_prob


    def sample_clicks(self, query_id, doc_permutation=None):
        if doc_permutation is None:
            doc_permutation = np.arange(len(self.h_doc_context_features[query_id]))

        doc_features = np.array(self.h_doc_context_features[query_id])[doc_permutation,:]

        sampled_clicks = []
        click_probs = []

        if self.context_feature_type =='query':
            for k, doc_feature in enumerate(doc_features):
                exam_prob = 1 / math.pow(k + 1, max(np.dot(self.weights, self.query_context_feature[query_id]) + 1, 0))
                if self.relevance_type == 'PBM':
                    if self.relevance[query_id][doc_permutation[k]]:
                        click = 1 if np.random.rand() < exam_prob else 0
                    else:
                        click = 1 if np.random.rand() < self.epsilon else 0
                        click_probs.append(exam_prob * self.relevance[query_id][doc_permutation[k]] +
                            (1-self.relevance[query_id][doc_permutation[k]])*self.epsilon )
                else:
                    click = 1 if np.random.rand() < exam_prob * (
                                self.epsilon + (1 - self.epsilon) * self.relevance[query_id][doc_permutation[k]]) else 0
                    click_probs.append(exam_prob * (
                            self.epsilon + (1 - self.epsilon) * self.relevance[query_id][doc_permutation[k]]))
                # click = 1 if np.random.rand() < exam_prob * (
                #                 self.epsilon + (1 - self.epsilon) * self.relevance[query_id][doc_permutation[k]]) else 0
                sampled_clicks.append(click)


        if self.context_feature_type =='doc':
            for k, doc_feature in enumerate(doc_features):
                exam_prob = 1 / math.pow(k + 1, max(np.dot(self.weights, doc_feature) + 1, 0))

                if self.relevance_type == 'PBM':
                    if self.relevance[query_id][doc_permutation[k]]:
                        click = 1 if np.random.rand() < exam_prob else 0
                    else:
                        click = 1 if np.random.rand() < self.epsilon else 0
                    click_probs.append(exam_prob * self.relevance[query_id][doc_permutation[k]] +
                            (1-self.relevance[query_id][doc_permutation[k]])*self.epsilon )
                else:
                    click = 1 if np.random.rand() < exam_prob * (
                                self.epsilon + (1 - self.epsilon) * self.relevance[query_id][doc_permutation[k]]) else 0
                    click_probs.append(exam_prob * (
                                self.epsilon + (1 - self.epsilon) * self.relevance[query_id][doc_permutation[k]]))

                sampled_clicks.append(click)

        return sampled_clicks, click_probs

    def read_context_feature(self):
        self.query_context_feature = []
        self.doc_context_feature = []
        self.category = []
        print('Reading ' + self.set_name + ' features...', end=' ')
        context_feature_fin = open(self.data_path + self.ranker + '/' + self.set_name + '/' + self.set_name + '.query_context_feature')
        for line in context_feature_fin:
            arr = line.strip().split(' ')
            self.query_context_feature.append([float(x) for x in arr])
        context_feature_fin.close()

        context_feature_fin = open(self.data_path + self.ranker + '/' + self.set_name + '/' + self.set_name + '.doc_context_feature')
        for line in context_feature_fin:
            arr = line.strip().split(' ')
            self.doc_context_feature.append([float(x) for x in arr])
        context_feature_fin.close()
        print('Done')


    def read_rel(self):
        print('Reading ' + self.set_name + ' relevance...', end=' ')
        self.relevance = []
        self.qids = []
        self.h_doc_context_features = []
        head = 0
        tail = 0
        rel_fin = open(self.data_path + self.ranker + '/' + self.set_name + '/' + self.set_name + '.weights')
        for line in rel_fin:
            arr = line.strip().split(' ')
            self.qids.append(arr[0])
            self.relevance.append([])
            for x in arr[1:]:
                if self.relevance_type == 'PBM':
                    self.relevance[-1].append(1 if float(x) > self.rel_split else 0)
                else:
                    self.relevance[-1].append((pow(2, float(x)) - 1) / (pow(2, self.rel_level) - 1))
            tail = tail + len(arr[1:])
            self.h_doc_context_features.append(self.doc_context_feature[head:tail])
            head = tail
        rel_fin.close()

    def save(self):
        print('Saving ' + self.set_name + ' clicks & examinations...', end=' ')
        click_fout = open(self.data_path + self.ranker + '/' + self.set_name + '/' + self.set_name + '.click', 'w')
        examination_fout = open(self.data_path + self.ranker + '/' + self.set_name + '/' + self.set_name + '.examination', 'w')
        nums = 0
        bid_fout = open(self.data_path + self.ranker + '/' + self.set_name + '/' + self.set_name + '.bid', 'w')
        for i in range(len(self.relevance)):
            click_line = str(self.qids[i]) + ' ' + ' '.join(str(t) for t in self.clicks[i]) + '\n'
            exam_line = str(self.qids[i]) + ' ' + ' '.join(str(t) for t in self.examination[i]) + '\n'
            bids = (np.random.rand(len(self.relevance[i])) * 9 + 1).tolist()
            nums += len(bids)
            bid_line = str(self.qids[i]) + ' ' + ' '.join(str(t) for t in bids) + '\n'
            click_fout.write(click_line)
            examination_fout.write(exam_line)
            bid_fout.write(bid_line)
        click_fout.close()
        examination_fout.close()
        bid_fout.close()
        print('Done', nums)

    def plotExam(self):
        center_fin = open(self.data_path + self.ranker + '/' + self.set_name + '/' + self.set_name + '.center', 'r')
        center = []
        for line in center_fin:
            center.append([float(x) for x in line.strip().split()])
        self.center = np.array(center)
        exam = []
        for i in range(self.center.shape[0]):
            exam.append(np.zeros(10))
            for k in range(10):
                exam[-1][k] = 1 / np.power(k + 1, max(np.dot(self.weights, self.center[i, :]) + 1, 0))
        # print(exam)
        pos = np.arange(0, 10)
        color = ['r', 'g', 'b', 'c', 'm', 'k', 'w', 'y', 'burlywood', 'chartreuse']
        plt.figure()
        for i, x in enumerate(exam):
            plt.plot(pos, x, color=color[i])
        plt.savefig(self.data_path + self.ranker + '/' + self.set_name +'/exam_curve.png')


def main():
    DATA_PATH = 'data/istella/'
    rel_type = 'DLA'
    # ETA = 0.5 if len(sys.argv) < 4 else float(sys.argv[3])
    ETA = 1.5
    EPSILION = 0.1
    # model = ClickModel(data_path=DATA_PATH, eta=ETA)

    # model.sample_clicks(20)

    # generate all clicks
    for ranker_name in ['ranker1', 'ranker2']:
        for name in ['train', 'valid', 'test']:
            model = ClickModel(data_path=DATA_PATH, ranker=ranker_name, set_name=name,
                               eta=ETA, epsilon=EPSILION, rel_type=rel_type)
            model.generate_clicks()
            # model.save()


if __name__ == '__main__':
    main()
