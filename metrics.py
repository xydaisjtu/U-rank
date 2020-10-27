import sys
import numpy as np
import json
import pickle


class ListMetrics:
    def __init__(self, pred_scores, rel_list, dataset):
        self.rel_divide = dataset.rel_split
        self.rel_level = dataset.rel_level
        self.pred_map = []
        self.gold_map = []
        self.pred_did = []
        self.pred_rel = []
        self.rerank_pos = []
        self.query_num = len(pred_scores)
        self.old_exam = []
        self.new_exam = []
        self.click = []
        self.debias_click = []
        self.weights = dataset.weights
        self.click_sum = 0
        self.click_prob_sum = 0
        self.click_prob_sum_list = [0.0] * dataset.rank_list_size
        self.click_prob_num_list = [0.0] * dataset.rank_list_size

        self._click = []
        self._click_prob = []

        idx = 0
        for i in range(self.query_num):
            scores = pred_scores[i]
            rel = rel_list[i]
            self.pred_did.append([k + 1 for k in range(len(scores))])
            # print(len(scores))
            pos = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
            self.rerank_pos.append([x + 1 for x in pos])
            self.pred_rel.append([rel[x] for x in pos]) # rel in pred_score order
            self.pred_map.append({})
            self.gold_map.append({})
            self.new_exam.append([])
            if len(pos) != len(dataset.propensity[i]):
                print('DIFF!', i, pos, dataset.propensity[i])
            exam = np.array(dataset.propensity[i])[np.array(pos)]
            old_click = np.array(dataset.clicks[i])[np.array(pos)]
            self.old_exam.append(exam)
            self.click.append(old_click)
            self.debias_click.append(old_click / exam)
            # if i<5:
            #     print('origin click', dataset.clicks[i])
            #     print('score', scores)
            #     print('rerank pos', pos)
            #     print('rerank click', old_click)
            #     print('rerank_rel', self.pred_rel[-1])
            #     print('origin exam', dataset.propensity[i])
            #     print('rerank exam', exam)
            #     print('debias_click', self.debias_click[-1])

            for j in range(len(scores)):
                self.pred_map[-1][pos[j] + 1] = j + 1   # origin pos: rerank pos
                self.gold_map[-1][j + 1] = rel[j]       # origin pos: rel
                feature = np.array(dataset.doc_features[idx + pos[j]])
                self.new_exam[-1].append(1 / np.power(j + 1, max(np.dot(self.weights, feature) + 1, 0)))
            rel = np.array(self.pred_rel[-1])

            exam_prob = np.array(self.new_exam[-1])
            if dataset.relevance_type == 'PBM':
                click_prob = exam_prob * (dataset.epsilon + (1 - dataset.epsilon) * (rel > self.rel_divide))
            else:
                click_prob = exam_prob * (
                            dataset.epsilon + (1 - dataset.epsilon) * (np.power(2, rel) - 1) / (pow(2, 5) - 1))
            self.click_prob_sum += np.sum(click_prob)
            self._click_prob.append(np.sum(click_prob))
            for ii in range(click_prob.shape[0]):
                # print(click_prob.shape)
                self.click_prob_sum_list[ii] += np.sum(click_prob[ii])
                self.click_prob_num_list[ii] += 1
            click = np.zeros_like(click_prob)
            click[np.random.rand(len(click_prob)) < click_prob] = 1
            self.click_sum += np.sum(click)
            self._click.append(np.sum(click))
            idx += len(scores)

        # print(self.pred_did)
        # print(self.pred_map)
        # print(self.gold_map)
        # print(self.pred_rel)
        self.did_num = idx
        self.click_per_query = self.click_sum / self.query_num
        self.click_prob_per_doc = self.click_prob_sum / self.did_num
        self.click_prob_list = [self.click_prob_sum_list[ii] / self.click_prob_num_list[ii] for ii in range(dataset.rank_list_size)]


        self._map = []
        self._ngcg = []
        self._debias_click = []
        self._debias_ndcg = []
        self._debias_precision = []



    def MAP(self):
        res = 0
        for i in range(self.query_num):
            dids = []
            sum = 0
            for (did, rel) in self.gold_map[i].items():
                if rel >= self.rel_divide:
                    dids.append(did)
            # print('dids ', dids)
            rank_list = [self.pred_map[i][x] for x in dids]
            # print('rank_list ', rank_list)
            rank_list.sort()
            for j in range(len(dids)):
                sum += (j + 1) / rank_list[j]
            if len(dids):
                sum /= len(dids)
            # print(sum)
            res += sum
            self._map.append(sum)
        return res / self.query_num

    def DCG(self, rel_list, k):
        r = np.asfarray(rel_list) if len(rel_list) < k else np.asfarray(rel_list)[:k]
        if r.size:
            return np.sum((np.power(2, r) - 1) / np.log2(np.arange(2, r.size + 2)))
        return 0

    def NDCG(self, k):
        res = 0
        for i in range(self.query_num):
            if k == 1:
                self._ngcg.append([])
            iDCG = self.DCG(sorted(self.pred_rel[i], reverse=True), k)
            DCG_at_k = self.DCG(self.pred_rel[i], k)
            if iDCG:
                res += DCG_at_k / iDCG
                self._ngcg[i].append(DCG_at_k / iDCG)
            else:
                self._ngcg[i].append(0)
        return res / self.query_num

    def EER(self, k):
        res = 0
        for i in range(self.query_num):
            n = min(len(self.pred_did[i]), k)
            Ri = [(2 ** x - 1) / 2 ** self.rel_level for x in self.pred_rel[i]][:n]
            RR = []
            for j in range(n):
                RRi = 1
                for t in range(j):
                    RRi *= (1 - Ri[t])
                RRi *= Ri[j]
                RR.append(RRi)
            for j in range(n):
                res += RR[j] / (j + 1)
        return res / self.query_num

    def debias_DCG(self, rel_list, propensity, k):
        r = rel_list[:k]
        if r.size:
            return np.sum((np.power(2, r) - 1) / (np.log2(np.arange(2, r.size + 2)) * propensity[:k]))
        return 0

    def debias_NDCG(self, k):
        res = 0
        for i in range(self.query_num):
            if k == 1:
                self._debias_ndcg.append([])
            rerank = np.array(sorted(range(len(self.debias_click[i])), key=lambda k: self.debias_click[i][k], reverse=True))
            iDCG = self.debias_DCG(self.click[i][rerank], self.old_exam[i][rerank], k)
            DCG_at_k = self.debias_DCG(self.click[i], self.old_exam[i], k)
            if iDCG:
                res += DCG_at_k / iDCG
                self._debias_ndcg[i].append(DCG_at_k / iDCG)
            else:
                self._debias_ndcg[i].append(0)
        return res / self.query_num

    def Click(self, k):
        res = 0
        for i in range(self.query_num):
            if k == 1:
                self._debias_click.append([])
            debias_click = (np.array(self.new_exam[i]) * self.debias_click[i]).tolist()
            res += sum(debias_click[:k])
            self._debias_click[i].append(sum(debias_click[:k]))
        return res / self.query_num

    def Precision(self, k):
        res = 0
        res2 = 0
        for i in range(self.query_num):
            if k == 1:
                self._debias_precision.append([])
            res += sum(self.debias_click[i].tolist()[:k])
            res2 += sum(self.click[i].tolist()[:k])
            self._debias_precision[i].append(sum(self.debias_click[i].tolist()[:k]))
        return res / self.query_num, res2 / self.query_num

    def save(self, path):
        data = {'MAP': self._map,
                'NDCG': self._ngcg,
                'debias_click': self._debias_click,
                'debias_NDCG': self._debias_ndcg,
                'debias_prec': self._debias_precision,
                'click_per_query': self._click,
                'click_prob': self._click_prob}
        output = open(path, 'wb')
        pickle.dump(data, output)
        output.close()


def compute_metrics(labels, scores, dataset, save_file=None):
    metrics = ListMetrics(scores, labels, dataset)
    print('MAP: ', metrics.MAP())
    for i in [1, 3, 5, 10]:
        print('NDCG@' + str(i) + ': ', metrics.NDCG(i), end=' ')
        print('EER@' + str(i) + ': ', metrics.EER(i))
    print()
    for i in [1, 3, 4, 5, 7, 10]:
        print('debias_click@' + str(i) + ': ', metrics.Click(i), end=' ')
        print('debias_NDCG@' + str(i) + ': ', metrics.debias_NDCG(i))
        print('debias_precision@' + str(i) + ': ', metrics.Precision(i))
    print()
    print('click_sum: ', metrics.click_sum, 'click_prob_sum: ', metrics.click_prob_sum)
    print('click_per_query: ', metrics.click_per_query, 'click_prob_per_doc: ', metrics.click_prob_per_doc)
    print()
    if save_file is not None:
        metrics.save(save_file)
        print('save successfully to {}!'.format(save_file))
    return metrics



class Matrics:
    def __init__(self, setting_path, data_path1, data_path2):
        # load setting and context_feature setting
        settings = json.load(open(setting_path + 'settings.json'))
        self.embed_size = settings['embed_size']
        self.rank_list_size = settings['rank_cutoff']
        self.context_feature_type = settings['context_feature_type']
        self.relevance_type = settings['relevance_type']
        self.rel_split = settings['rel_split']
        self.rel_level = settings['rel_level']

        context_feature_setting = json.load(open(setting_path + 'context_feature_setting.json'))
        self.selected_fid = context_feature_setting['selected_fid']
        self.selected_feature_num = context_feature_setting['rel_part_len']
        self.context_feature_len = context_feature_setting['context_feature_len']

        json_fin = open(setting_path + 'click_weight_setting.json')
        click_weight_json = json.load(json_fin)
        self.weights = np.array(click_weight_json['weight'])
        self.eta = click_weight_json['eta']
        self.epsilon = click_weight_json['epsilon']
        json_fin.close()

        print('matrics path: ', setting_path, data_path1, data_path2)




        data1 = open(data_path1 + 'test.trec.init_list', 'r')
        data2 = open(data_path2 + 'test.trec.gold_list', 'r')
        data4 = open(data_path1 + 'test.click_rel')
        data5 = open(data_path1 + 'test.exam')
        if self.context_feature_type == 'doc':
            data3 = open(data_path1 + 'test.doc_context_feature', 'r')
        else:
            data3 = open(data_path1 + 'test.query_context_feature', 'r')

        self.pred_list = data1.readlines()
        self.gold_list = data2.readlines()
        self.click_list = data4.readlines()
        self.exam_list = data5.readlines()
        self.context_feture = data3.readlines()
        self.did_num = len(self.gold_list)
        print('doc num', self.did_num)

        pred_list_len = len(self.pred_list)
        i = 0
        res = 0
        index = 0
        self.query_num = 0
        self.gold_map = []
        self.pred_map = []
        self.pred_did = []
        self.pred_rel = []
        self.old_exam = []
        self.new_exam = []
        self.click = []
        self.debias_click = []
        self.click_sum = 0
        self.click_prob_sum = 0
        self.click_prob_sum_list = [0.0]*self.rank_list_size
        self._click = []
        self._click_prob = []
        self._map = []
        self._ngcg = []
        self._debias_click = []
        self._debias_ndcg = []
        self._debias_precision = []
        while i < pred_list_len:
            self.gold_map.append({})
            self.pred_map.append({})
            self.pred_did.append([])
            new_exam_prob = []
            while (i == index or (
                    i > index and i < pred_list_len and self.pred_list[i].split()[0] == self.pred_list[i - 1].split()[
                0])):
                gold_query_list = self.gold_list[i].strip().split()
                pred_query_list = self.pred_list[i].strip().split()
                self.gold_map[self.query_num][int(gold_query_list[2])] = float(gold_query_list[4]) # did: rel
                self.pred_map[self.query_num][int(pred_query_list[2])] = float(pred_query_list[3]) # did: rerank pos
                self.pred_did[self.query_num].append(int(pred_query_list[2]))
                if self.context_feature_type == 'doc':
                    selected_feature = np.array([float(x) for x in self.context_feture[int(pred_query_list[2])].strip().split()])
                else:
                    selected_feature = np.array([float(x) for x in self.context_feture[self.query_num].strip().split()])
                new_exam_prob.append(1 / np.power(int(pred_query_list[3]), max(np.dot(self.weights, selected_feature) + 1, 0)))
                i = i + 1

            origin_pos = np.array([k - index for k in self.pred_did[self.query_num]])
            index = i
            query_click = [int(k) for k in self.click_list[self.query_num].split()]
            self.click.append(np.array(query_click)[origin_pos])
            old_exam_prob = np.array([float(k) for k in self.exam_list[self.query_num].split()])
            self.old_exam.append(np.array(old_exam_prob)[origin_pos])
            self.debias_click.append(self.click[-1] / self.old_exam[-1])
            rel = [self.gold_map[self.query_num][x] for x in self.pred_did[self.query_num]]
            self.pred_rel.append(rel)
            rel = np.array(rel)
            new_exam_prob = np.array(new_exam_prob)
            self.new_exam.append(new_exam_prob)
            # if i<50:
            #     print('rerank pos', origin_pos)
            #     print('rerank_rel', self.pred_rel[-1])
            #     print('origin exam', old_exam_prob)
            #     print('rerank exam', self.old_exam[-1])
            #     print('rerank click', self.click[-1])
            #     print('debias_click', self.debias_click[-1])

            if self.relevance_type == 'PBM':
                click_prob = new_exam_prob * (self.epsilon + (1 - self.epsilon) * (rel >= self.rel_split))
            else:
                click_prob = new_exam_prob * (self.epsilon + (1 - self.epsilon) * (np.power(2, rel) - 1) / (pow(2, self.rel_level) - 1))
            self.click_prob_sum += np.sum(click_prob)
            self._click_prob.append(np.sum(click_prob))
            for ii in range(self.rank_list_size):
                self.click_prob_sum_list[ii] += np.sum(click_prob[ii])
            click = np.zeros_like(click_prob)
            click[np.random.rand(len(click_prob)) < click_prob] = 1
            self.click_sum += np.sum(click)
            self._click.append(np.sum(click))
            self.query_num += 1
        self.click_per_query = self.click_sum / self.query_num
        self.click_prob_per_doc = self.click_prob_sum / self.did_num
        self.click_prob_list = [self.click_prob_sum_list[ii]/self.query_num for ii in range(self.rank_list_size)]
        print('query num: ', self.query_num)



    def MAP(self):
        res = 0
        for i in range(self.query_num):
            dids = []
            sum = 0
            for (did, rel) in self.gold_map[i].items():
                if rel >= self.rel_split:
                    dids.append(did)
            rank_list = [self.pred_map[i][x] for x in dids]
            rank_list.sort()
            for j in range(len(dids)):
                sum += (j + 1) / rank_list[j]
            if len(dids):
                sum /= len(dids)
            res += sum
            self._map.append(sum)
        return res / self.query_num

    def DCG(self, rel_list, k):
        r = np.asfarray(rel_list) if len(rel_list) < k else np.asfarray(rel_list)[:k]

        if r.size:
            return np.sum((np.power(2, r) - 1) / np.log2(np.arange(2, r.size + 2)))
        return 0

    def NDCG(self, k):
        res = 0
        for i in range(self.query_num):
            if k == 1:
                self._ngcg.append([])
            iDCG = self.DCG(sorted(self.pred_rel[i], reverse=True), k)
            DCG_at_k = self.DCG(self.pred_rel[i], k)
            if iDCG:
                res += DCG_at_k / iDCG
                self._ngcg[i].append(DCG_at_k / iDCG)
            else:
                self._ngcg[i].append(0)
        return res / self.query_num


    def EER(self, k):
        res = 0
        for i in range(self.query_num):
            n = min(len(self.pred_did[i]), k)
            Ri = [(2 ** x - 1) / 2 ** self.rel_level for x in self.pred_rel[i]][:n]
            RR = []
            for j in range(n):
                RRi = 1
                for t in range(j):
                    RRi *= (1 - Ri[t])
                RRi *= Ri[j]
                RR.append(RRi)
            for j in range(n):
                res += RR[j] / (j + 1)
        return res / self.query_num

    def debias_DCG(self, rel_list, propensity, k):
        r = rel_list[:k]
        if r.size:
            return np.sum((np.power(2, r) - 1) / (np.log2(np.arange(2, r.size + 2)) * propensity[:k]))
        return 0

    def debias_NDCG(self, k):
        res = 0
        for i in range(self.query_num):
            if k == 1:
                self._debias_ndcg.append([])
            rerank = np.array(
                sorted(range(len(self.debias_click[i])), key=lambda k: self.debias_click[i][k], reverse=True))
            iDCG = self.debias_DCG(self.click[i][rerank], self.old_exam[i][rerank], k)
            DCG_at_k = self.debias_DCG(self.click[i], self.old_exam[i], k)
            if iDCG:
                res += DCG_at_k / iDCG
                self._debias_ndcg[i].append(DCG_at_k / iDCG)
            else:
                self._debias_ndcg[i].append(0)
        return res / self.query_num

    def Click(self, k):
        res = 0
        for i in range(self.query_num):
            if k == 1:
                self._debias_click.append([])
            debias_click = (np.array(self.new_exam[i]) * self.debias_click[i]).tolist()
            res += sum(debias_click[:k])
            self._debias_click[i].append(sum(debias_click[:k]))
        return res / self.query_num

    def Precision(self, k):
        res = 0
        res2 = 0
        for i in range(self.query_num):
            if k == 1:
                self._debias_precision.append([])
            res += sum(self.debias_click[i].tolist()[:k])
            self._debias_precision[i].append(sum(self.debias_click[i].tolist()[:k]))
            res2 += sum(self.click[i].tolist()[:k])
        return res / self.query_num, res2 / self.query_num

    def save(self, path):
        data = {'MAP':self._map,
                'NDCG':self._ngcg,
                'debias_click':self._debias_click,
                'debias_NDCG':self._debias_ndcg,
                'debias_prec':self._debias_precision,
                'click_per_query':self._click,
                'click_prob':self._click_prob}
        output = open(path, 'wb')
        pickle.dump(data, output)
        output.close()



def calculate_metrics(setting_path, data1_path, data2_path, save_file=None):
    model = Matrics(setting_path, data1_path, data2_path)
    print('MAP: ', model.MAP())
    for i in [1, 3, 5, 10]:
        print('NDCG@' + str(i) + ': ', model.NDCG(i), end=' ')
        print('EER@' + str(i) + ': ', model.EER(i))
    print()
    for i in [1, 3, 4, 5, 7, 10]:
        print('debias_click@' + str(i) + ': ', model.Click(i), end=' ')
        print('debias_NDCG@' + str(i) + ': ', model.debias_NDCG(i))
        print('debias_precision@' + str(i) + ': ', model.Precision(i))
    print()
    print('click_sum: ', model.click_sum, 'click_prob_sum: ', model.click_prob_sum)
    print('click_per_query: ', model.click_per_query, 'click_prob_per_doc: ', model.click_prob_per_doc,
          '\n')
    print('for anlalysis!!!!!!!!!!!============>')
    print(model.click_prob_list)
    if save_file:
        model.save(save_file)


def main():
    # gold_data_path = 'result/test.trec.gold_list'
    # pred_list_path = 'result/'
    #
    # names = ['Initial Ranker:', 'DNN NoCorrection:', 'DNN RandList:', 'DNN oracle estimator:',
    #          'DLA use neither denoise:', 'DLA use click denoise:', 'DLA use prob denoise', 'DLA use both denoise']
    # filenames = ['test.trec.init_list', 'DNNNoCor.ranklist', 'DNNrandList.ranklist', 'DNNoracleIPW.ranklist',
    #              'DLAnone.ranklist', 'DLAclick.ranklist', 'DLAprob.ranklist', 'DLAboth.ranklist']
    #
    # for i in range(len(names)):
    #     print(names[i])
    #     calculate_mattics(pred_path=pred_list_path + filenames[i], gold_path=gold_data_path)
    #
    # path = 'result/test.svm'
    # names = ['Ranking SVM NoCorrection', 'Ranking SVM RandList']
    # filenames = ['NoCor', 'Rand']
    # for i in range(len(names)):
    #     print(names[i])
    #     calculate_mattics(pred_path=path + filenames[i] + '.init_list', gold_path=path + filenames[i] + '.gold_list')

    # for test
    # gold_data_path = 'data/Yahoo/ranker2/train/train.trec.gold_list'
    # pred_list_path = 'data/Yahoo/ranker2/train/train.trec.init_list'
    # rel_level = sys.argv[3] if len(sys.argv) > 3 else 5
    # calculate_metrics(pred_path=pred_list_path, gold_path=gold_data_path, rel_level=int(rel_level))

    # pred_list = [[5, 3, 1, 4, 2], [1, 3, 7, 5, 6, 2, 4]]
    # gold_list = [[3, 3, 3, 0, 0], [3, 0, 3, 0, 3, 0, 3]]
    # compute_metrics(gold_list, pred_list)

    # for SVM/propSVM/DLA
    setting_path = sys.argv[1]
    gold_path = sys.argv[2]
    pred_path = sys.argv[3]
    save_file = sys.argv[4]
    calculate_metrics(setting_path, pred_path, gold_path, save_file)

if __name__ == '__main__':
    main()
