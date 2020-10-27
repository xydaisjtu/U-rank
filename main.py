from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tqdm import tqdm
import pickle

import math
import os
import random
import sys
import time
from typing import List, Any

import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf
import json
from metrics import compute_metrics

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils
import Urank as CM
from click_model import ClickModel
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score



# rank list size should be read from data
# tf.app.flags.DEFINE_string("data_dir", "./UPRR_final/data/Yahoo/", "Data directory")
# tf.app.flags.DEFINE_string("train_dir", "./result/Yahoo_train/", "Training directory")
# tf.app.flags.DEFINE_string("store_dir", "./ttest/Yahoo/UPRR_NN/", "Training directory")
tf.app.flags.DEFINE_string("data_dir", "./data/MSLR10K/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./result/MSLR10K_train/", "Training directory")
tf.app.flags.DEFINE_string("store_dir", "./ttest/MSLR10K/UPRR_NN", "Training directory")
# tf.app.flags.DEFINE_string("data_dir", "./data/istella/", "Data directory")
# tf.app.flags.DEFINE_string("train_dir", "./result/istella/", "Training directory")
# tf.app.flags.DEFINE_string("store_dir", "./ttest/istella/UPRR_NN", "Training directory")
tf.app.flags.DEFINE_string("test_dir", "./result/test/", "Directory for output test results.")
tf.app.flags.DEFINE_string("hparams", "", "Hyper-parameters for models.")

tf.app.flags.DEFINE_string("train_stage", 'ranker',
                           "traing stage.")

tf.app.flags.DEFINE_integer("batch_size", 512,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("np_random_seed", 1385,
                            "random seed for numpy")
tf.app.flags.DEFINE_integer("tf_random_seed", 20933,
                            "random seed for tensorflow")
tf.app.flags.DEFINE_integer("emb_size", 10,
                            "Embedding to use during training.")
tf.app.flags.DEFINE_integer("train_list_cutoff", 10,
                            "The number of documents to consider in each list during training.")
tf.app.flags.DEFINE_integer("max_train_iteration", 0,
                            "Limit on the iterations of training (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 1,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for decoding data.")
tf.app.flags.DEFINE_boolean("decode_train", False,
                            "Set to True for decoding training data.")
tf.app.flags.DEFINE_boolean("decode_valid", False,
                            "Set to True for decoding valid data.")
# To be discarded.
tf.app.flags.DEFINE_boolean("feed_previous", False,
                            "Set to True for feed previous internal output for training.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Set to True for test program.")

FLAGS = tf.app.flags.FLAGS

tf.set_random_seed(FLAGS.tf_random_seed)
np.random.seed(FLAGS.np_random_seed)


def create_model(session, data_set, forward_only, ckpt = None):
    """Create model and initialize or load parameters in session."""
    print('create model', data_set.user_field_M, data_set.item_field_M)

    model = CM.URANK(data_set.rank_list_size, data_set.user_field_M, data_set.item_field_M, FLAGS.emb_size,
                    FLAGS.batch_size, FLAGS.hparams,
                    forward_only, train_stage=FLAGS.train_stage)

    print(ckpt)
    if not ckpt:
        print('reloading')
        if FLAGS.train_stage == 'ranker':
            # ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir + 'click/learning_rate=5e-4_batch_2048') #mslr
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir + 'click/learning_rate=7e-4_batch_2048')  # istella
        if FLAGS.train_stage == 'click':
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir + 'click/')

    if ckpt:
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())

    if not forward_only:
        if model.train_stage == 'click':
            model.separate_gradient_update()

        model.global_gradient_update()
        tf.summary.scalar('Loss', tf.reduce_mean(model.loss))
        # tf.summary.scalar('Gradient Norm', model.norm)
        tf.summary.scalar('Learning Rate', model.learning_rate)
        tf.summary.scalar('Final Loss', tf.reduce_mean(model.loss))
    model.summary = tf.summary.merge_all()

    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            session.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)

    initialize_op = tf.variables_initializer(uninitialized_vars)
    session.run(initialize_op)

    return model


def train(store_path):
    # Prepare data.
    print("Reading data in %s" % FLAGS.data_dir)

    train_set = data_utils.read_data(FLAGS.data_dir, 'train', FLAGS.train_list_cutoff)
    valid_set = data_utils.read_data(FLAGS.data_dir, 'test', FLAGS.train_list_cutoff)
    print("Rank list size %d" % train_set.rank_list_size)
    click_model_1 = ClickModel(data_path=FLAGS.data_dir, ranker='ranker1', set_name='test', eta=0.5)
    click_model_2 = ClickModel(data_path=FLAGS.data_dir, ranker='ranker2', set_name='test', eta=0.5)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Create model.
        print("Creating model...")
        # model = create_model(sess, train_set, None, ckpt = tf.train.get_checkpoint_state(check_point_dir))
        model = create_model(sess, train_set, False)
        if FLAGS.train_stage == 'ranker':
            check_point_dir = FLAGS.train_dir + FLAGS.train_stage + "/" + str(
                    FLAGS.hparams) + '_batch_' + str(FLAGS.batch_size)
        if FLAGS.train_stage == 'click':
            check_point_dir = FLAGS.train_dir + FLAGS.train_stage + "/" + str(
                    FLAGS.hparams)+ '_batch_' + str(FLAGS.batch_size)
        print(check_point_dir)

        # Create tensorboard summarizations.
        train_writer = tf.summary.FileWriter(FLAGS.train_dir + '/train_log',
                                             sess.graph)
        valid_writer = tf.summary.FileWriter(FLAGS.train_dir + '/valid_log')
        if not os.path.exists(check_point_dir):
            # print('mkdir:', check_point_dir)
            print('mkdir', check_point_dir)
            os.makedirs(check_point_dir)

        if FLAGS.train_stage == 'click':
            # Training of click debias model
            step_time, loss = 0.0, 0.0
            current_step = 0
            previous_losses = []
            best_loss = None
            while True:
                # Get a batch and make a step.
                start_time = time.time()
                input_feed, data = model.get_batch_for_click(train_set)
                step_loss, _, summary, auc, acc, click, rel_err = model.click_step(sess, input_feed, False, click_model_1, data, (current_step+1) % FLAGS.steps_per_checkpoint == 0)
                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                loss += step_loss / FLAGS.steps_per_checkpoint
                current_step += 1
                train_writer.add_summary(summary, current_step)

                # Once in a while, we save checkpoint, print statistics, and run evals.
                if current_step % FLAGS.steps_per_checkpoint == 0:

                    # Print statistics for the previous epoch.
                    # loss = math.exp(loss) if loss < 300 else float('inf')
                    print(tf.convert_to_tensor(model.global_step).eval(),
                          tf.convert_to_tensor(model.learning_rate).eval(), step_time, loss)
                    print("global step %d learning rate %.4f step-time %.2f loss "
                          "%.2f auc %.4f acc %.4f rel_err %.4f" % (tf.convert_to_tensor(model.global_step).eval(),
                                                      tf.convert_to_tensor(model.learning_rate).eval(), step_time, loss,
                                                      auc, acc, rel_err))
                    # train_writer.add_summary({'step-time':step_time, 'loss':loss}, current_step)

                    # Decrease learning rate if no improvement was seen over last 3 times.
                    if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                        sess.run(model.learning_rate_decay_op)
                    previous_losses.append(loss)

                    # Validate model
                    it = 0
                    count_batch = 0.0
                    valid_loss = 0
                    all_acc = 0
                    rel_error = 0
                    while it < valid_set.item_num / model.batch_size:
                        input_feed, data = model.get_batch_for_click(valid_set)
                        v_loss, summary, auc, acc, click, step_rel_error = model.click_step(sess, input_feed, True, click_model_1, data, True)
                        it += FLAGS.batch_size
                        valid_loss += v_loss
                        rel_error += step_rel_error
                        all_acc += acc
                        count_batch += 1.0
                    valid_writer.add_summary(summary, current_step)
                    valid_loss /= count_batch
                    acc = all_acc / count_batch
                    # valid_loss = math.exp(valid_loss) if valid_loss < 300 else float('inf')

                    rel_error = rel_error / count_batch
                    print("  eval: loss %.2f auc %.4f acc %.4f rel_err %.4f" % (valid_loss, auc, acc, rel_error))

                    checkpoint_path = check_point_dir + "/model.ckpt"
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                    if loss == float('inf'):
                        break

                    step_time, loss = 0.0, 0.0
                    sys.stdout.flush()

                    if FLAGS.max_train_iteration > 0 and current_step > FLAGS.max_train_iteration:
                        break

        # Training of the ranker
        if FLAGS.train_stage == 'ranker':

            # Training of ranking model
            step_time, loss = 0.0, 0.0
            current_step = 0
            previous_losses = []
            all_scores = []
            pre_scores = []
            best_loss = None
            while True:
                # Get a batch and make a step.
                start_time = time.time()
                input_feed, labels , clicks = model.get_batch_for_ranker(train_set)
                labels, scores, step_loss, summary, _ = model.ranker_step(sess, input_feed, labels, clicks, forward_only=False)
                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                loss += step_loss / FLAGS.steps_per_checkpoint
                current_step += 1
                train_writer.add_summary(summary, current_step)

                # Once in a while, we save checkpoint, print statistics, and run evals.
                if (current_step < 40) or (current_step % FLAGS.steps_per_checkpoint == 0):
                    print(tf.convert_to_tensor(model.global_step).eval(),
                          tf.convert_to_tensor(model.learning_rate).eval(), step_time, loss)
                    print("global step %d learning rate %.4f step-time %.2f loss "
                          "%.8f" % (tf.convert_to_tensor(model.global_step).eval(),
                                    tf.convert_to_tensor(model.learning_rate).eval(), step_time, loss))
                    # print('on train set')
                    # compute_metrics(labels, scores, train_set)
                    previous_losses.append(loss)

                    # Validate model
                    it = 0
                    count_batch = 0.0
                    pre_scores = all_scores
                    click_count, click_prob_count, origin_click_count, origin_click_prob_count = 0.0, 0.0, 0.0, 0.0
                    all_labels, all_scores, all_rank, all_deltas = [], [], [], []
                    while it < valid_set.user_num:
                        input_feed, labels, clicks = model.get_batch_for_ranker_by_index(valid_set, it)
                        labels, scores = model.ranker_step(sess, input_feed, labels, clicks, forward_only=True)
                        # labels, scores, v_loss, summary, deltaR = model.ranker_step(sess, input_feed, labels, forward_only=False)
                        if it < valid_set.user_num / 2:
                            click_list, click_prob = click_model_1.sample_clicks(it, np.argsort(-np.array(scores[0])).tolist())
                            origin_click_list, origin_click_prob = click_model_1.sample_clicks(it)
                        else:
                            click_list, click_prob = click_model_2.sample_clicks(it - valid_set.user_num, np.argsort(-np.array(scores[0])).tolist())
                            origin_click_list, origin_click_prob = click_model_2.sample_clicks(it - valid_set.user_num)
                        it += 1
                        count_batch += 1.0
                        # print(labels, scores)
                        all_labels.extend(labels)
                        all_scores.extend(scores)
                        # all_deltas.append(deltaR)
                        click_count += np.sum(click_list)
                        origin_click_count += np.sum(origin_click_list)
                        click_prob_count += np.sum(click_prob)
                        origin_click_prob_count += np.sum(click_prob)/np.sum(origin_click_prob)
                        scores = scores[0]
                        labels = labels[0]
                        rank = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
                        all_rank.append([labels[x] for x in rank])

                    print('on test set')
                    print(current_step)
                    compute_metrics(all_labels, all_scores, valid_set, None)
                    print('click count:' , click_count, 'sum of click prob', click_prob_count)
                    print('click num rate: ', click_count/origin_click_count, ' sum of click prob rate: ', origin_click_prob_count/valid_set.user_num)
                    rank_index = np.random.randint(0, valid_set.user_num, 2).tolist()
                    print('rank index ', rank_index)
                    for x in rank_index:
                        print('scores ', all_scores[x])
                        print('origin label ', all_labels[x])
                        print('rerank ', all_rank[x])
                    checkpoint_path = check_point_dir + "/model.ckpt"
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                    if loss == float('inf'):
                        break

                    step_time, loss = 0.0, 0.0
                    sys.stdout.flush()

                    if FLAGS.max_train_iteration > 0 and current_step > FLAGS.max_train_iteration:
                        break


def decode(model_path, store_path):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        # Load test data.
        print("Reading data in %s" % FLAGS.data_dir)
        test_set = None
        if FLAGS.decode_train:
            test_set = data_utils.read_data(FLAGS.data_dir, 'train', FLAGS.train_list_cutoff)
        elif FLAGS.decode_valid:
            test_set = data_utils.read_data(FLAGS.data_dir, 'valid', FLAGS.train_list_cutoff)
        else:
            test_set = data_utils.read_data(FLAGS.data_dir, 'test', FLAGS.train_list_cutoff)

        click_model_1 = ClickModel(data_path=FLAGS.data_dir, ranker='ranker1', set_name='test', eta=0.5)
        click_model_2 = ClickModel(data_path=FLAGS.data_dir, ranker='ranker2', set_name='test', eta=0.5)

        # Create model and load parameters.
        # model = create_model(sess, test_set, False, ckpt = tf.train.get_checkpoint_state('./result/train/share_model/together_withoutbias_imbalancelearning_rate=1e-4_interval_200_exam_vs_rel_20_batch_2048'))
        print("model_path", model_path)
        model = create_model(sess, test_set, False, ckpt=tf.train.get_checkpoint_state(model_path))
        if store_path:
            store_path = store_path + model_path.split('/')[-1] + '_UPRR'


        # model = create_model(sess, test_set, False)
        model.batch_size = 1  # We decode one sentence at a time.
        if FLAGS.train_stage == 'ranker':
            for i in range(1):
                all_labels, all_scores = [], []
                it_list = []
                click_count, click_prob_count, origin_click_count, origin_click_prob_count, prob5 = 0.0, 0.0, 0.0, 0.0, 0.0
                for it in tqdm(range(int(test_set.user_num))):
                    input_feed, labels, clicks = model.get_batch_for_ranker_by_index(test_set, it)
                    # labels, scores, v_loss, summary, _ = model.ranker_step(sess, input_feed, labels, clicks, forward_only=False)
                    labels, scores = model.ranker_step(sess, input_feed, labels, clicks, forward_only=True)
                    # test KM oracle
                    # labels, scores, click_table = model.KM_oracle(sess, input_feed, labels, clicks, True, click_model_1)
                    # test KM ranker
                    # labels, scores = model.KM_ranker(sess, input_feed, labels, clicks, forward_only=True)
                    #test direct ctr ranker
                    # labels, scores = model.direct_ctr_rank(sess, input_feed, labels, clicks, forward_only=True)
                    # labels, scores = model.rel_rank(sess, input_feed, labels, clicks, forward_only=True)
                    all_labels.extend(labels)
                    all_scores.extend(scores)
                    if it < test_set.user_num / 2:
                        click_list, click_prob = click_model_1.sample_clicks(it, np.argsort(-np.array(scores[0])).tolist())
                        origin_click_list, origin_click_prob = click_model_1.sample_clicks(it)
                    else:
                        click_list, click_prob = click_model_2.sample_clicks(it-int(test_set.user_num/2), np.argsort(-np.array(scores[0])).tolist())
                        origin_click_list, origin_click_prob = click_model_2.sample_clicks(it-int(test_set.user_num/2))

                    click_count += np.sum(click_list)
                    origin_click_count += np.sum(origin_click_list)
                    click_prob_count += np.sum(click_prob)
                    prob5 += np.sum(click_prob[:5])
                    origin_click_prob_count += np.sum(origin_click_prob)

                print(len(it_list), it_list)
                print()
                compute_metrics(all_labels, all_scores, test_set, store_path)
                # compute_bid_metrics(all_labels, all_scores, test_set, store_path + 'uprr_bid')
                # compute_metrics(all_labels, all_scores, test_set, None)
                print("origin_click_per_query: ", origin_click_count / test_set.user_num, " origin_click_prob_per_doc: ",
                      origin_click_prob_count / len(test_set.data))
                print("click_per_query: ", click_count / test_set.user_num, " click_prob_per_doc: ",
                      click_prob_count / len(test_set.data))
        elif FLAGS.train_stage == 'click':
            it = 0
            count_batch = 0.0
            valid_loss = 0
            all_acc = 0
            all_auc = 0
            rel_error = 0
            label_list =[]
            click_list = []
            for it in tqdm(range(int(len(test_set.qids)))):
                begin_index = sum(test_set.len_list[0:it])
                end_index = sum(test_set.len_list[0: it + 1])
                input_feed, data = model.get_batch_for_click_by_index(test_set, np.arange(begin_index, end_index), pos=-1)
                label_list.append(data[:,4])
                v_loss, _ , auc, acc, click, step_rel_error = model.click_step(sess, input_feed, True,
                                                                                    click_model_1, data, True)
                valid_loss += v_loss
                rel_error += step_rel_error
                all_acc += acc
                click_list.append(click)
                count_batch += 1.0
            valid_loss /= count_batch
            acc = all_acc / count_batch
            label_ary = np.concatenate(label_list)
            click_ary = np.concatenate(click_list)
            print(label_ary.shape, click_ary.shape)
            auc = roc_auc_score(label_ary, click_ary)
            print("finished!")

            rel_error = rel_error / count_batch
            print("  test: loss %.2f auc %.4f acc %.4f rel_err %.4f" % (valid_loss, auc, acc, rel_error))

    return



def main(_):
    if FLAGS.decode:
        # choose the model to test
        if FLAGS.data_dir == './data/Yahoo/':
            decode(
                FLAGS.train_dir + 'ranker/learning_rate=8e-4,pair_each_query=40_batch_1024',
                None)
        elif FLAGS.data_dir == './data/MSLR10K/':
            decode(FLAGS.train_dir + 'ranker/learning_rate=5e-4,pair_each_query=40_batch_1024', None)
        else:
            decode(
                FLAGS.train_dir + 'ranker/learning_rate=5e-4,pair_each_query=40_batch_1024',
                None)
    else:
        train(FLAGS.store_dir)


if __name__ == "__main__":
    tf.app.run()
