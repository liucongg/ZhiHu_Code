"""
@author: liucong
@contact: logcongcong@gmail.com
@time: 2020/5/25 18:11
"""
import pickle
import json
import time
import numpy as np
from _datetime import timedelta
import bert_modeling
import tensorflow as tf
from model_bert import Model
from prepro import convert_features
import bert_tokenization
import os


def train(config):
    print("load train_eval ...")
    with open(config.train_eval, "rb") as fh:
        train_data = pickle.load(fh)
    print("load dev_eval ...")
    with open(config.dev_eval, "rb") as fh:
        dev_data = pickle.load(fh)

    bert_config = bert_modeling.BertConfig.from_json_file(config.bert_config_file)
    print('Building model ...')
    graph = tf.Graph()
    firstSave = int(len(train_data.label) / config.batch_size)
    config.num_train_steps = firstSave * config.num_epochs
    print("first save model after %d batch" % firstSave)
    with graph.as_default() as g:
        model = Model(config, bert_config)
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        print("Constructing TensorFlow Graph ...")
        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            print("Generating batch")
            batch_train = batch_iter(data=list(zip(train_data.input_ids, train_data.input_mask,
                                                   train_data.segment_ids, train_data.label)),
                                     batch_size=config.batch_size, num_epochs=config.num_epochs, shuffle=True)
            def feed_data(batch):
                input_ids, input_mask, segment_ids, label_batch = zip(*batch)
                feed_dict = {
                    model.input_ids: input_ids,
                    model.input_mask: input_mask,
                    model.segment_ids: segment_ids,
                    model.y_true: label_batch
                }
                return feed_dict, len(label_batch)

            def evaluate(input_ids, input_mask, segment_ids, label):
                batch_eval = batch_iter(data=list(zip(input_ids, input_mask, segment_ids, label)),
                                        batch_size=8, num_epochs=1, shuffle=False)
                total_loss = 0
                total_acc = 0
                cnt = 0
                for batch in batch_eval:
                    feed_dict, cur_batch_len = feed_data(batch)
                    feed_dict[model.is_train] = False
                    loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
                    total_loss += loss * cur_batch_len
                    total_acc += acc * cur_batch_len
                    cnt += cur_batch_len
                return total_loss / cnt, total_acc / cnt

            print("training and evaluting")
            start_time = time.time()
            print_pre_batch = config.print_pre_batch
            Macc = 0.0
            patience = 0
            train_all_loss = 0
            train_all_acc = 0
            train_all_num = 0
            for i, batch in enumerate(batch_train):
                feed_dict, train_num = feed_data(batch)
                feed_dict[model.is_train] = True
                if i % print_pre_batch == print_pre_batch - 1:
                    loss_train, acc_train = train_all_loss / train_all_num, train_all_acc / train_all_num
                    train_all_loss = 0
                    train_all_acc = 0
                    train_all_num = 0
                    loss_val, acc_val = evaluate(dev_data.input_ids, dev_data.input_mask,
                                                 dev_data.segment_ids, dev_data.label)
                    end_time = time.time()
                    time_dif = end_time - start_time
                    time_dif = timedelta(seconds=int(round(time_dif)))
                    msg = "Iter:{0:6}, Train loss:{1:6.4}, Train acc:{2:7.4}, Val loss:{3:6.4}, Val acc:{4:7.4}, " \
                          "Time:{5} "
                    print(msg.format(i + 1, loss_train, acc_train, loss_val, acc_val, time_dif))
                    if acc_val > Macc:
                        patience = 0
                        Macc = acc_val
                        print("save model")
                        saver.save(sess, os.path.join(config.save_dir, "bert_model.ckpt"))
                    else:
                        patience += 1
                        if patience > config.early_stop:
                            break
                train_loss, train_acc, _ = sess.run([model.loss, model.acc, model.optim], feed_dict=feed_dict)
                train_all_loss += train_loss * train_num
                train_all_acc += train_acc * train_num
                train_all_num += train_num
            sess.close()


def batch_iter(data, batch_size=64, num_epochs=5, shuffle=True):
    data = list(data)
    data_size = len(data)
    temp = len(data) % batch_size
    if temp == 0:
        num_batches_per_epoch = int(len(data) / batch_size)
    else:
        num_batches_per_epoch = int(len(data) / batch_size) + 1
    for _ in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            data = np.array(data)
            shuffle_data = data[shuffle_indices]
        else:
            shuffle_data = np.array(data)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffle_data[start_index:end_index]


def save_pb(config):
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        saver = tf.train.import_meta_graph(os.path.join(config.save_dir, "bert_model.ckpt") + ".meta")
        saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
        path = config.save_dir
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                        output_node_names=["output_layer/predictions"])
        with tf.gfile.FastGFile(path + "/" + "bert_model.pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())












