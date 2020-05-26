"""
@author: liucong
@contact: logcongcong@gmail.com
@time: 2020/5/25 18:11
"""

import tensorflow as tf
import bert_modeling
import bert_optimization


class Model(object):
    def __init__(self, config, bert_config):
        self.max_seq_length = config.max_seq_length
        self.learning_rate = config.learning_rate
        self.num_train_steps = config.num_train_steps
        self.num_warmup_steps = config.num_warmup_steps
        self.use_one_hot_embeddings = config.use_one_hot_embeddings
        self.init_checkpoint = config.init_checkpoint
        self.bert_config = bert_config

        with tf.name_scope("input"):
            self.is_train = tf.placeholder(tf.bool, shape=[], name="is_train")
            print("is_train:", self.is_train)
            self.input_ids = tf.placeholder(tf.int32, [None, self.max_seq_length], "input_ids")
            print("input_ids:", self.input_ids)
            self.input_mask = tf.placeholder(tf.int32, [None, self.max_seq_length], "input_mask")
            print("input_mask:", self.input_mask)
            self.segment_ids = tf.placeholder(tf.int32, [None, self.max_seq_length], "segment_ids")
            print("segment_ids:", self.segment_ids)
            self.y_true = tf.placeholder(tf.int32, [None, 14], "true_label")
            print("y_true:", self.y_true)
        self.forward()

    def forward(self):
        model = bert_modeling.BertModel(
            config=self.bert_config,
            is_training=self.is_train,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=self.use_one_hot_embeddings)
        self.tvars = tf.trainable_variables()
        print(self.init_checkpoint)
        (self.assignment_map, _) = bert_modeling.get_assigment_map_from_checkpoint(self.tvars, self.init_checkpoint)
        tf.train.init_from_checkpoint(self.init_checkpoint, self.assignment_map)
        self.sequence_output_layer = model.get_pooled_output()

        with tf.variable_scope("output_layer"):
            self.predict_layer_logits = tf.layers.dense(self.sequence_output_layer,
                                                            units=14,
                                                            name="prediction_layer")
            self.y_pred = tf.nn.softmax(self.predict_layer_logits, name="scores")
            self.predictions = tf.argmax(self.y_pred, axis=1, name="predictions")
            print("self.predictions:", self.predictions)

        with tf.name_scope("loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.predict_layer_logits,
                                                                    labels=self.y_true)
            self.loss = tf.reduce_mean(cross_entropy, name="loss")

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.argmax(self.y_pred, 1),
                                           tf.argmax(self.y_true, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="acc")

        with tf.name_scope("optimize"):
            self.optim = bert_optimization.create_optimizer(self.loss, self.learning_rate,
                                                            self.num_train_steps, self.num_warmup_steps,
                                                            False)