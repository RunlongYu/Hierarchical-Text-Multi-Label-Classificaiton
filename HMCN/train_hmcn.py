# -*- coding:utf-8 -*-
__author__ = 'Randolph'

PRO_CUR_PATH = '/home/huangwei/PycharmProjects/Patent-Tag-Classification/'

import os
import sys
import time
import logging
import numpy as np
import tensorflow as tf

sys.path.append(PRO_CUR_PATH)

from text_ann_base import TextANN_BASE
from text_ann import TextANN
from utils import checkmate as cm
from utils import data_helpers as dh
from sklearn.metrics import roc_auc_score

# Parameters
# ==================================================

TRAIN_OR_RESTORE = input("☛ Train or Restore?(T/R): ")

while not (TRAIN_OR_RESTORE.isalpha() and TRAIN_OR_RESTORE.upper() in ['T', 'R']):
    TRAIN_OR_RESTORE = input("✘ The format of your input is illegal, please re-input: ")
logging.info("✔︎ The format of your input is legal, now loading to next step...")

TRAIN_OR_RESTORE = TRAIN_OR_RESTORE.upper()

if TRAIN_OR_RESTORE == 'T':
    logger = dh.logger_fn("tflog", "logs/training-{0}.log".format(time.asctime()))
if TRAIN_OR_RESTORE == 'R':
    logger = dh.logger_fn("tflog", "logs/restore-{0}.log".format(time.asctime()))

TRAININGSET_DIR = '../data/Train.json'
VALIDATIONSET_DIR = '../data/Test.json'
METADATA_DIR = '../data/metadata.tsv'

# Data Parameters
tf.flags.DEFINE_string("training_data_file", TRAININGSET_DIR, "Data source for the training data.")
tf.flags.DEFINE_string("validation_data_file", VALIDATIONSET_DIR, "Data source for the validation data.")
tf.flags.DEFINE_string("metadata_file", METADATA_DIR, "Metadata file for embedding visualization"
                                                      "(Each line is a word segment in metadata_file).")
tf.flags.DEFINE_string("train_or_restore", TRAIN_OR_RESTORE, "Train or Restore.")

# Model Hyperparameters
tf.flags.DEFINE_float("learning_rate", 0.001, "The learning rate (default: 0.001)")
tf.flags.DEFINE_integer("pad_seq_len", 150, "Recommended padding Sequence length of data (depends on the data)")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("embedding_type", 1, "The embedding type (default: 1)")
tf.flags.DEFINE_integer("fc_hidden_size", 1024, "Hidden size for fully connected layer (default: 1024)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("alpha", 100, "Weight of xxx in loss cal")
tf.flags.DEFINE_float("beta", 1.0, "Weight of global losses in loss cal")
tf.flags.DEFINE_string("num_classes_list", "9,128,661", "Number of labels list (depends on the task)")
tf.flags.DEFINE_integer("total_classes", 798, "Total number of labels list (depends on the task)")
tf.flags.DEFINE_integer("top_num", 5, "Number of top K prediction classes (default: 5)")
tf.flags.DEFINE_float("threshold", 0.5, "Threshold for prediction classes (default: 0.5)")
tf.flags.DEFINE_string("ann_type", "TextANN", "Type of ANN")

# Training Parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 256)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 5000)")
tf.flags.DEFINE_float("norm_ratio", 2, "The ratio of the sum of gradients norms of trainable variable (default: 1.25)")
tf.flags.DEFINE_integer("decay_steps", 5000, "how many steps before decay learning rate. (default: 500)")
tf.flags.DEFINE_float("decay_rate", 0.95, "Rate of decay for learning rate. (default: 0.95)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 1000)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 50)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
dilim = '-' * 100
logger.info(dilim)
for attr, value in sorted(FLAGS.__flags.items()):
    logger.info('{:>50}|{:<50}'.format(attr.upper(), value))
logger.info(dilim)


def train_ann():
    """Training ANN model."""

    # Load sentences, labels, and training parameters
    logger.info("✔︎ Loading data...")

    logger.info("✔︎ Training data processing...")
    train_data = dh.load_data_and_labels(FLAGS.training_data_file, FLAGS.num_classes_list, FLAGS.total_classes,
                                         FLAGS.embedding_dim, data_aug_flag=False)

    logger.info("✔︎ Validation data processing...")
    val_data = dh.load_data_and_labels(FLAGS.validation_data_file, FLAGS.num_classes_list, FLAGS.total_classes,
                                       FLAGS.embedding_dim, data_aug_flag=False)

    logger.info("Recommended padding Sequence length is: {0}".format(FLAGS.pad_seq_len))

    logger.info("✔︎ Training data padding...")
    x_train, y_train, y_train_tuple = dh.pad_data(train_data, FLAGS.pad_seq_len)

    logger.info("✔︎ Validation data padding...")
    x_val, y_val, y_val_tuple = dh.pad_data(val_data, FLAGS.pad_seq_len)

    # Build vocabulary
    VOCAB_SIZE = dh.load_vocab_size(FLAGS.embedding_dim)
    pretrained_word2vec_matrix = dh.load_word2vec_matrix(VOCAB_SIZE, FLAGS.embedding_dim)

    # Build a graph and ann object
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            ann = eval(FLAGS.ann_type)(
                sequence_length=FLAGS.pad_seq_len,
                num_classes_list=list(map(int, FLAGS.num_classes_list.split(','))),
                total_classes=FLAGS.total_classes,
                vocab_size=VOCAB_SIZE,
                fc_hidden_size=FLAGS.fc_hidden_size,
                embedding_size=FLAGS.embedding_dim,
                embedding_type=FLAGS.embedding_type,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                pretrained_embedding=pretrained_word2vec_matrix,
                alpha=FLAGS.alpha,
                beta=FLAGS.beta)

            # Define training procedure
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.learning_rate,
                                                           global_step=ann.global_step, decay_steps=FLAGS.decay_steps,
                                                           decay_rate=FLAGS.decay_rate, staircase=True)
                optimizer = tf.train.AdamOptimizer(learning_rate)
                grads, vars = zip(*optimizer.compute_gradients(ann.loss))
                grads, _ = tf.clip_by_global_norm(grads, clip_norm=FLAGS.norm_ratio)
                train_op = optimizer.apply_gradients(zip(grads, vars), global_step=ann.global_step, name="train_op")

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in zip(grads, vars):
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{0}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{0}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            if FLAGS.train_or_restore == 'R':
                MODEL = input("☛ Please input the checkpoints model you want to restore, "
                              "it should be like(1490175368): ")  # The model you want to restore

                while not (MODEL.isdigit() and len(MODEL) == 10):
                    MODEL = input("✘ The format of your input is illegal, please re-input: ")
                logger.info("✔︎ The format of your input is legal, now loading to next step...")
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", MODEL))
                logger.info("✔︎ Writing to {0}\n".format(out_dir))
            else:
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                logger.info("✔︎ Writing to {0}\n".format(out_dir))

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            best_checkpoint_dir = os.path.abspath(os.path.join(out_dir, "bestcheckpoints"))

            # Summaries for loss
            loss_summary = tf.summary.scalar("loss", ann.loss)

            # Train summaries
            train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            validation_summary_op = tf.summary.merge([loss_summary])
            validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
            validation_summary_writer = tf.summary.FileWriter(validation_summary_dir, sess.graph)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            best_saver = cm.BestCheckpointSaver(save_dir=best_checkpoint_dir, num_to_keep=3, maximize=True)

            if FLAGS.train_or_restore == 'R':
                # Load ann model
                logger.info("✔︎ Loading model...")
                checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                logger.info(checkpoint_file)

                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
            else:
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                # Save the embedding visualization
                saver.save(sess, os.path.join(out_dir, "embedding", "embedding.ckpt"))

            current_step = sess.run(ann.global_step)

            def train_step(x_batch, y_batch, y_batch_tuple):
                """A single training step"""
                y_batch_first = [i[0] for i in y_batch_tuple]
                y_batch_second = [j[1] for j in y_batch_tuple]
                y_batch_third = [k[2] for k in y_batch_tuple]

                feed_dict = {
                    ann.input_x: x_batch,
                    ann.input_y_first: y_batch_first,
                    ann.input_y_second: y_batch_second,
                    ann.input_y_third: y_batch_third,
                    ann.input_y: y_batch,
                    ann.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    ann.is_training: True
                }
                _, step, summaries, loss = sess.run(
                    [train_op, ann.global_step, train_summary_op, ann.loss], feed_dict)
                logger.info("step {0}: loss {1:g}".format(step, loss))
                train_summary_writer.add_summary(summaries, step)

            def validation_step(x_val, y_val, y_val_tuple, writer=None):
                """Evaluates model on a validation set"""
                batches_validation = dh.batch_iter(
                    list(zip(x_val, y_val, y_val_tuple)), FLAGS.batch_size, 1)

                # Predict classes by threshold or topk ('ts': threshold; 'tk': topk)
                eval_counter, eval_loss, eval_auc = 0, 0.0, 0.0
                eval_rec_ts, eval_pre_ts, eval_F_ts = 0.0, 0.0, 0.0
                eval_rec_tk = [0.0] * FLAGS.top_num
                eval_pre_tk = [0.0] * FLAGS.top_num
                eval_F_tk = [0.0] * FLAGS.top_num
                val_scores = []

                for batch_validation in batches_validation:
                    x_batch_val, y_batch_val, y_batch_val_tuple = zip(*batch_validation)

                    y_batch_val_first = [i[0] for i in y_batch_val_tuple]
                    y_batch_val_second = [j[1] for j in y_batch_val_tuple]
                    y_batch_val_third = [k[2] for k in y_batch_val_tuple]

                    feed_dict = {
                        ann.input_x: x_batch_val,
                        ann.input_y_first: y_batch_val_first,
                        ann.input_y_second: y_batch_val_second,
                        ann.input_y_third: y_batch_val_third,
                        ann.input_y: y_batch_val,
                        ann.dropout_keep_prob: 1.0,
                        ann.is_training: False
                    }
                    step, summaries, scores, cur_loss = sess.run(
                        [ann.global_step, validation_summary_op, ann.scores, ann.loss], feed_dict)

                    for predicted_scores in scores:
                        val_scores.append(predicted_scores)

                    # Predict by threshold
                    predicted_labels_threshold, predicted_values_threshold = \
                        dh.get_label_using_scores_by_threshold(scores=scores, threshold=FLAGS.threshold)

                    cur_rec_ts, cur_pre_ts, cur_F_ts = 0.0, 0.0, 0.0

                    for index, predicted_label_threshold in enumerate(predicted_labels_threshold):
                        rec_inc_ts, pre_inc_ts = dh.cal_metric(predicted_label_threshold, y_batch_val[index])
                        cur_rec_ts, cur_pre_ts = cur_rec_ts + rec_inc_ts, cur_pre_ts + pre_inc_ts

                    cur_rec_ts = cur_rec_ts / len(y_batch_val)
                    cur_pre_ts = cur_pre_ts / len(y_batch_val)

                    cur_F_ts = dh.cal_F(cur_rec_ts, cur_pre_ts)

                    eval_rec_ts, eval_pre_ts = eval_rec_ts + cur_rec_ts, eval_pre_ts + cur_pre_ts

                    # Predict by topK
                    topK_predicted_labels = []
                    for top_num in range(FLAGS.top_num):
                        predicted_labels_topk, predicted_values_topk = \
                            dh.get_label_using_scores_by_topk(scores=scores, top_num=top_num+1)
                        topK_predicted_labels.append(predicted_labels_topk)

                    cur_rec_tk = [0.0] * FLAGS.top_num
                    cur_pre_tk = [0.0] * FLAGS.top_num
                    cur_F_tk = [0.0] * FLAGS.top_num

                    for top_num, predicted_labels_topK in enumerate(topK_predicted_labels):
                        for index, predicted_label_topK in enumerate(predicted_labels_topK):
                            rec_inc_tk, pre_inc_tk = dh.cal_metric(predicted_label_topK, y_batch_val[index])
                            cur_rec_tk[top_num], cur_pre_tk[top_num] = \
                                cur_rec_tk[top_num] + rec_inc_tk, cur_pre_tk[top_num] + pre_inc_tk

                        cur_rec_tk[top_num] = cur_rec_tk[top_num] / len(y_batch_val)
                        cur_pre_tk[top_num] = cur_pre_tk[top_num] / len(y_batch_val)

                        cur_F_tk[top_num] = dh.cal_F(cur_rec_tk[top_num], cur_pre_tk[top_num])

                        eval_rec_tk[top_num], eval_pre_tk[top_num] = \
                            eval_rec_tk[top_num] + cur_rec_tk[top_num], eval_pre_tk[top_num] + cur_pre_tk[top_num]

                    eval_loss = eval_loss + cur_loss
                    eval_counter = eval_counter + 1

                    if writer:
                        writer.add_summary(summaries, step)

                # Calculate the average AUC
                val_scores = np.array(val_scores)
                y_val = np.array(y_val)
                missing_labels_num = 0
                for index in range(FLAGS.total_classes):
                    y_true = y_val[:, index]
                    y_score = val_scores[:, index]
                    try:
                        eval_auc = eval_auc + roc_auc_score(y_true=y_true, y_score=y_score)
                    except:
                        missing_labels_num += 1

                eval_auc = eval_auc / (FLAGS.total_classes - missing_labels_num)
                eval_loss = float(eval_loss / eval_counter)
                eval_rec_ts = float(eval_rec_ts / eval_counter)
                eval_pre_ts = float(eval_pre_ts / eval_counter)
                eval_F_ts = dh.cal_F(eval_rec_ts, eval_pre_ts)

                for top_num in range(FLAGS.top_num):
                    eval_rec_tk[top_num] = float(eval_rec_tk[top_num] / eval_counter)
                    eval_pre_tk[top_num] = float(eval_pre_tk[top_num] / eval_counter)
                    eval_F_tk[top_num] = dh.cal_F(eval_rec_tk[top_num], eval_pre_tk[top_num])

                return eval_loss, eval_auc, eval_rec_ts, eval_pre_ts, eval_F_ts, eval_rec_tk, eval_pre_tk, eval_F_tk

            # Generate batches
            batches_train = dh.batch_iter(
                list(zip(x_train, y_train, y_train_tuple)), FLAGS.batch_size, FLAGS.num_epochs)

            num_batches_per_epoch = int((len(x_train) - 1) / FLAGS.batch_size) + 1

            # Training loop. For each batch...
            for batch_train in batches_train:
                x_batch_train, y_batch_train, y_batch_train_tuple = zip(*batch_train)
                train_step(x_batch_train, y_batch_train, y_batch_train_tuple)
                current_step = tf.train.global_step(sess, ann.global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    logger.info("\nEvaluation:")
                    eval_loss, eval_auc, eval_rec_ts, eval_pre_ts, eval_F_ts, eval_rec_tk, eval_pre_tk, eval_F_tk = \
                        validation_step(x_val, y_val, y_val_tuple, writer=validation_summary_writer)

                    logger.info("All Validation set: Loss {0:g} | AUC {1:g}".format(eval_loss, eval_auc))

                    # Predict by threshold
                    logger.info("☛ Predict by threshold: Recall {0:g}, Precision {1:g}, F {2:g}"
                                .format(eval_rec_ts, eval_pre_ts, eval_F_ts))

                    # Predict by topK
                    logger.info("☛ Predict by topK:")
                    for top_num in range(FLAGS.top_num):
                        logger.info("Top{0}: Recall {1:g}, Precision {2:g}, F {3:g}"
                                    .format(top_num+1, eval_rec_tk[top_num], eval_pre_tk[top_num], eval_F_tk[top_num]))
                    best_saver.handle(eval_auc, sess, current_step)
                if current_step % FLAGS.checkpoint_every == 0:
                    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logger.info("✔︎ Saved model checkpoint to {0}\n".format(path))
                if current_step % num_batches_per_epoch == 0:
                    current_epoch = current_step // num_batches_per_epoch
                    logger.info("✔︎ Epoch {0} has finished!".format(current_epoch))

    logger.info("✔︎ Done.")


if __name__ == '__main__':
    train_ann()
