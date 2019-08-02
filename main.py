# encoding=utf8
import os
import codecs
import pickle
import itertools
from collections import OrderedDict
import os
import tensorflow as tf
import numpy as np
from model import Model
from loader import load_sentences, update_tag_scheme, build_gaz_file, build_gaz_alphabet
from loader import char_mapping, tag_mapping, read_instance_with_gaz,build_alphabet
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, make_path, clean, create_model, save_model
from utils import print_config, save_config, load_config, test_ner
from data_utils import load_word2vec, create_input, input_from_line, BatchManager, load_gaz2vec, get_seg_features
from utils_ls.alphabet import Alphabet
from utils_ls.gazetteer import Gazetteer

flags = tf.app.flags
flags.DEFINE_boolean("clean",       True,      "clean train folder")
flags.DEFINE_boolean("train",       True,      "Wither train the model")
# configurations for the model
flags.DEFINE_integer("seg_dim",     0,         "Embedding size for segmentation, 0 if not used")
#flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim",    50,        "Embedding size for characters")
flags.DEFINE_integer("lstm_dim",    200,        "Num of hidden units in LSTM")
flags.DEFINE_integer("context_lstm_dim",    100,        "Num of hidden units in context LSTM")
flags.DEFINE_string("tag_schema",   "iobes",    "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip")
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_float("batch_size",    64,         "batch size")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros",       False,      "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       True,       "Wither lower case")
flags.DEFINE_boolean("gaz_lower",   False,       "Wither gaz lower case")
flags.DEFINE_boolean("number_normalized",   True,       "number_normalized")

flags.DEFINE_integer("max_epoch",   200,        "maximum training epochs")
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "ckpt",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "/workspace/baseline_multiattention_gpu/result",       "Path for results")
#flags.DEFINE_string("emb_file",     "wiki_100.utf8", "Path for pre_trained embedding")
flags.DEFINE_string("emb_file",     "gigaword_chn.all.a2b.uni.ite50.vec", "Path for pre_trained embedding")
flags.DEFINE_string("gaz_file",     "ctb.50d_1.vec", "Path for gaz embedding")
# flags.DEFINE_string("train_file",   os.path.join("data", "example.train"),  "Path for train data")
# flags.DEFINE_string("dev_file",     os.path.join("data", "example.dev"),    "Path for dev data")
# flags.DEFINE_string("test_file",    os.path.join("data", "example.test"),   "Path for test data")
flags.DEFINE_string("train_file",   os.path.join("sighan", "sighan_train_iobes"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join("sighan", "sighan_dev_iobes"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join("sighan", "sighan_test_iobes"),   "Path for test data")

FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


# config for the model
def config_model(char_to_id, tag_to_id, word_to_id):
    config = OrderedDict()
    config["num_chars"] = len(char_to_id)
    config["num_words"] = len(word_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["context_lstm_dim"] = FLAGS.context_lstm_dim
    config["batch_size"] = FLAGS.batch_size

    config["emb_file"] = FLAGS.emb_file
    config["gaz_file"] = FLAGS.gaz_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    config["gaz_lower"] = FLAGS.gaz_lower
    return config


def evaluate(sess, model, name, data, id_to_tag, id_to_char, logger):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag, id_to_char)
    eval_lines = test_ner(ner_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def train():


    # load gaz data
    gaz = Gazetteer(FLAGS.gaz_lower)
    gaz_alphabet = Alphabet('gaz')
    gaz_data = build_gaz_file(gaz, FLAGS.gaz_file)
    train_gaz_alphabet = build_gaz_alphabet(gaz, gaz_alphabet, FLAGS.train_file, FLAGS.number_normalized)
    dev_gaz_alphabet = build_gaz_alphabet(gaz, gaz_alphabet, FLAGS.dev_file, FLAGS.number_normalized)
    test_gaz_alphabet = build_gaz_alphabet(gaz, gaz_alphabet, FLAGS.test_file, FLAGS.number_normalized)


    word_alphabet = Alphabet('word')
    biword_alphabet = Alphabet('biword')
    char_alphabet = Alphabet('character')
    label_alphabet = Alphabet('label', True)
    build_alphabet(FLAGS.train_file, FLAGS.number_normalized, label_alphabet, word_alphabet, biword_alphabet, char_alphabet)
    build_alphabet(FLAGS.dev_file, FLAGS.number_normalized, label_alphabet, word_alphabet, biword_alphabet, char_alphabet)
    build_alphabet(FLAGS.test_file, FLAGS.number_normalized, label_alphabet, word_alphabet, biword_alphabet, char_alphabet)

    word_alphabet.close()
    biword_alphabet.close()
    label_alphabet.close()
    char_alphabet.close()
    gaz_alphabet.close()


    train_text, train_id = read_instance_with_gaz(FLAGS.train_file, gaz, word_alphabet, biword_alphabet,
                                                  char_alphabet, gaz_alphabet, label_alphabet, FLAGS.number_normalized,
                                                  250)
    dev_text, dev_id = read_instance_with_gaz(FLAGS.dev_file, gaz, word_alphabet, biword_alphabet,
                                                  char_alphabet, gaz_alphabet, label_alphabet, FLAGS.number_normalized,
                                                  250)
    test_text, test_id = read_instance_with_gaz(FLAGS.test_file, gaz, word_alphabet, biword_alphabet,
                                                  char_alphabet, gaz_alphabet, label_alphabet, FLAGS.number_normalized,
                                                  250)

    # Use selected tagging scheme (IOB / IOBES)
    #update_tag_scheme(train_sentences, FLAGS.tag_schema)
    #update_tag_scheme(test_sentences, FLAGS.tag_schema)

    # create maps if not exist
    if not os.path.isfile(FLAGS.map_file):
        # create dictionary for word
        char_to_id = word_alphabet.instance2index
        id_to_char = word_alphabet.index2instance
        word_to_id = gaz_alphabet.instance2index
        id_to_word = gaz_alphabet.index2instance
        tag_to_id = label_alphabet.instance2index
        id_to_tag = label_alphabet.index2instance
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag, word_to_id, id_to_word], f)
    else:
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag, word_to_id, id_to_word = pickle.load(f)

    # prepare data, get a collection of list containing index
    '''
    train_data = prepare_dataset(
        train_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), 0, len(test_data)))
    '''

    train_manager = BatchManager(train_id, FLAGS.batch_size)
    dev_manager = BatchManager(dev_id, 100)
    test_manager = BatchManager(test_id, 100)
    # make path for store log and model if not exist
    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(char_to_id, tag_to_id, word_to_id)
        save_config(config, FLAGS.config_file)
    make_path(FLAGS)

    log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(log_path)
    print_config(config, logger)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, load_gaz2vec, config, id_to_char, id_to_word, logger)
        logger.info("start training")
        loss = []
        for i in range(FLAGS.max_epoch):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                                "NER loss:{:>9.6f}".format(
                        iteration, step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            best = evaluate(sess, model, "dev", dev_manager, id_to_tag, id_to_char, logger)
            if best:
                save_model(sess, model, FLAGS.ckpt_path, logger)
            evaluate(sess, model, "test", test_manager, id_to_tag, id_to_char, logger)


def evaluate_line():
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        while True:
            # try:
            #     line = input("请输入测试句子:")
            #     result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
            #     print(result)
            # except Exception as e:
            #     logger.info(e)

                line = input("请输入测试句子:")
                result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
                print(result)


def main(_):

    if FLAGS.train:
        if FLAGS.clean:
            clean(FLAGS)
        train()
    else:
        evaluate_line()


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    tf.app.run(main)



