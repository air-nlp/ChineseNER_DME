#!-*-codeing=utf-8-*-
import codecs
'''
#clean weibo_2nd data
file = open('/workspace/multiword-ner/sighan/sighan_train.utf8', 'rb')
clean_file = open('/workspace/multiword-ner/sighan/sighan_train_bio','w', encoding='utf-8')
for line in file:
    line = line.decode('utf-8')
    if line != '\n':
        words = line.split()
        words[0] = words[0][:-1]
        clean_file.write('\t'.join(words))
        clean_file.write('\n')
    else:
        clean_file.write('\n')
file.close()
clean_file.close()

#clean sighan data
file = open('/workspace/multiword-ner/sighan/sighan_valid.utf8', 'rb')
clean_file = open('/workspace/multiword-ner/sighan/sighan_dev_bio','w', encoding='utf-8')
for line in file:
    line = line.decode('utf-8')
    if line != '\n':
        words = line.split()
        del words[1]
        clean_file.write('\t'.join(words))
        clean_file.write('\n')
    else:
        clean_file.write('\n')
file.close()
clean_file.close()
'''
# bio -> bioes
from loader import load_sentences, update_tag_scheme
import tensorflow as tf
import os

def bio2bioes(list, file):
    for sentence in list:
        for word in sentence:
            file.write('\t'.join(word))
            file.write('\n')
        file.write('\n')
    file.close()
if __name__== '__main__':
    flags = tf.app.flags
    flags.DEFINE_string("train_file",   os.path.join("sighan", "sighan_train_bio"),  "Path for train data")
    flags.DEFINE_string("dev_file",     os.path.join("sighan", "sighan_dev_bio"),    "Path for dev data")
    flags.DEFINE_string("test_file",    os.path.join("sighan", "sighan_test_bio"),   "Path for test data")
    flags.DEFINE_boolean("zeros",       False,      "Wither replace digits with zero")
    flags.DEFINE_boolean("lower",       True,       "Wither lower case")
    FLAGS = tf.app.flags.FLAGS
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
    dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)
    update_tag_scheme(train_sentences, "iobes")
    update_tag_scheme(dev_sentences, "iobes")
    update_tag_scheme(test_sentences, "iobes")
    train = open('/workspace/multiword-ner/sighan/sighan_train_iobes', 'w', encoding='utf-8')
    dev = open('/workspace/multiword-ner/sighan/sighan_dev_iobes', 'w', encoding='utf-8')
    test = open('/workspace/multiword-ner/sighan/sighan_test_iobes', 'w', encoding='utf-8')
    bio2bioes(train_sentences, train)
    bio2bioes(dev_sentences, dev)
    bio2bioes(test_sentences, test)

