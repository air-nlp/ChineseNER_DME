#-*-coding:utf-8-*-
import os
import re
import codecs
from utils_ls.alphabet import Alphabet
import numpy as np

from data_utils import create_dico, create_mapping, zero_digits
from data_utils import iob2, iob_iobes, get_seg_features
NULLKEY = "-null-"

def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    num = 0
    for line in codecs.open(path, 'rb', 'utf8'):
        num+=1
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        # print(list(line))
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()
                # word[0] = " "
            else:
                word= line.split()
            assert len(word) >= 2, print([word[0]])
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def char_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(chars)
    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in chars)
    ))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[char[-1] for char in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, train=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """

    none_index = tag_to_id["O"]

    def f(x):
        return x.lower() if lower else x
    data = []
    for s in sentences:
        string = [w[0] for w in s]
        chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
                 for w in string]
        segs = get_seg_features("".join(string))
        if train:
            tags = [tag_to_id[w[-1]] for w in s]
        else:
            tags = [none_index for _ in chars]
        data.append([string, chars, segs, tags])

    return data


def augment_with_pretrained(dictionary, ext_emb_path, chars):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'rb', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary:
                dictionary[char] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def save_maps(save_path, *params):
    """
    Save mappings and invert mappings
    """
    pass
    # with codecs.open(save_path, "w", encoding="utf8") as f:
    #     pickle.dump(params, f)


def load_maps(save_path):
    """
    Load mappings from the file
    """
    pass
    # with codecs.open(save_path, "r", encoding="utf8") as f:
    #     pickle.load(save_path, f)

def build_gaz_file(gaz, gaz_file):
    ## build gaz file,initial read gaz embedding file
    if gaz_file:
        fins = codecs.open(gaz_file, 'r', encoding='utf8').readlines()
        for fin in fins:
            fin = fin.strip().split()[0]
            if fin:
                gaz.insert(fin, "one_source")
        print("Load gaz file: ", gaz_file, " total size:", gaz.size())
    else:
        print("Gaz file is None, load nothing")

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

def build_gaz_alphabet(gaz, gaz_alphabet, input_file, number_normalized):
    in_lines = codecs.open(input_file,'r', encoding='utf8').readlines()
    word_list = []
    #gaz_alphabet = Alphabet('gaz')
    for line in in_lines:
        if len(line) > 3:
            word = line.split()[0]
            if number_normalized:
                word = normalize_word(word)
            word_list.append(word)
        else:
            w_length = len(word_list)
            for idx in range(w_length):
                matched_entity = gaz.enumerateMatchList(word_list[idx:])
                for entity in matched_entity:
                    # print entity, self.gaz.searchId(entity),self.gaz.searchType(entity)
                    gaz_alphabet.add(entity)
            word_list = []
    print("gaz alphabet size:", gaz_alphabet.size())


def read_instance_with_gaz(input_file, gaz, word_alphabet, biword_alphabet, char_alphabet, gaz_alphabet, label_alphabet,
                           number_normalized, max_sent_length, char_padding_size=-1, char_padding_symbol='</pad>'):
    in_lines = codecs.open(input_file, 'r', encoding='utf8').readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    biwords = []
    chars = []
    labels = []
    word_Ids = []
    biword_Ids = []
    char_Ids = []
    label_Ids = []
    for idx in range(0, len(in_lines)):
        line = in_lines[idx]
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0]
            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            if idx < len(in_lines) - 1 and len(in_lines[idx + 1]) > 2:
                biword = word + in_lines[idx + 1].strip().split()[0]
            else:
                biword = word + NULLKEY
            biwords.append(biword)
            words.append(word)
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
            biword_Ids.append(biword_alphabet.get_index(biword))
            label_Ids.append(label_alphabet.get_index(label))
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
                assert (len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)

        else:
            if ((max_sent_length < 0) or (len(words) < max_sent_length)) and (len(words) > 0):
                gazs = []
                gaz_Ids = []
                w_length = len(words)
                # print sentence
                # for w in words:
                #     print w," ",
                # print
                for idx in range(w_length):
                    matched_list = gaz.enumerateMatchList(words[idx:])
                    matched_length = [len(a) for a in matched_list]
                    # print idx,"----------"
                    # print "forward...feed:","".join(words[idx:])
                    # for a in matched_list:
                    #     print a,len(a)," ",
                    # print

                    # print matched_length

                    gazs.append(matched_list)
                    matched_Id = [gaz_alphabet.get_index(entity) for entity in matched_list]
                    if matched_Id:
                        gaz_Ids.append([matched_Id, matched_length])
                    else:
                        gaz_Ids.append([])
                gaz_perword_ids = gaz_perword_id(gaz_Ids, 4)
                match_words_num = get_match_words_num(gaz_perword_ids, 4)
                instence_texts.append([words, biwords, chars, gazs, labels])
                instence_Ids.append([word_Ids, biword_Ids, char_Ids, gaz_Ids, gaz_perword_ids, label_Ids, match_words_num])
            words = []
            biwords = []
            chars = []
            labels = []
            word_Ids = []
            biword_Ids = []
            char_Ids = []
            label_Ids = []
            gazs = []
            gaz_Ids = []
    return instence_texts, instence_Ids
def gaz_perword_id(gaz_ids, words_num):
    perword_ids = []
    for i in range(len(gaz_ids)):
        perword_ids.append([])
    for i in range(0, len(gaz_ids)):
        if gaz_ids[i] == []:
            continue
        for num, word_id in enumerate(gaz_ids[i][0]):
            for word_len in range(gaz_ids[i][1][num]):
                perword_ids[i + word_len].append(word_id)
    for word_place in range(len(perword_ids)):
        if len(perword_ids[word_place]) > words_num:
            perword_ids[word_place] = perword_ids[word_place][:words_num]
        else:
            perword_ids[word_place] = perword_ids[word_place] + [0] *(words_num - len(perword_ids[word_place]))
    return np.transpose(perword_ids).tolist()

def build_alphabet(input_file, number_normalized, label_alphabet, word_alphabet, biword_alphabet, char_alphabet):
    in_lines = codecs.open(input_file, 'r', encoding='utf-8').readlines()
    for idx in range(len(in_lines)):
        line = in_lines[idx]
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0]
            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            label_alphabet.add(label)
            word_alphabet.add(word)
            if idx < len(in_lines) - 1 and len(in_lines[idx+1]) > 2:
                biword = word + in_lines[idx+1].strip().split()[0]
            else:
                biword = word + NULLKEY
            biword_alphabet.add(biword)
            for char in word:
                char_alphabet.add(char)
    word_alphabet_size = word_alphabet.size()
    biword_alphabet_size = biword_alphabet.size()
    char_alphabet_size = char_alphabet.size()
    label_alphabet_size = label_alphabet.size()
    print('word_alphabet_size', word_alphabet_size)
    print('biword_alphabet_size', biword_alphabet_size)
    print('char_alphabet_size', char_alphabet_size)
    print('label_alphabet_size', label_alphabet_size)
def get_match_words_num(gaz_perword_ids, words_num):
    match_words_nums = []
    for loop_num in range(0, words_num):
        match_words_num = [-1] * len(gaz_perword_ids[0])
        match_words_nums.append(match_words_num)
        match_words_num = []
    for i in range(words_num-1, -1 ,-1):
        for num, word_id in enumerate(gaz_perword_ids[i]):
            if word_id != 0 and  match_words_nums[i][num] == -1:
                for j in range(0, i + 1):
                    match_words_nums[j][num] = 1.0/(i+1)
            if word_id == 0 :
                match_words_nums[i][num] = 0.0
    return match_words_nums
