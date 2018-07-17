"""
Dataloader for event data. this includes

- rocstories
- didemo
- MPII
- activitynet captions
"""
import pandas as pd
import json
from collections import defaultdict
import numpy as np
import re
import os
from tqdm import tqdm
from allennlp.common.util import get_spacy_model
from tqdm import tqdm
from unidecode import unidecode
from num2words import num2words

# TODO! change with where you're keeping the data
raise ValueError("you should make sure the data you want is here. then you can delete this error")
DATA_PATH = os.path.dirname(os.path.realpath(__file__))

def remove_allcaps(sent):
    """
    Given a sentence, filter it so that it doesn't contain some words that are ALLcaps
    :param sent: string, like SOMEONE wheels SOMEONE on, mouthing silent words of earnest prayer.
    :return:                  Someone wheels someone on, mouthing silent words of earnest prayer.
    """
    # Remove all caps
    def _sanitize(word, is_first):
        if word in ("I"):
            return word

        num_capitals = len([x for x in word if not x.islower()])
        if num_capitals > len(word) // 2:
            # We have an all caps word here.
            if is_first:
                return word[0] + word[1:].lower()
            return word.lower()
        return word

    return ' '.join([_sanitize(word, i == 0) for i, word in enumerate(sent.split(' '))])


def load_rocstories(split):
    """
    Load rocstories dataset
    :param split: in train, val, or test. note that training doesn't have the endings.
    for now we'll remove the endings
    :return:
    """
    assert split in ('train', 'val', 'test')

    if split in 'train':
        df = pd.concat((
            pd.read_csv(os.path.join(DATA_PATH, 'rocstories', 'ROCStories__spring2016 - ROCStories_spring2016.csv')),
            pd.read_csv(os.path.join(DATA_PATH, 'rocstories', 'ROCStories_winter2017 - ROCStories_winter2017.csv')),
        ), 0)
    else:
        df = pd.read_csv(
            os.path.join(DATA_PATH, 'rocstories', 'cloze_test_{}__spring2016 - cloze_test_ALL_{}.csv'.format(
                split, split)))

        # FOR NOW REMOVE THE ENDINGS AND PRETEND IT'S THE SAME AS TRAINING DATA
        df['InputSentence5'] = df.apply(
            lambda x: [x['RandomFifthSentenceQuiz1'], x['RandomFifthSentenceQuiz2']][x['AnswerRightEnding'] - 1],
            axis=1)
        df = df[['InputStoryid'] + ['InputSentence{}'.format(i + 1) for i in range(5)]]
        df.columns = ['storyid'] + ['sentence{}'.format(i + 1) for i in range(5)]

    rocstories = []
    for i, item in df.iterrows():
        rocstories.append(
            {'id': item['storyid'], 'sentences': [item['sentence{}'.format(i + 1)] for i in range(5)]})
    return rocstories


# def load_rocstories_nogender(split):
#     spacy = get_spacy_model("en_core_web_sm", pos_tags=False, parse=False, ner=True)
#     import gender_guesser.detector as gender
#     d = gender.Detector()
#
#     def _replace(name, is_cap=False):
#         gend = d.get_gender(name)
#         if gend in ('male', 'mostly_male', 'andy'):
#             return 'The man' if is_cap else 'the man'
#         if gend in ('female', 'mostly_female'):
#             return 'The woman' if is_cap else 'the woman'
#         return name
#
#     def _fix_sentence(sent):
#         doc = spacy(sent)
#         sentence = doc.text
#         for ent in doc.ents:
#             if ent.label_ == 'PERSON':
#                 repl_text_cap = _replace(ent.text, is_cap=True)
#                 if repl_text_cap != ent.text:
#                     repl_text_lower = _replace(ent.text, is_cap=False)
#                     sentence = re.sub('^' + ent.text, repl_text_cap, sentence)
#                     sentence = re.sub(ent.text, repl_text_lower, sentence)
#         return sentence
#
#     roc = load_rocstories(split)
#     for story in tqdm(roc):
#         for i in range(5):
#             story['sentences'][i] = _fix_sentence(story['sentences'][i])
#     return roc


def _to_time(pandas_col):
    offset = pd.to_datetime('00.00.00.000', format='%H.%M.%S.%f')
    return (pd.to_datetime(pandas_col, format='%H.%M.%S.%f') - offset).dt.total_seconds()


def _lsmdc_to_list(lsmdc, lsmdc_window=20):
    """
    :param lsmdc: Dataframe
    :param lmsdc_window: # We'll allow lsmdc_window seconds of a gap between chains
    :return: a list of annotations
    """
    movie = ''
    t_end = 0
    lsmdc_list = []
    cur_chain = {'sentences': []}
    for (i, item) in lsmdc.iterrows():
        # If new movie then push
        if movie != item['movie'] or item['start_aligned'] > t_end + lsmdc_window:
            if len(cur_chain['sentences']) > 1:
                lsmdc_list.append(cur_chain)
            cur_chain = {'sentences': [],
                         'timestamps': [],
                         'id': '{}-{}'.format(item['movie'], i)}
            movie = item['movie']
        t_end = item['end_aligned']
        cur_chain['sentences'].append(item['sentence'])
        cur_chain['timestamps'].append((item['start_aligned'], item['end_aligned']))
        cur_chain['duration'] = item['end_aligned'] - cur_chain['timestamps'][0][0]
    lsmdc_list.append(cur_chain)
    return lsmdc_list


def load_mpii(split):
    """
    Load MPII dataset with the <person> </person> annotations
    :return:
    """
    lsmdc = pd.read_csv(os.path.join(DATA_PATH, 'movies', 'annotations-original-coreferences-ner.csv'), sep='\t',
                        header=None, names=['id', 'sentence'])
    lsmdc['movie'] = lsmdc['id'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    lsmdc['start_aligned'] = _to_time(lsmdc['id'].apply(lambda x: x.split('_')[-1].split('-')[0]))
    lsmdc['end_aligned'] = _to_time(lsmdc['id'].apply(lambda x: x.split('_')[-1].split('-')[1]))

    datasplit = pd.read_csv(os.path.join(DATA_PATH, 'movies', 'dataSplit.txt'), sep='\t', header=None,
                            names=['movie', 'split'])

    include_these = set([m['movie'] for _, m in datasplit.iterrows()
                         if m['split'] == {'train': 'training', 'val': 'validation', 'test': 'test'}[split]])

    lsmdc = lsmdc[lsmdc.movie.isin(include_these)]

    assert len(include_these) == len(set(lsmdc['movie'].values))

    # This is a bit crude but whatever
    def _pop_tags(sent):
        sent_no_person = re.sub(r'</?PERSON>', '', sent)

        # Match everything of the form She<BLAH BLAH>. sometimes there is she<> so we dont want that
        sent_no_pronoun = re.sub(r'\w*<([^<>]+?)>', lambda x: x[1], sent_no_person)

        # Some have brackets in the original sentences so get rid of those
        sent_no_pronoun = re.sub(r'<>', '', sent_no_pronoun)
        return sent_no_pronoun

    lsmdc['sentence'] = lsmdc['sentence'].apply(_pop_tags)
    return _lsmdc_to_list(lsmdc)


def load_mpii_depersonized(split):
    """
    Loads the MPII dataset, but remove names. Instead we'll replace them with their genders. EX <PERSON>Helen</PERSON> gets replaced with the woman.
    :return:
    """
    lsmdc = pd.read_csv(os.path.join(DATA_PATH, 'movies', 'annotations-original-coreferences-ner.csv'), sep='\t',
                        header=None, names=['id', 'sentence'])
    lsmdc['movie'] = lsmdc['id'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    lsmdc['start_aligned'] = _to_time(lsmdc['id'].apply(lambda x: x.split('_')[-1].split('-')[0]))
    lsmdc['end_aligned'] = _to_time(lsmdc['id'].apply(lambda x: x.split('_')[-1].split('-')[1]))

    datasplit = pd.read_csv(os.path.join(DATA_PATH, 'movies', 'dataSplit.txt'), sep='\t', header=None,
                            names=['movie', 'split'])

    include_these = set([m['movie'] for _, m in datasplit.iterrows()
                         if m['split'] == {'train': 'training', 'val': 'validation', 'test': 'test'}[split]])

    lsmdc = lsmdc[lsmdc.movie.isin(include_these)]

    movies_depersonized = []
    for movie_name, movie_df in tqdm(lsmdc.groupby('movie')):
        chars = pd.read_csv(os.path.join(DATA_PATH, 'movies', 'coref_characters',
                                         '{}-characters.csv'.format(movie_name)),
                            sep='\t', header=None, names=['longname', 'shortname', 'gender'],
                            index_col='longname')
        char2gender = {char: gender for char, gender in chars['gender'].items()}

        def _replace(name, is_cap=False):
            if name not in char2gender:
                return 'Someone' if is_cap else 'someone'
            if char2gender[name] == 'W':
                return 'The woman' if is_cap else 'the woman'
            return 'The man' if is_cap else 'the man'

        def _pop_tags(sent):
            # Match the start of the string.
            sent0 = re.sub(
                r'^<PERSON>(.+?)</PERSON>',
                lambda match: _replace(match[1], is_cap=True),
                sent)

            # Match other parts of the string
            sent1 = re.sub(
                r'<PERSON>(.+?)</PERSON>',
                lambda match: _replace(match[1], is_cap=False),
                sent0)

            # Match everything of the form She<BLAH BLAH>.
            # sometimes there is she<> so we dont want that
            sent2 = re.sub(r'(\w*)<[^<>]+?>', lambda x: x[1], sent1)

            # Some have brackets in the original sentences so get rid of those
            sent3 = re.sub(r'<>', '', sent2)

            sent4 = remove_allcaps(sent3)

            # This is really minor but if it ends with " ." then change that.
            sent5 = sent4[:-1].rstrip() + '.' if sent4.endswith(' .') else sent4
            # # Verbose
            # print("----\n{}->{}->{}->{}->{}\n".format(sent0, sent1, sent2, sent3, sent4))
            return sent5

        movie_df_copy = movie_df.copy()
        movie_df_copy['sentence'] = movie_df_copy['sentence'].apply(_pop_tags)
        movies_depersonized.append(movie_df_copy)

    return _lsmdc_to_list(pd.concat(movies_depersonized, 0))


def load_visual_madlibs(split):
    """
    Loads the Visual Madlibs dataset, including captions from COCO as the premises
    :return:
    """
    # Let's make sure each contains a verb.
    spacy = get_spacy_model("en_core_web_sm", pos_tags=True, parse=True, ner=False)
    from nltk.tokenize.moses import MosesDetokenizer
    # from pattern.en import conjugate, verbs, PRESENT
    detokenizer = MosesDetokenizer()

    def _sentence_contains_verb(sent):
        spacy_parse = spacy(sent)
        for tok in spacy_parse:
            if tok.pos_ == 'VERB' and tok.lemma_ not in ('is', 'has'):
                return True
        return False

    def order_sents(sent_list):
        sentence_has_verb = np.array([_sentence_contains_verb(s) for i, s in enumerate(sent_list)])
        sentence_length = np.array([len(x) for x in sent_list])

        sentence_score = sentence_has_verb.astype(np.float32) + sentence_length / sentence_length.max()
        best_to_worst = np.argsort(-sentence_score).tolist()
        return [sent_list[i] for i in best_to_worst]

    futures_fn = {
        'train': 'tr_futures.json',
        'val': 'val_easy_multichoice_futures.json',
        'test': 'val_hard_multichoice_futures.json',
    }[split]
    key = {'train': 'tr_futures',
           'val': "multichoice_futures",
           'test': "multichoice_futures",
           }[split]

    id2futureandpast = defaultdict(lambda: {'captions': [], 'future': [], 'past': []})
    # with open(os.path.join(DATA_PATH, 'visualmadlibs', 'tr_pasts.json'), 'r') as f:
    #     for item in json.load(f)['tr_pasts']:
    #         id2futureandpast[item['image_id']]['past'] = order_sents(item['fitbs'])
    with open(os.path.join(DATA_PATH, 'visualmadlibs', futures_fn), 'r') as f:
        for item in json.load(f)[key]:
            if split == 'train':
                id2futureandpast[item['image_id']]['future'] = order_sents(item['fitbs'])
            else:
                id2futureandpast[item['image_id']]['future'] = [item['pos']]

    with open(os.path.join(DATA_PATH, 'coco', 'dataset_coco.json'), 'r') as f:
        imgid2caps = {item['cocoid']: ([sent['raw'] for sent in item['sentences']],
                                       item['split'])
                      for item in json.load(f)['images']}

    vml = []
    for k in tqdm(id2futureandpast):
        for cap, future in zip(order_sents(imgid2caps[k][0]), id2futureandpast[k]['future']):
            # Spacy parse the future sentence, change to present tense
            spacy_parse = [(x.orth_, x.pos_, x.dep_) for x in spacy(future)]

            # If there's a ROOT that doesn't start with ing, parse that
            is_match = False
            for i, (word, pos, dep) in enumerate(spacy_parse):
                if pos == 'VERB' and dep == 'ROOT' and not word.endswith('ing'):
                    spacy_parse[i] = (conjugate(word, tense=PRESENT), pos, dep)
                    is_match = True

            # Else convert AUXes
            if not is_match:
                for i, (word, pos, dep) in enumerate(spacy_parse):
                    if pos == 'VERB' and dep == 'aux':
                        spacy_parse[i] = (conjugate(word, tense=PRESENT), pos, dep)

            future_fixed = detokenizer.detokenize([x[0] for x in spacy_parse], return_str=True)
            print("{} -> {}".format(future, future_fixed), flush=True)

            future_fixed = future_fixed[0].capitalize() + future_fixed[1:]
            vml.append({'id': k, 'sentences': [cap, future_fixed]})

    # ABANDON THIS FOR NOW.

    #
    #     id2futureandpast[k]['captions'] = (order_sents(imgid2caps[k][0]), imgid2caps[k][1])
    #
    #
    # # Join everything
    # vml = []
    # for id, val in id2futureandpast.items():
    #     vml.append({'id': id, 'sentences': ['{} Afterwards, {}.'.format(cap, future)
    #                                         for cap, future in zip(
    #             val['captions'][0], val['future'])]})
    # import ipdb
    # ipdb.set_trace()
    return vml


def load_vist(split):
    """
    Loads the VIST dataset
    :return:
    """
    id_to_annos = defaultdict(list)
    with open(os.path.join(DATA_PATH, 'vist', '{}.story-in-sequence.json'.format(split)), 'r') as f:
        for item in json.load(f)['annotations']:
            assert len(item) == 1
            id_to_annos[int(item[0]['story_id'])].append(item[0])
    for k in id_to_annos:
        id_to_annos[k] = sorted(id_to_annos[k], key=lambda x: x['story_id'])

    res = [{'id': id, 'sentences': [x['original_text'] for x in v]} for id, v in id_to_annos.items()]
    return res

def load_lsmdc(split):
    """
    Loads LSMDC with <someone> annotations
    #TODO: investigate filtering things.
    # 1: all sentences need verbs
    # 2: filter out all sentences that don't begin with a capital letter (these are often incomplete)
    # 3. all sentences need objects

    :return:
    """
    lsmdc = pd.read_csv(os.path.join(DATA_PATH, 'movies', 'LSMDC16_annos_{}.csv'.format(
        {'train': 'training', 'val': 'val', 'test': 'test'}[split])),
                        sep='\t',
                        header=None,
                        names=['movie', 'start_aligned', 'end_aligned', 'start_extracted', 'end_extracted', 'sentence'])
    lsmdc['movie'] = lsmdc['movie'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    del lsmdc['start_extracted']
    del lsmdc['end_extracted']

    lsmdc['start_aligned'] = _to_time(lsmdc['start_aligned'])
    lsmdc['end_aligned'] = _to_time(lsmdc['end_aligned'])
    def _fix_sent(sent):
        sent1 = remove_allcaps(sent)
         # This is really minor but if it ends with " ." then change that.
        sent2 = sent1[:-1].rstrip() + '.' if sent1.endswith(' .') else sent1
        return unidecode(sent2)

    from nltk.tokenize.moses import MosesDetokenizer
    detokenizer = MosesDetokenizer()
    spacy = get_spacy_model("en_core_web_sm", pos_tags=True, parse=True, ner=False)
    def check_if_sent_is_grammatical(sent):
        if sent[0].islower():
            print("{} is not grammatical (lowercase start)".format(sent))
            return ''
        # Sanitize the sentence
        sent_sanitized = remove_allcaps(sent)

        # Loop over subsentences to find a good one
        for sent_parsed in spacy(sent_sanitized).sents:
            root = sent_parsed.root
            if root.pos_ != 'VERB':
                print("{} is not grammatical (noverb)".format(sent))
                pass
            elif sent_parsed[0].orth_ in ('and', 'where', 'when'):
                print("{} is not grammatical (and)".format(sent))
                pass
            elif sent_parsed[-2].orth_ in ('and', 'then'):
                print("{} is not grammatical (and then)".format(sent))
                pass
            elif not any([x.dep_ in ('nsubj', 'nsubjpass') for x in sent_parsed]):
                print("{} is not grammatical (no subj)".format(sent))
                pass
            else:
                print('good! {}'.format(sent))
                return unidecode(detokenizer.detokenize([x.orth_ for x in sent_parsed], return_str=True))
        return ''

    lsmdc['sentence'] = lsmdc['sentence'].apply(check_if_sent_is_grammatical)
    lsmdc = lsmdc[lsmdc['sentence'].str.len() > 0]
    return _lsmdc_to_list(lsmdc)


def load_didemo(split):
    with open(os.path.join(DATA_PATH, 'didemo', '{}_data.json'.format(split)), 'r') as f:
        didemo = json.load(f)
    all_didemo = defaultdict(list)
    for item in didemo:
        # Make didemo look the same as activitynet captions
        timestamp = np.round(np.array(item['times']).mean(0)) * 5
        timestamp[1] += 5  # technically it can be less if we're at the last segment but whatever
        item['timestamp'] = timestamp.tolist()
        all_didemo[item['dl_link']].append(item)

    didemo_list = []
    for k, vid in sorted(all_didemo.items(), key=lambda x: x[0]):
        clip_info = {'duration': max([item['timestamp'][1] for item in vid]),
                     'sentences': [],
                     'timestamps': [],
                     'id': k}
        for item in sorted(vid, key=lambda x: x['timestamp'][0]):
            clip_info['sentences'].append(item['description'])
            clip_info['timestamps'].append(item['timestamp'])
        didemo_list.append(clip_info)
    return didemo_list


def load_anet(split):
    with open(os.path.join(DATA_PATH, 'activitynetcaptions', '{}.json'.format(
            {'train': 'train', 'val': 'val_1', 'test': 'val_2'}[split])), 'r') as f:
        anet = json.load(f)
    for k, v in anet.items():
        v['id'] = k
        v['sentences'] = [remove_allcaps(unidecode(x.strip())) for x in v['sentences']]
    anet = [anet[k] for k in sorted(anet.keys())]
    return anet


def load_ava(split):
    assert split in ('val', 'test')
    ava_annos = pd.read_csv(os.path.join(DATA_PATH, 'ava', 'ava_{}_v2.0.csv'.format(split)),
                            sep=',',
                            header=None,
                            names=['video_id', 'middle_frame_timestamp', 'x1', 'y1', 'x2', 'y2', 'action_id'])

    key = pd.read_csv(os.path.join(DATA_PATH, 'ava', 'ava_action_list_v2.0.csv'), index_col='label_id')

    ava_annos['action'] = ava_annos['action_id'].replace({i: item['label_name'] for i, item in key.iterrows()})
    ava_annos_grouped = ava_annos.groupby('video_id')

    for name, group in ava_annos_grouped:
        results_by_time = {}
        for timestep, subgroup in group.groupby('middle_frame_timestamp'):
            results_by_time[timestep] = set(subgroup['action'].unique().tolist())

def n2w_1k(x, use_ordinal=False):
    if x > 1000:
        return ''
    return num2words(x, to='ordinal' if use_ordinal else 'cardinal')

def _postprocess(sentence):
    """
    make sure punctuation is followed by a space
    :param sentence:
    :return:
    """
    # Aggressively get rid of some punctuation markers
    sent0 = re.sub(r'^.*(\\|/|!!!|~|=|#|@|\*|¡|©|¿|«|»|¬|{|}|\||\(|\)|\+|\]|\[).*$', ' ', sentence, flags=re.MULTILINE|re.IGNORECASE)

    # Less aggressively get rid of quotes, apostrophes
    sent1 = re.sub(r'"', ' ', sent0)
    sent2 = re.sub(r'`', '\'', sent1)

    # match ordinals
    sent3 = re.sub(r'(\d+(?:rd|st|nd))', lambda x: n2w_1k(int(x.group(0)[:-2]), use_ordinal=True), sent2)


    #These things all need to be followed by spaces or else we'll run into problems
    sent4 = re.sub(r'[:;,\"\!\.\-\?](?! )', lambda x: x.group(0) + ' ', sent3)

    #These things all need to be preceded by spaces or else we'll run into problems
    sent5 = re.sub(r'(?! )[-]', lambda x: ' ' + x.group(0), sent4)

    # Several spaces
    sent6 = re.sub(r'\s\s+', ' ', sent5)

    sent7 = sent6.strip()
    return sent7

def load_everything():
    def _stamp(l, stamp_name):
        for x in l:
            x['dataset'] = stamp_name
        return l

    everything = {}
    for split in ('train', 'val', 'test'):
        everything_this_split = []
        # everything_this_split += _stamp(load_mpii_depersonized(split), 'mpii')
        # everything_this_split += _stamp(load_didemo(split), 'didemo')
        everything_this_split += _stamp(load_lsmdc(split), 'lsmdc')
        everything_this_split += _stamp(load_anet(split), 'anet')
        # if split == 'train':
            # everything_this_split += _stamp(load_visual_madlibs(split), 'vml')
        # everything_this_split += _stamp(load_vist(split), 'vist')
        # everything_this_split += _stamp(load_rocstories_nogender(split), 'rocstories')
        everything[split] = everything_this_split

    # Postprocessing
    for split in everything:
        for item in everything[split]:
            for i in range(len(item['sentences'])):
                item['sentences'][i] = _postprocess(item['sentences'][i])

    with open('events-3.json', 'w') as f:
        json.dump(everything, f)
    return everything


#####
# Get what portion is projective
# non_projective = [x for x in anet if any(t0[1] > t1[0]+0.5 for t0, t1 in zip(x['timestamps'][:-1],
#                                                                          x['timestamps'][1:]))]
if __name__ == '__main__':
    # mpii = load_mpii_depersonized('train')
    # lsmdc = load_lsmdc('train')
    # didemo = load_didemo('train')
    # anet = load_anet('train')
    # visualmadlibs = load_visual_madlibs('train')
    everything = load_everything()
    #
    # assert False
    # # Count how many sentences are there
    # mpii_num = sum([len(item['sentences']) for item in mpii])
    # lsmdc_num = sum([len(item['sentences']) for item in lsmdc])
    # didemo_num = sum([len(item['sentences']) for item in didemo])
    # anet_num = sum([len(item['sentences']) for item in anet])
    # roc_num = len(roc) * 5
    #
    # spacy = get_spacy_model("en_core_web_sm", pos_tags=True, parse=True, ner=False)
    #
    #
    # def _count_verbs_in_dataset(dataset):
    #     verb_counts = defaultdict(int)
    #     for x in tqdm(dataset):
    #         for sent in x['sentences']:
    #             for tok in spacy(sent):
    #                 if tok.pos_ == 'VERB':
    #                     verb_counts[tok.lemma_.lower()] += 1
    #     return verb_counts
    #
    #
    # mpii_counts = _count_verbs_in_dataset(mpii)
    # anet_counts = _count_verbs_in_dataset(anet)
    #
    # # rocstories scrambled:
    # import random
    #
    # random.shuffle(roc)
    #
    #
    # def scrambled_stories():
    #     for story in roc:
    #         perm = np.random.permutation(5)
    #         unperm = np.argsort(perm)
    #         story_permed = [story['sentences'][i] for i in perm]
    #         yield story_permed, unperm
