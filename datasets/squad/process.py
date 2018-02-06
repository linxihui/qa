from __future__ import print_function
import json
import codecs
import nltk
import re
from unidecode import unidecode
from tqdm import tqdm
import numpy
# import spacy


# nlp = spacy.load('en')


# with codecs.open('train-v1.1.json', 'r', 'utf-8') as f:
with codecs.open('dev-v1.1.json', 'r', 'utf-8') as f:
    data = json.load(f)


ids, questions, stories_id = [], [], []
ans_char_idx, ans_word_idx, ans_strs = [], [], []
all_story_ids, all_stories = [], []


errors = []


def rm_punkt(x):
    return re.sub(r'\W', '', x, flags=re.UNICODE)


def word_tokenize(x):
    x = re.sub(u"\s'(\w+)", r" ' \1", x, flags=re.UNICODE)
    # x = re.sub(u"(\w)'(?!s|ve|re")
    x = re.sub(ur'(-|\u2013|\u2014)', r' - ', x, flags=re.UNICODE)
    x = x.replace(u'\u201d', '"').replace('.[', '. [').replace('+', ' + ')  # Beyonce.[note 1] xxxx
    tokens = nltk.word_tokenize(x)
    # tokens = [t.text for t in nlp.tokenizer(x)]
    tokens = [t.replace("''", '"').replace("``", '"') for t in tokens]
    tokens = (re.search(r'^(\W)(\w{3,})$', t).groups() if re.match(r'\W\w{3,}$', t) else [t] for t in tokens)
    # tokens = ([t[0], t[1]] if re.match('\w\W$', t) else [t] for t in tokens)  # Jay  Z. She xxxx
    tokens = [t for ts in tokens for t in ts]
    return tokens
    

for ai, a in enumerate(data['data']):
    for pi, p in enumerate(a['paragraphs']):
        sid = u'{}#{}'.format(a['title'], pi)
        all_story_ids.append(sid)
        story = p['context']
        story_ = word_tokenize(story)
        all_stories.append(story_)
        for qi, q in enumerate(p['qas']):
            ans = q['answers'][0]
            s = ans['answer_start']
            text = ans['text']
            if a['title'] == u'Fr\xe9d\xe9ric_Chopin' and ans['text'] == u'7' and ans['answer_start'] == 391:   # error
                text = 'seven'
                s = 332
            elif q['id'] in ['56ce726faab44d1400b88793', '56cc57466d243a140015ef24']:
                print('SKIP')  # iPhone -> iP, one
                print('----')
                continue
                # s = 302
                # text = 'three'
            elif q['id'] == '56ce750daab44d1400b887b4':
                s = 396
            elif q['id'] in ['56cee58daab44d1400b88c1c', '56cf5710aab44d1400b89061'] and ans['text'] == '10':
                s = 242
            text_ = word_tokenize(text)
            if re.match(r'\W$', text_[-1]):  # mistake of including ending punctuation, e.g., 'Robert.', but not 'U.K.'
                text = ans['text'][:-1]
            e = s + len(text)
            if e != len(story) and re.match('\w', story[e]):
                if sid == 'Spectre_(2015_film)#3':
                    if ans['text'] == 'M':
                        s = 523
                        e = s + len(text)
                    if ans['text'] == 'C':
                        s = 571
                        e = s + len(text)
                else:
                    for e in xrange(e + 1, len(story)):
                        if not re.match('\w', story[e]):
                            text = story[s:e]
                            break
                    print('Q:', q['question'])
                    print(story[s:e] + '\n' + ans['text'])
                    print('-----')
            ws = len(word_tokenize(story[:s]))
            we = len(word_tokenize(story[:e]))
            assert text == story[s:e]
            text_ = word_tokenize(text)
            tl = len(text_)
            if ' '.join(story_[ws:we]) != ' '.join(text_):  # tolerance
                if len(text_) == we - ws:
                    if ' '.join(story_[ws - 1:we - 1]) == ' '.join(text_):
                        ws, we = ws - 1, we - 1
                        print('-')
                        print('----')
                    elif ' '.join(story_[ws + 1:we + 1]) == ' '.join(text_):
                        ws, we = ws + 1, we + 1
                        print('+')
                        print('-----')
                else:
                    if ' '.join(story_[ws:ws + tl]) == ' '.join(text_):
                        we = ws + tl
                        print('we = ws + tl')
                        print('-----')
                    elif ' '.join(story_[we - tl:we]) == ' '.join(text_):
                        ws = we - tl
                        print('ws = we + tl')
                        print('-----')
                if ' '.join(story_[ws:we]) != ' '.join(text_):
                    if len(text) > 5 and rm_punkt(text) == rm_punkt(''.join(story_[ws:we])):
                        print(text_, ' :+ ', story_[ws:we])
                        text_ = story_[ws:we]
                        print('-->>')
                if len(text_) == 1:
                    zi = numpy.where([z == text for z in story_])[0]
                    if len(zi) == 1:
                        ws, we = zi[0], zi[0] + 1
                        print('<:')
                        print('----')
            if ' '.join(story_[ws:we]) == ' '.join(text_):
                ids.append(q['id'])
                questions.append(word_tokenize(q['question']))
                stories_id.append(sid)
                ans_char_idx.append((s, e))
                ans_word_idx.append((ws, we))
                ans_strs.append(text)
            else:
                print('e------------------------->')
                print(ai, pi, qi)
                print('original:', ' | '.join([story[s-20:s], story[s:e], story[e:e+20]]))
                print('tokenized:', story_[ws-1:we+2])
                print('selected:', story_[ws:we])
                print('text:', text)
                print('text_:', text_)
                print('<-------------------------e')
                errors.append((ai, pi, qi))



output = {'qid': ids,
          'question': questions,
          'story_id': stories_id,
          'ans_word_idx': ans_word_idx,
          'ans_char_idx': ans_char_idx,
          'ans_strs': ans_strs,
          'story': dict(zip(all_story_ids, all_stories))}

with codecs.open('SQuAD_tokenized_dev.json', mode='w', encoding='utf-8') as f:
    json.dump(output, f)
