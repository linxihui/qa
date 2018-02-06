import codecs
import json
import numpy
import pandas
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


fX = 'float32'
intX = 'int64'

with codecs.open('SQuAD_tokenized.json', 'r', 'utf-8') as f:
    data = json.load(f)

with codecs.open('SQuAD_tokenized_dev.json', 'r', 'utf-8') as f:
    dev = json.load(f)

vocab = {w for s in data['story'].values() + data['question'] + dev['story'].values() + dev['question'] for w in s}

vocab = ['<pad>', '<unk>'] + list(vocab)
word2id = {w: i for i, w in enumerate(vocab)}

'''
embedding = []
embedding_vocab = []
with codecs.open('/home/eric/Downloads/glove.840B.300d.txt', 'r', 'utf-8') as f:
    for i, l in enumerate(f):
        if i % 10000 == 0:
            print(i)
        l = l.strip().split()
        w = ' '.join(l[:-300])
        if w in vocab:
            embedding_vocab.append(w)
            embedding.append([float(li) for li in l[-300:]])
embedding2 = pandas.DataFrame(numpy.asarray(embedding, dtype=fX), index=embedding_vocab)
embedding2.to_hdf('embedding_squad.h5', key='glove_at_squad')
'''



embedding = pandas.read_hdf('embedding_squad.h5')
embedding = embedding.loc[vocab]
embedding.fillna(0., inplace=True)


qids = data['qid']
sids = data['story_id']
questions = [[word2id.get(w, 1) for w in q] for q in data['question']]
all_stories = {sid: [word2id.get(w, 1) for w in s] for sid, s in data['story'].iteritems()}

# ans_rng = numpy.asarray(data['ans_word_idx'], dtype=intX)
ans_rng = data['ans_word_idx']


def take(x, idx_list):
    return [x[i] for i in idx_list]


def pad(x, maxlen=None, dtype='int64'):
    if not maxlen:
        maxlen = max(len(xi) for xi in x)
    x = (xi[:maxlen] for xi in x)
    x = [xi + [0] * (maxlen - len(xi)) for xi in x]
    return numpy.asarray(x, dtype=dtype)
    

def to_variable(cuda, *args):
    args = [Variable(torch.from_numpy(a)) for a in args]
    if cuda:
        args = [a.cuda() for a in args]
    if len(args) == 1:
        return args[0]
    else:
        return args


def call_rnn(rnn, x, lens=None):
    if lens is None:
        lens = numpy.count_nonzero(x.data.cpu().numpy().any(-1), 1).tolist()
    argsort = numpy.ascontiguousarray(numpy.argsort(lens)[::-1])
    lens = [lens[i] for i in argsort]
    x = x[torch.from_numpy(argsort)]
    x = pack_padded_sequence(x, lens, batch_first=True)
    output, hidden = rnn(x)
    output, _ = pad_packed_sequence(output, batch_first=True)
    return output, hidden
    

def masked_softmax(x, msk=None):
    if msk is None:
        msk = (x > 0).type_as(x)
    attn = (x - x.max(-1, keepdim=True)[0]).exp()
    if msk is not None:
        attn = attn * msk
    attn = attn / attn.sum(-1, keepdim=True).clamp(1e-6)
    return attn


def data_gen(sids, questions, all_stories, ans_rng, batch_size=16, onehot=False, shuffle=True):
    N = len(sids)
    indices = numpy.arange(N)
    if shuffle:
        numpy.random.shuffle(indices)
    for s in xrange(0, N + batch_size - 1, batch_size):
        e = s + batch_size
        qs = take(questions, indices[s:e])
        # sort by len
        len_arg = numpy.argsort([-len(q) for q in qs])
        idx = indices[s:e][len_arg]

        qs = take(questions, idx)
        q_lens = numpy.asarray([len(q) for q in qs], dtype=intX)
        qs = pad(qs)

        ids = take(sids, idx)
        ss = take(all_stories, ids)
        s_lens = numpy.asarray([len(s) for s in ss], dtype=intX)
        ss = pad(ss)
        ans = numpy.asarray(take(ans_rng, idx), dtype=intX)
        if onehot:
            ans2 = numpy.eye(ss.shape[1])[ans]
        yield qs, ss, ans, q_lens, s_lens
            

class AttnQA(nn.Module):
    def __init__(self, hidden, embedding, out_rank=1):
        super(AttnQA, self).__init__()
        self.hidden = hidden
        self.embedding = embedding
        self.out_rank = out_rank
        embedding_dim = embedding.shape[-1]
        self.emb = nn.Embedding(len(vocab), embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden // 2, batch_first=True, bidirectional=True)
        self.start_rnn = nn.LSTM(2 * hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.end_rnn = nn.LSTM(2 * hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.start_layer = nn.Linear(hidden, out_rank, bias=False)
        self.end_layer = nn.Linear(hidden, out_rank, bias=False)

    def forward(self, qs, ss, q_lens, s_lens):
        e_qs = self.emb(qs)
        re_qs, _ = call_rnn(self.rnn, e_qs)
        e_ss = self.emb(ss)
        re_ss, _ = call_rnn(self.rnn, e_ss)

        m = re_ss.bmm(re_qs.permute(0, 2, 1))
        msk = (m > 0).type_as(m)
        attn = masked_softmax(m, msk)
        c2q = attn.bmm(re_qs)

        re_ssq = torch.cat([re_ss, c2q], -1)
        re_ssq, _ = call_rnn(self.start_rnn, re_ssq)
        re_ssq = re_ssq.contiguous()
        start = self.start_layer(re_ssq.view(-1, self.hidden)).view(re_ssq.size(0), -1, self.out_rank)
        
        re_ssq_end, _ = call_rnn(self.end_rnn, torch.cat([re_ss, re_ssq], -1))
        re_ssq_end = re_ssq.contiguous()
        end = self.end_layer(re_ssq_end.view(-1, self.hidden)).view(re_ssq.size(0), -1, self.out_rank)

        join = start.bmm(end.permute(0, 2, 1))

        '''
        rg = torch.arange(join.size(1))
        join_mask = Variable((rg.unsqueeze(0) >= rg.unsqueeze(1)).unsqueeze(0)).type_as(join)
        join = join * join_mask
        '''
        # return join
        return start.squeeze(-1), end.squeeze(-1)


def criterion2(ans, start, end, msk):
    index = torch.arange(ans.size(0)).long()
    start = start - start.max(1, keepdim=True)[0].detach()
    end = end - end.max(1, keepdim=True)[0].detach()
    start_exp = start.exp() * msk
    end_exp = end.exp() * msk
    lss = -start[index, ans[:, 0]] + start_exp.sum(1).log()
    lss = lss - end[index, ans[:, 1]] + end_exp.sum(1).log()
    return lss.mean()
    

def criterion(join, ans):
    rg = torch.arange(join.size(1))
    join_mask = Variable((rg.unsqueeze(0) >= rg.unsqueeze(1)).unsqueeze(0)).type_as(join)
    join = join * join_mask
    join_max = join.max(2, keepdim=True)[0].max(1, keepdim=True)[0].detach()
    join = join - join_max
    join_exp = join.exp() * join_mask
    z_exp_sum = join_exp.sum(2).sum(1)

    index = torch.arange(join.size(0)).long()
    z = join[index, ans[:, 0], ans[:, 0]]
    nll = -z + z_exp_sum.log()
    return nll.mean()


def get_start_end(join):
    rg = torch.arange(join.size(1))
    join_mask = Variable((rg.unsqueeze(0) >= rg.unsqueeze(1)).unsqueeze(0)).type_as(join)
    join = join * join_mask - 9999. * (1. - join_mask)
    score, end_idx = join.max(2)
    _, start_idx = score.max(1)

    index = torch.arange(join.size(0)).long()
    end_idx = end_idx[index, start_idx]
    start_idx, end_idx = start_idx.cpu().data.numpy(), end_idx.cpu().data.numpy()
    return numpy.stack([start_idx, end_idx], 1)


def get_start_end2(start, end, msk):
    index = torch.arange(ans.size(0)).long()
    si = (start * msk - 9999. * (1. - msk)).max(1)[0].cpu().data.numpy()
    ei = (end * msk - 9999. * (1. - msk)).max(1)[0].cpu().data.numpy()
    return numpy.stack([si, ei], 1)


def get_acc(pred_ans, true_ans):
    # true_ans = ans.cpu().data.numpy()
    acc = (pred_ans == true_ans)
    return acc.mean(0), acc.prod(1).mean(0)


model = AttnQA(100, embedding, out_rank=1)


tr = data_gen(sids, questions, all_stories, ans_rng, batch_size=16, onehot=False)
batch = tr.next()
qs, ss, ans, q_lens, s_lens = to_variable(False, *batch)



# join = model(qs, ss,  q_lens, s_lens)

opt = optim.Adam(model.parameters())


batch_size = 16
for epoch in range(10):
    tr = data_gen(sids, questions, all_stories, ans_rng, batch_size=batch_size, onehot=False)
    n_batch = (len(questions) + batch_size - 1) // batch_size
    p = tqdm(tr, total=n_batch)
    for batch in p:
        opt.zero_grad()
        qs, ss, ans, q_lens, s_lens = to_variable(False, *batch)
        # join = model(qs, ss, q_lens, s_lens)
        # loss = criterion(join, ans)
        # pred_ans = get_start_end(join)
        start, end = model(qs, ss, q_lens, s_lens)
        msk = (ss > 0).type(torch.FloatTensor)
        loss = criterion2(ans, start, end, msk)
        pred_ans = get_start_end2(start, end, msk)
        acc = get_acc(pred_ans, ans.cpu().data.numpy())
        loss.backward()
        opt.step()
        mssg = 'loss=%.2f, s_acc=%.2f, e_acc=%.2f, acc=%.2f' % (loss.cpu().data.numpy()[0], acc[0][0], acc[0][1], acc[1])
        p.set_description(mssg)

