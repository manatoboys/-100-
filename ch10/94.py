import copy
from heapq import heappush, heappop

import torch

import polars as pl
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import os
import logging
from sacrebleu.metrics import BLEU
import matplotlib.pyplot as plt

logger = logging.getLogger("ログ")
logger.setLevel(logging.DEBUG)

# コンソールハンドラの作成と設定
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# フォーマットの設定
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# ロガーにハンドラを追加
logger.addHandler(ch)


class datasets(Dataset):
    def __init__(self, text, label):
        self.jp_datas = text
        self.en_datas = label

    def __len__(self):
        return len(self.jp_datas)

    def __getitem__(self, index):
        jp = self.jp_datas[index]
        en = self.en_datas[index]
        return jp,en, index

class DataLoaderCreater:

    def __init__(self, src_tokenizer, tgt_tokenizer):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def build_vocab_src(self, texts, tokenizer):
        counter = Counter()
        for text in texts:
            counter.update(tokenizer(text))
        specials = ['<unk>', '<pad>']
        v = vocab(counter, specials=specials, min_freq=1)   #1回しか出てきていない単語は語彙に入れない
        v.set_default_index(v['<unk>'])
        return v

    def build_vocab_tgt(self, texts, tokenizer):
        counter = Counter()
        for text in texts:
            counter.update(tokenizer(text))
        specials = ['<unk>', '<pad>', '<bos>', '<eos>']
        v = vocab(counter, specials=specials, min_freq=1)   #1回しか出てきていない単語は語彙に入れない
        v.set_default_index(v['<unk>'])
        return v

    def convert_text_to_indexes_tgt(self, text, vocab, tokenizer):
        return [vocab['<bos>']] + [
            vocab[token] if token in vocab else vocab['<unk>'] for token in tokenizer(text.strip("\n"))
        ] + [vocab['<eos>']]

    def convert_text_to_indexes_src(self, text, vocab, tokenizer):
        return  [vocab[token] if token in vocab else vocab['<unk>'] for token in tokenizer(text.strip("\n"))]

    def create_dataloader(self, jp_list, en_list, collate_fn):
        vocab_src = self.build_vocab_src(jp_list, tokenizer_src)
        vocab_tgt = self.build_vocab_tgt(en_list, tokenizer_tgt)
        self.vocab_src_itos = vocab_src.get_itos()
        self.vocab_tgt_itos = vocab_tgt.get_itos()
        self.vocab_src_stoi = vocab_src.get_stoi()
        self.vocab_tgt_stoi = vocab_tgt.get_stoi()
        self.vocab_size_src = len(self.vocab_src_stoi)
        self.vocab_size_tgt = len(self.vocab_tgt_stoi)
        jp_en_list = [[jp,en] for jp, en in zip(jp_list, en_list) if len(jp)<100 and len(en)<100] #系列長が250未満のものだけを訓練に使用する
        jp_list = [data[0] for data in jp_en_list]
        en_list = [data[1] for data in jp_en_list]
        src_data = [torch.tensor(self.convert_text_to_indexes_src(jp_data, self.vocab_src_stoi, self.src_tokenizer)) for jp_data in jp_list]
        tgt_data = [torch.tensor(self.convert_text_to_indexes_tgt(en_data, self.vocab_tgt_stoi, self.tgt_tokenizer)) for en_data in en_list]
        dataset = datasets(src_data, tgt_data)

        dataloader = DataLoader(dataset, batch_size=100, collate_fn=collate_fn, num_workers = 16, shuffle=True)

        return dataloader
    
class Test_DataLoaderCreater:

    def __init__(self, src_tokenizer, tgt_tokenizer):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def convert_text_to_indexes_tgt(self, text, vocab, tokenizer):
        return [vocab['<bos>']] + [
            vocab[token] if token in vocab else vocab['<unk>'] for token in tokenizer(text.strip("\n"))
        ] + [vocab['<eos>']]

    def convert_text_to_indexes_src(self, text, vocab, tokenizer):
        return  [vocab[token] if token in vocab else vocab['<unk>'] for token in tokenizer(text.strip("\n"))]

    def create_dataloader(self, jp_list, en_list, src_stoi, tgt_stoi, collate_fn):
        jp_en_list = [[jp,en] for jp, en in zip(jp_list, en_list) if len(jp)<100 and len(en)<100] #系列長が250未満のものだけを訓練に使用する
        self.jp_list = [data[0] for data in jp_en_list]
        self.en_list = [data[1] for data in jp_en_list]
        src_data = [torch.tensor(self.convert_text_to_indexes_src(jp_data, src_stoi, self.src_tokenizer)) for jp_data in self.jp_list]
        tgt_data = [torch.tensor(self.convert_text_to_indexes_tgt(en_data, tgt_stoi, self.tgt_tokenizer)) for en_data in self.en_list]
        dataset = datasets(src_data, tgt_data)

        dataloader = DataLoader(dataset, batch_size=100, collate_fn=collate_fn, num_workers = 16, shuffle=True)

        return dataloader

def collate_fn(batch):
    src_batch, tgt_batch, index_batch = zip(*batch)

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX,  batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, tgt_batch, index_batch

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def get_sin(self,i,k):
        return torch.sin(torch.tensor(i/(10000)**(k/self.embedding_dim)))

    def get_cos(self,i,k):
        return torch.cos(torch.tensor(i/(10000)**(k/self.embedding_dim)))

    def forward(self, x):
        len_sequence = x.shape[1]
        batch_size = x.shape[0]
        pe = torch.zeros(len_sequence, self.embedding_dim)
        for pos in range(len_sequence):
            for i in range(0, self.embedding_dim):
                if i % 2 == 0:
                    pe[pos, i] = self.get_sin(pos, i // 2) ##//は整数除算　5//2 = 2
                else:
                    pe[pos, i] = self.get_cos(pos, i // 2)
        pe = pe.unsqueeze(0).repeat(batch_size, 1, 1).to(x.device)
        return pe + x


# Transformerモデルの定義
class TransformerModel(nn.Module):
    def __init__(self, vocab_size_src, vocab_size_tgt, embedding_dim, num_heads, num_layers,  dropout=0.1):
        super().__init__()
        # Positional Encoderを加算する必要あり
        self.pos_encoding = PositionalEncoding(embedding_dim)
        self.embedding_dim = embedding_dim
        self.embedding_src = nn.Embedding(vocab_size_src, embedding_dim)
        self.embedding_tgt = nn.Embedding(vocab_size_tgt, embedding_dim)
        self.transformer = nn.Transformer(d_model=embedding_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dropout=dropout, batch_first=True, dim_feedforward=512)
        self.fc_out = nn.Linear(embedding_dim, vocab_size_tgt)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        src = self.embedding_src(src)
        tgt = self.embedding_tgt(tgt)
        src = self.pos_encoding(src)
        tgt = self.pos_encoding(tgt)
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
        output = self.fc_out(output)
        return output


class BeamSearchNode(object):
    def __init__(self, wid, logp, length):
        self.wid = wid
        self.logp = logp
        self.length = length

    def eval(self):
        #正則化項を導入することで長い文章を生成しやすくしている。
        return self.logp / float(self.length - 1 + 1e-6)

def beam_search_decoding(model, src, beam_width, n_best, sos_token, eos_token, max_dec_steps, device):
    model.eval()
    src = src.to(device)
    memory = model.embedding_src(src)
    memory = model.pos_encoding(memory)
    memory = model.transformer.encoder(memory)

    batch_size = src.size(0)
    n_best_list = [[] for _ in range(batch_size)]

    for batch_id in range(batch_size):
        end_nodes = []
        decoder_input = torch.tensor([[sos_token]]).to(device)
        node = BeamSearchNode(wid=[sos_token], logp=0, length=1)
        nodes = []
        heappush(nodes, (-node.eval(), id(node), node))
        n_dec_steps = 0

        while True:
            if n_dec_steps > max_dec_steps:
                break

            temp_nodes = []
            for _ in range(min(len(nodes), beam_width)):
                score, _, n = heappop(nodes)
                decoder_input = torch.tensor([n.wid]).to(device)

                if n.wid[-1] == eos_token and len(n.wid) > 1:
                    end_nodes.append((score, id(n), n))
                    if len(end_nodes) >= n_best:
                        break
                    else:
                        continue

                tgt_emb = model.embedding_tgt(decoder_input)
                tgt_emb = model.pos_encoding(tgt_emb)
                decoder_output = model.transformer.decoder(tgt_emb, memory[batch_id:batch_id+1])
                logits = model.fc_out(decoder_output[:, -1, :])
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                topk_log_prob, topk_indexes = torch.topk(log_probs, beam_width)

                for new_k in range(beam_width):
                    decoded_t = topk_indexes[0][new_k].item()
                    logp = topk_log_prob[0][new_k].item()
                    new_wid = n.wid + [decoded_t]
                    new_node = BeamSearchNode(wid=new_wid, logp=n.logp + logp, length=n.length + 1)
                    heappush(temp_nodes, (-new_node.eval(), id(new_node), new_node))

            nodes = temp_nodes
            n_dec_steps += 1

            if len(end_nodes) >= beam_width:
                break

        if len(end_nodes) == 0:
            end_nodes = [heappop(nodes) for _ in range(beam_width)]

        n_best_seq_list = []

        #一番確率が高いやつを抽出
        for score, _id, n in sorted(end_nodes, key=lambda x: x[0]):
            sequence = n.wid
            if sequence[0] == sos_token:
                sequence = sequence[1:]
            if sequence[-1] == eos_token:
                sequence = sequence[:-1]
            n_best_seq_list.append(sequence)
            if len(n_best_seq_list) >= n_best:
                break

        n_best_list[batch_id] = n_best_seq_list

    return n_best_list

def translate_from_index(model, index_list, tgt_itos, device, bos_token=2, eos_token=3):
    with torch.no_grad():
        model.eval()
        model.to(device)
        if index_list[0]==bos_token:
            index_list = index_list[1:]
        if index_list[-1]==eos_token:
            index_list = index_list[:-1]
        #.や,の前に空白がある場合は空白を削除する
        english_text = " ".join([tgt_itos[en] for en in index_list]).replace(' .', '.').replace(' ,', ',').replace(" ' ", "'").replace(" n't", "n't").replace(" 'm", "'m").replace(" 're", "'re").replace(" 've", "'ve").replace(" 'll", "'ll").replace(" 's", "'s")
    return english_text

if __name__ == "__main__":
    PAD_IDX = 1
    JP_TRAIN_FILE_PATH = "./kftt-data-1.0/data/orig/kyoto-train.ja"
    EN_TRAIN_FILE_PATH = "./kftt-data-1.0/data/orig/kyoto-train.en"

    JP_DEV_FILE_PATH = "./kftt-data-1.0/data/orig/kyoto-dev.ja"
    EN_DEV_FILE_PATH = "./kftt-data-1.0/data/orig/kyoto-dev.en"

    logger.info("Loading tokenizers...")
    tokenizer_src = get_tokenizer('spacy', language='ja_core_news_sm')
    tokenizer_tgt = get_tokenizer('spacy', language='en_core_web_sm')

    logger.info("Preparing...")
    with open(JP_TRAIN_FILE_PATH, "r", encoding="utf-8")as f:
        train_jp_list = f.readlines()
        train_jp_list = [jp.strip("\n") for jp in train_jp_list]

    with open(EN_TRAIN_FILE_PATH, "r", encoding="utf-8")as f:
        train_en_list = f.readlines()
        train_en_list = [en.strip("\n") for en in train_en_list]

    with open(JP_DEV_FILE_PATH, "r", encoding="utf-8")as f:
        dev_jp_list = f.readlines()
        dev_jp_list = [jp.strip("\n") for jp in dev_jp_list]

    with open(EN_DEV_FILE_PATH, "r", encoding="utf-8")as f:
        dev_en_list = f.readlines()
        dev_en_list = [en.strip("\n") for en in dev_en_list]

    dataloader_creater = DataLoaderCreater(tokenizer_src, tokenizer_tgt)
    dataloader_creater.create_dataloader(train_jp_list, train_en_list, collate_fn=collate_fn) #create_dataloaderを実行することで語彙が作成される
    tgt_itos = dataloader_creater.vocab_tgt_itos #出力結果をindex→英文に変換
    tgt_stoi = dataloader_creater.vocab_tgt_stoi
    src_stoi = dataloader_creater.vocab_src_stoi
    vocab_size_src = dataloader_creater.vocab_size_src
    vocab_size_tgt = dataloader_creater.vocab_size_tgt
    
    dev_dataloader_creater = Test_DataLoaderCreater(tokenizer_src, tokenizer_tgt)
    dev_dataloader = dev_dataloader_creater.create_dataloader(dev_jp_list, dev_en_list, src_stoi, tgt_stoi, collate_fn=collate_fn) #create_dataloaderを実行することで語彙が作成される
    jp_list = dev_dataloader_creater.jp_list
    en_list = dev_dataloader_creater.en_list
    # モデルのハイパーパラメータ
    embedding_dim = 512
    num_heads = 4
    num_layers = 4
    lr_rate = 1e-4

    # モデルの状態をロード
    logger.info("Loading model...")
    model = TransformerModel(vocab_size_src, vocab_size_tgt, embedding_dim, num_heads, num_layers)
    model.load_state_dict(torch.load('model_weight_3.pth'))
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # パラメータ設定
    n_best = 1 #これ必要ない
    bos_token = 2  # <sos> トークンのインデックス
    eos_token = 3  # <eos> トークンのインデックス
    max_dec_steps = 100
    scores = []
    beam_width_list =[1, 5, 10, 20]
    logger.info("Decoding...")
    for beam_width in beam_width_list:
        all_predicted = []
        predicted_text_list = []
        refs = []
        index_list =[]
        for src, tgt, index in tqdm(dev_dataloader):
            output_sequences = beam_search_decoding(model, src, beam_width, n_best, bos_token, eos_token, max_dec_steps, device)
            predicted_list = [data[0] for data in output_sequences]
            all_predicted += predicted_list
            index_list += index
            break
        for en_index in all_predicted:
            predicted_text = translate_from_index(model, en_index, tgt_itos, device)
            predicted_text_list.append(predicted_text)
        refs = []
        for i in index_list:
            refs.append(en_list[i])
        max_index = max(index_list)
        
        bleu = BLEU()
        score = bleu.corpus_score(predicted_text_list, [refs])
        scores.append(score.score)
        print(f"beam_width:{beam_width}, score:{score}")

    plt.plot(beam_width_list, scores)

    # グラフのタイトルとラベルを設定
    plt.title('beam_width Score')
    plt.xlabel('beam_width')
    plt.ylabel('score')

    # グラフを表示
    plt.savefig("94_graph_3.png")


    '''
    beam_width:1, score:BLEU = 3.83 41.4/11.6/4.6/2.2 (BP = 0.460 ratio = 0.563 hyp_len = 13676 ref_len = 24281)
    '''

#graph_4 weight4 & alpha=0.8

#weight4
'''
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [04:32<00:00, 38.89s/it]
max(index_list):{max_index}
beam_width:1, score:BLEU = 8.92 36.5/12.7/6.3/3.3 (BP = 0.898 ratio = 0.903 hyp_len = 4905 ref_len = 5432)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [16:25<00:00, 140.84s/it]
max(index_list):{max_index}
beam_width:5, score:BLEU = 9.58 41.7/15.7/8.3/4.7 (BP = 0.758 ratio = 0.783 hyp_len = 4255 ref_len = 5432)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [29:42<00:00, 254.71s/it]
max(index_list):{max_index}
beam_width:10, score:BLEU = 9.18 41.9/15.8/8.5/4.4 (BP = 0.732 ratio = 0.762 hyp_len = 4141 ref_len = 5432)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [55:42<00:00, 477.46s/it]
max(index_list):{max_index}
beam_width:20, score:BLEU = 9.26 42.5/16.1/8.4/4.6 (BP = 0.725 ratio = 0.757 hyp_len = 4110 ref_len = 5432)
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [1:22:21<00:00, 705.92s/it]
max(index_list):{max_index}
beam_width:30, score:BLEU = 9.42 41.3/15.9/8.4/4.6 (BP = 0.748 ratio = 0.775 hyp_len = 4210 ref_len = 5432)
'''