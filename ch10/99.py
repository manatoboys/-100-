from flask import Flask, request, render_template_string

app = Flask(__name__)
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

model = None
tokenizer_src = None
tokenizer_tgt = None
tgt_itos = None
tgt_stoi = None
src_stoi = None
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
        return jp,en

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
        vocab_src = self.build_vocab_src(jp_list, self.src_tokenizer)
        vocab_tgt = self.build_vocab_tgt(en_list, self.tgt_tokenizer)
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

        dataloader = DataLoader(dataset, batch_size=200, collate_fn=collate_fn, num_workers = 16, shuffle=True)

        return dataloader

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    src_batch = pad_sequence(src_batch, padding_value=1,  batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=1, batch_first=True)
    return src_batch, tgt_batch

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


def translate(model, text, tgt_itos, tgt_stoi, src_stoi, tokenizer_src, device):
    with torch.no_grad():
        model.eval()
        model.to(device)
        text_index = [src_stoi[token] if token in src_stoi else src_stoi['<unk>'] for token in tokenizer_src(text)]
        src = torch.tensor(text_index).unsqueeze(0).to(device)
        tgt = torch.tensor([tgt_stoi["<bos>"]]).unsqueeze(0).to(device)
        finish_index = tgt_stoi["<eos>"]
        for i in range(100):
            pred=model(src, tgt)
            next_word = pred.argmax(dim=2)
            tgt = torch.cat((tgt, next_word[:,-1].unsqueeze(0)), dim=1)
            # print(f"tgt:{tgt}")
            english_index = tgt[0,1:]
            if tgt[0,-1]==finish_index:
                english_index = english_index[:-1]
                break
        english_text = " ".join([tgt_itos[en] for en in english_index])
    return english_text



@app.before_request
def initialize():
    global model, tokenizer_src, tokenizer_tgt, tgt_itos, tgt_stoi, src_stoi
    PAD_IDX = 1
    JP_TRAIN_FILE_PATH = "./kftt-data-1.0/data/orig/kyoto-train.ja"
    EN_TRAIN_FILE_PATH = "./kftt-data-1.0/data/orig/kyoto-train.en"

    logger.info("Preparing...")
    with open(JP_TRAIN_FILE_PATH, "r", encoding="utf-8")as f:
        train_jp_list = f.readlines()
        train_jp_list = [jp.strip("\n") for jp in train_jp_list]

    with open(EN_TRAIN_FILE_PATH, "r", encoding="utf-8")as f:
        train_en_list = f.readlines()
        train_en_list = [en.strip("\n") for en in train_en_list]

    logger.info("Loading tokenizers...")
    tokenizer_src = get_tokenizer('spacy', language='ja_core_news_sm')
    tokenizer_tgt = get_tokenizer('spacy', language='en_core_web_sm')
    dataloader_creater = DataLoaderCreater(tokenizer_src, tokenizer_tgt)
    dataloader_creater.create_dataloader(train_jp_list, train_en_list, collate_fn=collate_fn)  # create_dataloaderを実行することで語彙が作成される
    tgt_itos = dataloader_creater.vocab_tgt_itos  # 出力結果をindex→英文に変換
    tgt_stoi = dataloader_creater.vocab_tgt_stoi
    src_stoi = dataloader_creater.vocab_src_stoi

    # モデルのハイパーパラメータ
    embedding_dim = 512
    num_heads = 4
    num_layers = 4

    model = TransformerModel(dataloader_creater.vocab_size_src, dataloader_creater.vocab_size_tgt, embedding_dim, num_heads, num_layers)

    # モデルの状態をロード
    model.load_state_dict(torch.load('model_weight_4.pth'))
    model.to(device)
    logger.info("Model and vocabularies loaded.")




@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    if request.method == 'POST':
        text = request.form['text']
        result = translate(model, text, tgt_itos, tgt_stoi, src_stoi, tokenizer_src, device)
    return render_template_string('''
        <!doctype html>
        <html lang="ja">
        <head>
            <meta charset="utf-8">
            <title>Text Modifier</title>
        </head>
        <body>
            <h1>機械翻訳システム</h1>
            <form method="post">
                <input type="text" name="text" placeholder="テキストを入力してください">
                <input type="submit" value="送信">
            </form>
            <p>翻訳結果: {{ result }}</p>
        </body>
        </html>
    ''', result=result)

if __name__ == '__main__':
    app.run(debug=False, port=5050)