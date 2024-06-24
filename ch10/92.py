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

    def build_vocab(self, texts, tokenizer):
        counter = Counter()
        for text in texts:
            counter.update(tokenizer(text))
        specials = ['<unk>', '<pad>', '<start>', '<end>']
        v = vocab(counter, specials=specials, min_freq=2)   #1回しか出てきていない単語は語彙に入れない
        v.set_default_index(v['<unk>'])
        return v

    def convert_text_to_indexes(self, text, vocab, tokenizer):
        return [vocab['<start>']] + [
            vocab[token] if token in vocab else vocab['<unk>'] for token in tokenizer(text.strip("\n"))
        ] + [vocab['<end>']]

    def create_dataloader(self, jp_list, en_list, collate_fn):
        vocab_src = self.build_vocab(jp_list, tokenizer_src)
        vocab_tgt = self.build_vocab(en_list, tokenizer_tgt)
        self.vocab_src_itos = vocab_src.get_itos()
        self.vocab_tgt_itos = vocab_tgt.get_itos()
        self.vocab_src_stoi = vocab_src.get_stoi()
        self.vocab_tgt_stoi = vocab_tgt.get_stoi()
        self.vocab_size_src = len(self.vocab_src_stoi)
        self.vocab_size_tgt = len(self.vocab_tgt_stoi)
        jp_en_list = [[jp,en] for jp, en in zip(jp_list, en_list) if len(jp)<150 and len(en)<150] #系列長が250未満のものだけを訓練に使用する
        jp_list = [data[0] for data in jp_en_list]
        en_list = [data[1] for data in jp_en_list]
        src_data = [torch.tensor(self.convert_text_to_indexes(jp_data, self.vocab_src_stoi, self.src_tokenizer)) for jp_data in jp_list]
        tgt_data = [torch.tensor(self.convert_text_to_indexes(en_data, self.vocab_tgt_stoi, self.tgt_tokenizer)) for en_data in en_list]
        dataset = datasets(src_data, tgt_data)

        dataloader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn, num_workers = 16, shuffle=True)

        return dataloader

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX,  batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, tgt_batch

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def get_sin(self,i,k, len_sequence):
        return torch.sin(torch.tensor(i/(10000)**(k/len_sequence)))

    def get_cos(self,i,k, len_sequence):
        return torch.cos(torch.tensor(i/(10000)**(k/len_sequence)))

    def forward(self, x):
        len_sequence = x.shape[1]
        batch_size = x.shape[0]
        pe = torch.zeros(len_sequence, self.embedding_dim)
        for pos in range(len_sequence):
            for i in range(0, self.embedding_dim):
                if i % 2 == 0:
                    pe[pos, i] = self.get_sin(pos, i // 2, len_sequence) ##//は整数除算　5//2 = 2
                else:
                    pe[pos, i] = self.get_cos(pos, i // 2, len_sequence)
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



def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_pad_mask(input_word, pad_idx=1):
    pad_mask = input_word == pad_idx
    return pad_mask.float()

def train_epoch(model, dataloader, criterion, optimizer, device):
    # model.train()
    epoch_loss = 0
    for batch_idx, (src, tgt) in enumerate(tqdm(dataloader)):
        src = src.to(device)
        tgt = tgt.to(device)
        src_pad_mask = create_pad_mask(src).to(device)
        tgt_pad_mask = create_pad_mask(tgt).to(device)
        tgt_mask =generate_square_subsequent_mask(tgt.size(1), device).to(device)
        # src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt, PAD_IDX, device)

        output = model.forward(src, tgt, tgt_mask=tgt_mask, src_padding_mask=src_pad_mask, tgt_padding_mask=tgt_pad_mask)
        output = output.permute(0, 2, 1) #(バッチサイズ、　シークエンス長、　vocab_size) => (バッチサイズ、　vocab_size、　シークエンス長)にする
        # print(f"output:{output.shape}, tgt:{tgt.shape}")
        loss = criterion(output, tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        torch.cuda.empty_cache()
    return epoch_loss/len(dataloader)

def translate(model, text, tgt_itos, src_stoi, tokenizer_src, device):
    model.eval()
    model.to(device)
    text_index = [src_stoi ['<start>']] + [src_stoi [token] if token in src_stoi else src_stoi['<unk>'] for token in tokenizer_src(text)] + [src_stoi ['<end>']]
    src = torch.tensor(text_index).to(device)
    tgt = src_stoi["<start>"].to(device)
    finish_index = src_stoi["<end>"]
    len_seq = 1
    while tgt[-1]!=finish_index:
        len_seq += 1
        pred = model(src,tgt)
        tgt = pred[:len_seq]
    english_index = tgt[1:-1]
    english_text = "".join([en if en in tgt_itos else src_stoi["<unk>"] for en in english_index])
    return english_text
        
    
    
    


if __name__ == "__main__":
    PAD_IDX = 1
    JP_TRAIN_FILE_PATH = "./kftt-data-1.0/data/orig/kyoto-train.ja"
    EN_TRAIN_FILE_PATH = "./kftt-data-1.0/data/orig/kyoto-train.en"

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
    dataloader_creater = DataLoaderCreater(tokenizer_src, tokenizer_tgt)
    tgt_itos = dataloader_creater.vocab_tgt_itos #出力結果をindex→英文に変換
    src_stoi = dataloader_creater.vocab_src_stoi
    vocab_size_src = dataloader_creater.vocab_size_src
    vocab_size_tgt = dataloader_creater.vocab_size_tgt
    
    text = "私は日本人です。"
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logger.info("Loading model...")
    model = torch.load("model_weight.pth")
    logger.info("Translating...")
    translate(model, text, tgt_itos, src_stoi, tokenizer_src, device)

 
