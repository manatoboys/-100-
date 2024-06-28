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

    def build_vocab_src(self, texts, tokenizer):
        counter = Counter()
        for text in texts:
            counter.update(tokenizer(text))
        specials = ['<unk>', '<pad>']
        v = vocab(counter, specials=specials, min_freq=2)   
        v.set_default_index(v['<unk>'])
        return v

    def build_vocab_tgt(self, texts, tokenizer):
        counter = Counter()
        for text in texts:
            counter.update(tokenizer(text))
        specials = ['<unk>', '<pad>', '<bos>', '<eos>']
        v = vocab(counter, specials=specials, min_freq=2)  
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

        dataloader = DataLoader(dataset, batch_size=256, collate_fn=collate_fn, num_workers = 16, shuffle=True)

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



def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_pad_mask(input_word, pad_idx=1):
    pad_mask = input_word == pad_idx
    return pad_mask.float()

#デコーダ
#入力は<start> token1 token2 ... <pad>...  (<end>は含めない)
#ラベルは token1 token2 ... <end> <pad>...  (<start>は含めない)

def train_epoch(model, dataloader, criterion, optimizer, device):
    epoch_loss = 0
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)

        src_pad_mask = create_pad_mask(src).to(device)
        

        mask = tgt != 3
        input_tgt = tgt[mask].view(tgt.size(0), -1) #3は<eos>
        tgt_pad_mask = create_pad_mask(input_tgt).to(device)
        tgt_mask =generate_square_subsequent_mask(input_tgt.size(1), device).to(device)
        targets = tgt[:, 1:]
        
        output = model(src, input_tgt, tgt_mask=tgt_mask, src_padding_mask=src_pad_mask, tgt_padding_mask=tgt_pad_mask)
        output = output.permute(0, 2, 1) #(バッチサイズ、　シークエンス長、　vocab_size) => (バッチサイズ、　vocab_size、　シークエンス長)にする
        # print(f"output:{output.shape}, tgt:{tgt.shape}")
        loss = criterion(output, targets)
        
        mask = (targets != PAD_IDX).float()

        # マスクを適用してロスを再計算
        loss = loss * mask
        loss = loss.sum() / mask.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        torch.cuda.empty_cache()
    return epoch_loss/len(dataloader.dataset)


if __name__ == "__main__":
    PAD_IDX = 1
    JP_TRAIN_FILE_PATH = "./kftt-data-1.0/data/orig/kyoto-train.ja"
    EN_TRAIN_FILE_PATH = "./kftt-data-1.0/data/orig/kyoto-train.en"

    logger.info("Loading tokenizers...")
    tokenizer_src = get_tokenizer('spacy', language='ja_core_news_sm')
    tokenizer_tgt = get_tokenizer('spacy', language='en_core_web_sm')

    logger.info("Loading datasets...")
    with open(JP_TRAIN_FILE_PATH, "r", encoding="utf-8")as f:
        train_jp_list = f.readlines()
        train_jp_list = [jp.strip("\n") for jp in train_jp_list]

    with open(EN_TRAIN_FILE_PATH, "r", encoding="utf-8")as f:
        train_en_list = f.readlines()
        train_en_list = [en.strip("\n") for en in train_en_list]
    
    logger.info("Creating dataloader...")
    dataloader_creater = DataLoaderCreater(tokenizer_src, tokenizer_tgt)
    train_dataloader = dataloader_creater.create_dataloader(jp_list=train_jp_list, en_list=train_en_list, collate_fn=collate_fn)
    vocab_size_src = dataloader_creater.vocab_size_src
    vocab_size_tgt = dataloader_creater.vocab_size_tgt
    
    # モデルのハイパーパラメータ
    embedding_dim = 512
    num_heads = 4
    num_layers = 4
    lr_rate = 1e-5

    # モデルの初期化
    logger.info("Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(vocab_size_src, vocab_size_tgt, embedding_dim, num_heads, num_layers)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)

    # トレーニングループ
    num_epochs = 10
    
    model.train()
    logger.info("Starting training loop...")
    for epoch in tqdm(range(num_epochs)):
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, device)
        logger.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')

    logger.info("Saving model...")
    torch.save(model.state_dict(),'model_weight.pth') #nn.Dataparallelを使用した場合はmodel.state_dictではなくmodel.module.state_dictと書かなければいけない
    logger.info("Training complete.")






##output
# 2024-06-27 22:01:00,043 - INFO - Loading tokenizers...
# 2024-06-27 22:01:02,383 - INFO - Loading datasets...
# 2024-06-27 22:01:02,802 - INFO - Creating dataloader...
# 2024-06-27 22:04:43,253 - INFO - Initializing model...
# 2024-06-27 22:04:47,039 - INFO - Starting training loop...
#   0%|                                                                                              2024-06-27 22:13:58,940 - INFO - Epoch 1, Train Loss: 0.0322
#   3%|█████▏                                                                                                                                                       | 1/30 [09:11<4:26:45, 551.90s/it]2024-06-27 22:22:40,378 - INFO - Epoch 2, Train Loss: 0.0248
#   7%|██████████▍                                                                                                                                                  | 2/30 [17:53<4:09:11, 533.98s/it]2024-06-27 22:31:19,243 - INFO - Epoch 3, Train Loss: 0.0231
#  10%|███████████████▋                                                                                                                                             | 3/30 [26:32<3:57:11, 527.08s/it]2024-06-27 22:40:00,857 - INFO - Epoch 4, Train Loss: 0.0222
#  13%|████████████████████▉                                                                                                                                        | 4/30 [35:13<3:47:27, 524.92s/it]2024-06-27 22:48:41,102 - INFO - Epoch 5, Train Loss: 0.0214
#  17%|██████████████████████████▏                                                                                                                                  | 5/30 [43:54<3:38:00, 523.23s/it]2024-06-27 22:57:21,337 - INFO - Epoch 6, Train Loss: 0.0208
#  20%|███████████████████████████████▍                                                                                                                             | 6/30 [52:34<3:28:53, 522.21s/it]2024-06-27 23:06:06,912 - INFO - Epoch 7, Train Loss: 0.0203
#  23%|████████████████████████████████████▏                                                                                                                      | 7/30 [1:01:19<3:20:36, 523.31s/it]2024-06-27 23:15:34,840 - INFO - Epoch 8, Train Loss: 0.0198
#  27%|█████████████████████████████████████████▎                                                                                                                 | 8/30 [1:10:47<3:17:05, 537.52s/it]2024-06-27 23:24:46,252 - INFO - Epoch 9, Train Loss: 0.0195
#  30%|██████████████████████████████████████████████▌                                                                                                            | 9/30 [1:19:59<3:09:39, 541.86s/it]2024-06-27 23:33:51,305 - INFO - Epoch 10, Train Loss: 0.0191
#  33%|███████████████████████████████████████████████████▎                                                                                                      | 10/30 [1:29:04<3:00:56, 542.85s/it]2024-06-27 23:42:34,307 - INFO - Epoch 11, Train Loss: 0.0188
#  37%|████████████████████████████████████████████████████████▍                                                                                                 | 11/30 [1:37:47<2:49:58, 536.77s/it]2024-06-27 23:51:13,291 - INFO - Epoch 12, Train Loss: 0.0185
#  40%|█████████████████████████████████████████████████████████████▌                                                                                            | 12/30 [1:46:26<2:39:24, 531.36s/it]2024-06-27 23:59:51,704 - INFO - Epoch 13, Train Loss: 0.0182
#  43%|██████████████████████████████████████████████████████████████████▋                                                                                       | 13/30 [1:55:04<2:29:26, 527.44s/it]2024-06-28 00:08:33,199 - INFO - Epoch 14, Train Loss: 0.0180
#  47%|███████████████████████████████████████████████████████████████████████▊                                                                                  | 14/30 [2:03:46<2:20:10, 525.64s/it]2024-06-28 00:17:12,791 - INFO - Epoch 15, Train Loss: 0.0178
#  50%|█████████████████████████████████████████████████████████████████████████████                                                                             | 15/30 [2:12:25<2:10:57, 523.82s/it]2024-06-28 00:25:51,084 - INFO - Epoch 16, Train Loss: 0.0176
#  53%|██████████████████████████████████████████████████████████████████████████████████▏                                                                       | 16/30 [2:21:04<2:01:50, 522.16s/it]2024-06-28 00:34:33,737 - INFO - Epoch 17, Train Loss: 0.0174
#  57%|███████████████████████████████████████████████████████████████████████████████████████▎                                                                  | 17/30 [2:29:46<1:53:09, 522.31s/it]2024-06-28 00:43:13,200 - INFO - Epoch 18, Train Loss: 0.0172
#  60%|████████████████████████████████████████████████████████████████████████████████████████████▍                                                             | 18/30 [2:38:26<1:44:17, 521.45s/it]2024-06-28 00:51:51,902 - INFO - Epoch 19, Train Loss: 0.0170
#  63%|█████████████████████████████████████████████████████████████████████████████████████████████████▌                                                        | 19/30 [2:47:04<1:35:26, 520.63s/it]2024-06-28 01:00:29,128 - INFO - Epoch 20, Train Loss: 0.0168
#  67%|██████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                   | 20/30 [2:55:42<1:26:36, 519.60s/it]2024-06-28 01:09:09,548 - INFO - Epoch 21, Train Loss: 0.0166
#  70%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                              | 21/30 [3:04:22<1:17:58, 519.85s/it]2024-06-28 01:17:49,170 - INFO - Epoch 22, Train Loss: 0.0165
#  73%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                         | 22/30 [3:13:02<1:09:18, 519.78s/it]2024-06-28 01:26:28,486 - INFO - Epoch 23, Train Loss: 0.0163
#  77%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                    | 23/30 [3:21:41<1:00:37, 519.64s/it]2024-06-28 01:35:07,842 - INFO - Epoch 24, Train Loss: 0.0162
#  80%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                               | 24/30 [3:30:20<51:57, 519.56s/it]2024-06-28 01:43:46,295 - INFO - Epoch 25, Train Loss: 0.0160
#  83%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                          | 25/30 [3:38:59<43:16, 519.23s/it]2024-06-28 01:52:25,974 - INFO - Epoch 26, Train Loss: 0.0159
#  87%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                    | 26/30 [3:47:38<34:37, 519.36s/it]2024-06-28 02:01:03,951 - INFO - Epoch 27, Train Loss: 0.0157
#  90%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍               | 27/30 [3:56:16<25:56, 518.95s/it]2024-06-28 02:09:44,826 - INFO - Epoch 28, Train Loss: 0.0156
#  93%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌          | 28/30 [4:04:57<17:19, 519.52s/it]2024-06-28 02:18:22,960 - INFO - Epoch 29, Train Loss: 0.0155
#  97%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊     | 29/30 [4:13:35<08:39, 519.11s/it]2024-06-28 02:27:05,180 - INFO - Epoch 30, Train Loss: 0.0153
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [4:22:18<00:00, 524.60s/it]
# 2024-06-28 02:27:05,181 - INFO - Saving model...
# 2024-06-28 02:27:06,392 - INFO - Training complete.