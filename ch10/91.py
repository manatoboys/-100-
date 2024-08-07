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
        v = vocab(counter, specials=specials, min_freq=1)   
        v.set_default_index(v['<unk>'])
        return v

    def build_vocab_tgt(self, texts, tokenizer):
        counter = Counter()
        for text in texts:
            counter.update(tokenizer(text))
        specials = ['<unk>', '<pad>', '<bos>', '<eos>']
        v = vocab(counter, specials=specials, min_freq=1)  
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
        
        #修正箇所　系列庁制限なくした
        jp_en_list = [[jp,en] for jp, en in zip(jp_list, en_list) if len(jp)<100 and len(en)<100] #系列長が100未満のものだけを訓練に使用する
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
    for batch_idx, (src, tgt) in enumerate(tqdm(dataloader)):
        src = src.to(device)
        tgt = tgt.to(device)

        src_pad_mask = create_pad_mask(src).to(device)
        

        mask = tgt != 3
        input_tgt = tgt[mask].view(tgt.size(0), -1) #3は<eos>
        tgt_pad_mask = create_pad_mask(input_tgt).to(device)
        tgt_mask =generate_square_subsequent_mask(input_tgt.size(1), device).to(device)
        targets = tgt[:, 1:]
        
        output = model(src, input_tgt, tgt_mask=tgt_mask, src_padding_mask=src_pad_mask, tgt_padding_mask=tgt_pad_mask)
        target_shape = (-1, output.shape[2]) 
        output = output.reshape(target_shape)
        targets = targets.reshape(-1)
        # output = output.permute(0, 2, 1) #(バッチサイズ、　シークエンス長、　vocab_size) => (バッチサイズ、　vocab_size、　シークエンス長)にする
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
    lr_rate = 1e-4

    # モデルの初期化
    logger.info("Initializing model...")
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(vocab_size_src, vocab_size_tgt, embedding_dim, num_heads, num_layers)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)

    # トレーニングループ
    num_epochs = 100
    
    model.train()
    logger.info("Starting training loop...")
    for epoch in tqdm(range(num_epochs)):
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, device)
        logger.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')

    logger.info("Saving model...")
    torch.save(model.state_dict(),'model_weight_4.pth') #nn.Dataparallelを使用した場合はmodel.state_dictではなくmodel.module.state_dictと書かなければいけない
    logger.info("Training complete.")

#weight2は一回しか出現してない単語も語彙に含めている
#weight3は学習率を1e-4

#weight4は3の100エポックver

#weight3が一番良さそう
##output
# 2024-06-28 14:59:49,502 - INFO - Loading tokenizers...
# 2024-06-28 14:59:53,058 - INFO - Loading datasets...
# 2024-06-28 14:59:53,484 - INFO - Creating dataloader...
# 2024-06-28 15:03:31,316 - INFO - Initializing model...
# 2024-06-28 15:03:35,330 - INFO - Starting training loop...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [07:34<00:00,  1.64it/s]
# 2024-06-28 15:11:09,949 - INFO - Epoch 1, Train Loss: 0.0321██████████████████████████████████████████████████████████████| 744/744 [07:34<00:00,  1.52it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [07:33<00:00,  1.64it/s]
# 2024-06-28 15:18:43,501 - INFO - Epoch 2, Train Loss: 0.0247██████████████████████████████████████████████████████████████| 744/744 [07:33<00:00,  1.72it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [07:35<00:00,  1.63it/s]
# 2024-06-28 15:26:19,299 - INFO - Epoch 3, Train Loss: 0.0231██████████████████████████████████████████████████████████████| 744/744 [07:35<00:00,  1.75it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [07:34<00:00,  1.64it/s]
# 2024-06-28 15:33:53,599 - INFO - Epoch 4, Train Loss: 0.0222██████████████████████████████████████████████████████████████| 744/744 [07:34<00:00,  1.68it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [07:34<00:00,  1.64it/s]
# 2024-06-28 15:41:28,016 - INFO - Epoch 5, Train Loss: 0.0214██████████████████████████████████████████████████████████████| 744/744 [07:34<00:00,  1.66it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [07:32<00:00,  1.64it/s]
# 2024-06-28 15:49:00,391 - INFO - Epoch 6, Train Loss: 0.0208██████████████████████████████████████████████████████████████| 744/744 [07:32<00:00,  1.70it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [07:33<00:00,  1.64it/s]
# 2024-06-28 15:56:33,870 - INFO - Epoch 7, Train Loss: 0.0203██████████████████████████████████████████████████████████████| 744/744 [07:33<00:00,  1.54it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [07:35<00:00,  1.63it/s]
# 2024-06-28 16:04:09,726 - INFO - Epoch 8, Train Loss: 0.0199██████████████████████████████████████████████████████████████| 744/744 [07:35<00:00,  1.65it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [07:34<00:00,  1.64it/s]
# 2024-06-28 16:11:43,936 - INFO - Epoch 9, Train Loss: 0.0195██████████████████████████████████████████████████████████████| 744/744 [07:34<00:00,  1.72it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [07:33<00:00,  1.64it/s]
# 2024-06-28 16:19:17,509 - INFO - Epoch 10, Train Loss: 0.0192█████████████████████████████████████████████████████████████| 744/744 [07:33<00:00,  1.73it/s]
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [1:15:42<00:00, 454.22s/it]
# 2024-06-28 16:19:17,509 - INFO - Saving model...
# 2024-06-28 16:19:18,538 - INFO - Training complete.



#weight_3 output
# 2024-06-28 23:07:06,557 - INFO - Loading tokenizers...
# 2024-06-28 23:07:10,039 - INFO - Loading datasets...
# 2024-06-28 23:07:10,466 - INFO - Creating dataloader...
# 2024-06-28 23:10:48,866 - INFO - Initializing model...
# 2024-06-28 23:10:53,697 - INFO - Starting training loop...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:18<00:00,  1.33it/s]
# 2024-06-28 23:20:12,491 - INFO - Epoch 1, Train Loss: 0.0238██████████████████████████████████████████████████████████████| 744/744 [09:18<00:00,  1.35it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:16<00:00,  1.34it/s]
# 2024-06-28 23:29:29,368 - INFO - Epoch 2, Train Loss: 0.0189██████████████████████████████████████████████████████████████| 744/744 [09:16<00:00,  1.35it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:14<00:00,  1.34it/s]
# 2024-06-28 23:38:43,838 - INFO - Epoch 3, Train Loss: 0.0170██████████████████████████████████████████████████████████████| 744/744 [09:14<00:00,  1.30it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:18<00:00,  1.33it/s]
# 2024-06-28 23:48:02,688 - INFO - Epoch 4, Train Loss: 0.0157██████████████████████████████████████████████████████████████| 744/744 [09:18<00:00,  1.49it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:19<00:00,  1.33it/s]
# 2024-06-28 23:57:21,745 - INFO - Epoch 5, Train Loss: 0.0146██████████████████████████████████████████████████████████████| 744/744 [09:18<00:00,  1.33it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:16<00:00,  1.34it/s]
# 2024-06-29 00:06:38,645 - INFO - Epoch 6, Train Loss: 0.0136██████████████████████████████████████████████████████████████| 744/744 [09:16<00:00,  1.43it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:18<00:00,  1.33it/s]
# 2024-06-29 00:15:57,558 - INFO - Epoch 7, Train Loss: 0.0128██████████████████████████████████████████████████████████████| 744/744 [09:18<00:00,  1.28it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:17<00:00,  1.33it/s]
# 2024-06-29 00:25:14,867 - INFO - Epoch 8, Train Loss: 0.0119██████████████████████████████████████████████████████████████| 744/744 [09:17<00:00,  1.38it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:19<00:00,  1.33it/s]
# 2024-06-29 00:34:34,615 - INFO - Epoch 9, Train Loss: 0.0112██████████████████████████████████████████████████████████████| 744/744 [09:19<00:00,  1.37it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:17<00:00,  1.33it/s]
# 2024-06-29 00:43:52,571 - INFO - Epoch 10, Train Loss: 0.0105█████████████████████████████████████████████████████████████| 744/744 [09:17<00:00,  1.34it/s]
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [1:32:58<00:00, 557.89s/it]
# 2024-06-29 00:43:52,572 - INFO - Saving model...
# 2024-06-29 00:43:54,313 - INFO - Training complete.



#weight_4
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:14<00:00,  1.34it/s]
# 2024-06-29 12:54:49,475 - INFO - Epoch 77, Train Loss: 0.0009█████████████████████████████████████████████████████████████| 744/744 [09:14<00:00,  1.43it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:13<00:00,  1.34it/s]
# 2024-06-29 13:04:03,452 - INFO - Epoch 78, Train Loss: 0.0009█████████████████████████████████████████████████████████████| 744/744 [09:13<00:00,  1.38it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:14<00:00,  1.34it/s]
# 2024-06-29 13:13:18,284 - INFO - Epoch 79, Train Loss: 0.0008█████████████████████████████████████████████████████████████| 744/744 [09:14<00:00,  1.45it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:15<00:00,  1.34it/s]
# 2024-06-29 13:22:33,831 - INFO - Epoch 80, Train Loss: 0.0008█████████████████████████████████████████████████████████████| 744/744 [09:15<00:00,  1.39it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:14<00:00,  1.34it/s]
# 2024-06-29 13:31:48,493 - INFO - Epoch 81, Train Loss: 0.0008█████████████████████████████████████████████████████████████| 744/744 [09:14<00:00,  1.43it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:13<00:00,  1.34it/s]
# 2024-06-29 13:41:02,051 - INFO - Epoch 82, Train Loss: 0.0008█████████████████████████████████████████████████████████████| 744/744 [09:13<00:00,  1.45it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:13<00:00,  1.34it/s]
# 2024-06-29 13:50:15,509 - INFO - Epoch 83, Train Loss: 0.0008█████████████████████████████████████████████████████████████| 744/744 [09:13<00:00,  1.39it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:13<00:00,  1.34it/s]
# 2024-06-29 13:59:29,368 - INFO - Epoch 84, Train Loss: 0.0008█████████████████████████████████████████████████████████████| 744/744 [09:13<00:00,  1.33it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:14<00:00,  1.34it/s]
# 2024-06-29 14:08:44,027 - INFO - Epoch 85, Train Loss: 0.0008█████████████████████████████████████████████████████████████| 744/744 [09:14<00:00,  1.39it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:15<00:00,  1.34it/s]
# 2024-06-29 14:17:59,662 - INFO - Epoch 86, Train Loss: 0.0007█████████████████████████████████████████████████████████████| 744/744 [09:15<00:00,  1.48it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:16<00:00,  1.34it/s]
# 2024-06-29 14:27:16,038 - INFO - Epoch 87, Train Loss: 0.0007█████████████████████████████████████████████████████████████| 744/744 [09:16<00:00,  1.32it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:16<00:00,  1.34it/s]
# 2024-06-29 14:36:32,655 - INFO - Epoch 88, Train Loss: 0.0007█████████████████████████████████████████████████████████████| 744/744 [09:16<00:00,  1.24it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:15<00:00,  1.34it/s]
# 2024-06-29 14:45:47,775 - INFO - Epoch 89, Train Loss: 0.0007█████████████████████████████████████████████████████████████| 744/744 [09:14<00:00,  1.45it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:17<00:00,  1.33it/s]
# 2024-06-29 14:55:05,383 - INFO - Epoch 90, Train Loss: 0.0007█████████████████████████████████████████████████████████████| 744/744 [09:17<00:00,  1.38it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:13<00:00,  1.34it/s]
# 2024-06-29 15:04:18,680 - INFO - Epoch 91, Train Loss: 0.0007█████████████████████████████████████████████████████████████| 744/744 [09:13<00:00,  1.35it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:15<00:00,  1.34it/s]
# 2024-06-29 15:13:34,394 - INFO - Epoch 92, Train Loss: 0.0007█████████████████████████████████████████████████████████████| 744/744 [09:15<00:00,  1.40it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:14<00:00,  1.34it/s]
# 2024-06-29 15:22:49,211 - INFO - Epoch 93, Train Loss: 0.0007█████████████████████████████████████████████████████████████| 744/744 [09:14<00:00,  1.44it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:16<00:00,  1.34it/s]
# 2024-06-29 15:32:05,670 - INFO - Epoch 94, Train Loss: 0.0007█████████████████████████████████████████████████████████████| 744/744 [09:16<00:00,  1.37it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:15<00:00,  1.34it/s]
# 2024-06-29 15:41:21,550 - INFO - Epoch 95, Train Loss: 0.0006█████████████████████████████████████████████████████████████| 744/744 [09:15<00:00,  1.38it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:14<00:00,  1.34it/s]
# 2024-06-29 15:50:36,541 - INFO - Epoch 96, Train Loss: 0.0006█████████████████████████████████████████████████████████████| 744/744 [09:14<00:00,  1.38it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:14<00:00,  1.34it/s]
# 2024-06-29 15:59:50,980 - INFO - Epoch 97, Train Loss: 0.0006█████████████████████████████████████████████████████████████| 744/744 [09:14<00:00,  1.35it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:17<00:00,  1.33it/s]
# 2024-06-29 16:09:08,343 - INFO - Epoch 98, Train Loss: 0.0006█████████████████████████████████████████████████████████████| 744/744 [09:17<00:00,  1.29it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:16<00:00,  1.34it/s]
# 2024-06-29 16:18:24,348 - INFO - Epoch 99, Train Loss: 0.0006█████████████████████████████████████████████████████████████| 744/744 [09:15<00:00,  1.38it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 744/744 [09:15<00:00,  1.34it/s]
# 2024-06-29 16:27:39,660 - INFO - Epoch 100, Train Loss: 0.0006████████████████████████████████████████████████████████████| 744/744 [09:15<00:00,  1.40it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [15:25:07<00:00, 555.07s/it]
# 2024-06-29 16:27:39,661 - INFO - Saving model...
# 2024-06-29 16:27:41,371 - INFO - Training complete.