import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import os
import logging
import sentencepiece as spm

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
PAD_IDX = 1

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

    def __init__(self, sp_ja, sp_en):
        self.sp_ja=sp_ja
        self.sp_en=sp_en


    def convert_text_to_indexes_src(self, text):
        ans = [self.sp_ja.PieceToId('<s>')] + self.sp_ja.EncodeAsIds(text.strip("\n")) + [self.sp_ja.PieceToId('</s>')]
        return torch.tensor(ans, dtype=torch.long)
        

    def convert_text_to_indexes_tgt(self, text):
        ans = [self.sp_en.PieceToId('<s>')] + self.sp_en.EncodeAsIds(text.strip("\n")) + [self.sp_en.PieceToId('</s>')]
        return torch.tensor(ans, dtype=torch.long)


    def create_dataloader(self, jp_list, en_list, collate_fn):
        jp_en_list = [[jp,en] for jp, en in zip(jp_list, en_list) if len(jp)<200 and len(en)<200] #系列長が250未満のものだけを訓練に使用する
        jp_list = [data[0] for data in jp_en_list]
        en_list = [data[1] for data in jp_en_list]
        src_data = [self.convert_text_to_indexes_src(jp) for jp in jp_list]
        tgt_data = [self.convert_text_to_indexes_tgt(en) for en in en_list]
        dataset = datasets(src_data, tgt_data)

        dataloader = DataLoader(dataset, batch_size=400, collate_fn=collate_fn, num_workers = 16, shuffle=True)

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
    
    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_pad_mask(self, input_word, pad_idx=1):
        pad_mask = input_word == pad_idx
        return pad_mask.float()
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        device = src.device
        src_pad_mask = self.create_pad_mask(src).to(device)
        tgt_pad_mask = self.create_pad_mask(tgt).to(device)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), device).to(device)
        
        src = self.embedding_src(src)
        tgt = self.embedding_tgt(tgt)
        src = self.pos_encoding(src)
        tgt = self.pos_encoding(tgt)
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        output = self.fc_out(output)
        return output

def train_epoch(model, dataloader, criterion, optimizer, device):
    epoch_loss = 0
    for batch_idx, (src, tgt) in enumerate(tqdm(dataloader)):
        src = src.to(device)
        tgt = tgt.to(device)

        mask = tgt != 3
        input_tgt = tgt[mask].view(tgt.size(0), -1) #3は<eos>
        targets = tgt[:, 1:]
        
        
        output = model(src, input_tgt)
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

    JP_TEST_FILE_PATH = "./kftt-data-1.0/data/orig/kyoto-test.ja"
    EN_TEST_FILE_PATH = "./kftt-data-1.0/data/orig/kyoto-test.en"

    logger.info("Preparing SentencePiece")
    # 学習の実行
    spm.SentencePieceTrainer.Train(
    '--pad_id=1 --unk_id=0 --bos_id=2 --eos_id=3 --input=./kftt-data-1.0/data/orig/kyoto-train.ja --model_prefix=sentencepiece_ja --vocab_size=32000 --character_coverage=0.9995'
    )
    
    sp_ja = spm.SentencePieceProcessor()
    sp_ja.Load("sentencepiece_ja.model")
    
     # 学習の実行
    spm.SentencePieceTrainer.Train(
    '--pad_id=1 --unk_id=0 --bos_id=2 --eos_id=3 --input=./kftt-data-1.0/data/orig/kyoto-train.en --model_prefix=sentencepiece_en --vocab_size=20000 --character_coverage=1.000'
    )
    
    sp_en = spm.SentencePieceProcessor()
    sp_en.Load("sentencepiece_en.model")
    

    with open(JP_TRAIN_FILE_PATH, "r", encoding="utf-8")as f:
        train_jp_list = f.readlines()
        train_jp_list = [jp.strip("\n") for jp in train_jp_list]

    with open(EN_TRAIN_FILE_PATH, "r", encoding="utf-8")as f:
        train_en_list = f.readlines()
        train_en_list = [en.strip("\n") for en in train_en_list]

    with open(JP_TEST_FILE_PATH, "r", encoding="utf-8")as f:
        test_jp_list = f.readlines()
        test_jp_list = [jp.strip("\n") for jp in test_jp_list]

    with open(EN_TEST_FILE_PATH, "r", encoding="utf-8")as f:
        test_en_list = f.readlines()
        test_en_list = [en.strip("\n") for en in test_en_list]

    logger.info("Creating Dataloader...")
    dataloader_creater = DataLoaderCreater(sp_ja, sp_en)
    train_dataloader = dataloader_creater.create_dataloader(train_jp_list, train_en_list, collate_fn=collate_fn) #create_dataloaderを実行することで語彙が作成される


    # モデルのハイパーパラメータ
    embedding_dim = 256
    num_heads = 4
    num_layers = 4
    lr_rate = 5e-4
    vocab_size_src = sp_ja.GetPieceSize()
    vocab_size_tgt = sp_en.GetPieceSize()
    model = TransformerModel(vocab_size_src, vocab_size_tgt, embedding_dim, num_heads, num_layers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)

    logger.info("Starting training loop...")
    num_epochs = 10

    model.train()
    for epoch in tqdm(range(num_epochs)):
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, device)
        logger.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')

    logger.info("Saving model...")
    torch.save(model.module.state_dict(),'model_weight_sentencepiece.pth') #nn.Dataparallelを使用した場合はmodel.state_dictではなくmodel.module.state_dictと書かなければいけない
    logger.info("Training complete.")
    

'''
trainer_interface.cc(687) LOG(INFO) Saving model: sentencepiece_en.model
trainer_interface.cc(699) LOG(INFO) Saving vocabs: sentencepiece_en.vocab
2024-07-08 00:32:09,024 - INFO - Creating Dataloader...
Let's use 4 GPUs!
2024-07-08 00:32:25,346 - INFO - Starting training loop...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 557/557 [39:00<00:00,  4.20s/it]
2024-07-08 01:11:25,401 - INFO - Epoch 1, Train Loss: 0.0123█████████████████████████████████████████████████| 557/557 [38:59<00:00,  3.42s/it]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 557/557 [38:07<00:00,  4.11s/it]
2024-07-08 01:49:33,214 - INFO - Epoch 2, Train Loss: 0.0103█████████████████████████████████████████████████| 557/557 [38:07<00:00,  3.35s/it]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 557/557 [38:19<00:00,  4.13s/it]
2024-07-08 02:27:53,060 - INFO - Epoch 3, Train Loss: 0.0095█████████████████████████████████████████████████| 557/557 [38:19<00:00,  4.17s/it]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 557/557 [38:13<00:00,  4.12s/it]
2024-07-08 03:06:06,138 - INFO - Epoch 4, Train Loss: 0.0090█████████████████████████████████████████████████| 557/557 [38:12<00:00,  3.50s/it]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 557/557 [38:12<00:00,  4.11s/it]
2024-07-08 03:44:18,173 - INFO - Epoch 5, Train Loss: 0.0085█████████████████████████████████████████████████| 557/557 [38:11<00:00,  3.54s/it]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 557/557 [38:21<00:00,  4.13s/it]
2024-07-08 04:22:39,995 - INFO - Epoch 6, Train Loss: 0.0082█████████████████████████████████████████████████| 557/557 [38:21<00:00,  4.98s/it]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 557/557 [38:09<00:00,  4.11s/it]
2024-07-08 05:00:49,903 - INFO - Epoch 7, Train Loss: 0.0078█████████████████████████████████████████████████| 557/557 [38:09<00:00,  3.72s/it]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 557/557 [38:19<00:00,  4.13s/it]
2024-07-08 05:39:09,604 - INFO - Epoch 8, Train Loss: 0.0075█████████████████████████████████████████████████| 557/557 [38:19<00:00,  3.86s/it]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 557/557 [38:15<00:00,  4.12s/it]
2024-07-08 06:17:25,440 - INFO - Epoch 9, Train Loss: 0.0073█████████████████████████████████████████████████| 557/557 [38:15<00:00,  3.59s/it]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 557/557 [38:10<00:00,  4.11s/it]
2024-07-08 06:55:36,318 - INFO - Epoch 10, Train Loss: 0.0070████████████████████████████████████████████████| 557/557 [38:10<00:00,  3.78s/it]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [6:23:10<00:00, 2299.10s/it]
2024-07-08 06:55:36,319 - INFO - Saving model...
2024-07-08 06:55:36,847 - INFO - Training complete.
'''