# Speech Synthesis

## 实验步骤

### 1. 下载 FastSpeech2

```bash
git clone git@github.com:ming024/FastSpeech2.git
```

### 2. 下载模型

下载模型并按 README 中的提示解压到 `./output/ckpt` 目录下.

### 3. 安装依赖

```bash
conda create -n FastSpeech2 python=3.9
conda activate FastSpeech2
pip install -r requirements.txt
```

### 4. 运行

```bash
python synthesize.py --text "大家好" --speaker_id 0 --restore_step 600000 --mode single -p config/AISHELL3/preprocess.yaml -m config/AISHELL3/model.yaml -t config/AISHELL3/train.yaml
```

## 模型理解

### 1. 模型结构

![](./img/model.png)

模型结构如上图所示, 主要分为四个部分: Phoneme Embedding, Encoder, Variance Adaptor, Decoder.

### 2. FFTBlock

```python
class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout
        )

    def forward(self, enc_input, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn
```

FFTBlock 是一个基于 FFT 的卷积层, 由自注意力层 slf_attn 和前馈神经网络 pos_ffn 组成, 用于提取频谱特征. 它是 Encoder 和 Decoder 的基础结构. 与其他模型的不同之处在于作者这里使用了 1D 卷积, 而不是普通的全连接神经网络.

### 3. Encoder

Encoder 由 1 个 word embedding 层和多个 FFTBlock 组成. phoneme 首先经过 Encoder 得到 Encoder Output. 为了让 Encoder Output 的长度和 Mel-spectrogram 的长度一致, 需要对 Encoder Output 进行变长适配.

### 4. Variance Adaptor

Variance Adaptor 中包含三个 predictor, 分别是:

  * duration predictor: 预测每个 phoneme 的持续时间
  * pitch predictor: 预测每个 phoneme 的音高
  * energy predictor: 预测每个 phoneme 的能量

### 5. Decoder

Decoder 由多个 FFTBlock 组成, 作用是把经过 Variance Adaptor 的输入转换为 Mel-spectrogram.

## 实验结果

生成的音频如下:

  * [大家好](./result/AISHELL3/大家好.wav)
  * [越过长城，走向世界](./result/AISHELL3/越过长城，走向世界.wav)
  * [苟利国家生死以，岂因祸福避趋之](./result/AISHELL3/苟利国家生死以，岂因祸福避趋之.wav)
  * [Hello, world](./result/LJSpeech/Hello,%20world.wav)
  * [AI intro](./result/LJSpeech/AI%20intro.wav)
  * [I will be there for you](./result/LJSpeech/I%20will%20be%20there%20for%20you.wav)
  * [Beyond the barricade, is there a world you long to see?](./result/LJSpeech/Beyond%20the%20barricade,%20is%20there%20a%20world%20you%20long%20to%20see%3F.wav)