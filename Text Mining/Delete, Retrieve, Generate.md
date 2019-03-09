# Delete, Retrieve, Generate: A Simple Approach to Sentiment and Style Transfer

*Li J, Jia R, He H, et al. Delete, retrieve, generate: A simple approach to sentiment and style transfer[J]. arXiv preprint arXiv:1804.06437, 2018.*

## Code
这里我们使用的是pytorch版本的代码。

#### 初始参数
1. "training"

2. "data"
- src_vocab_size: 源语料库大小
- tgt_vocab_size: 目标语料库大小

3. "model"
- model_type("delete", "seq2seq", "delete_retrieve"): 决定了模型的类型。
- encoder("lstm"): 使用的encoder。在这个代码里面，只实现了lstm一种encoder。
- emp_dim: Embedding层的维度。
- src_hidden_dim: 用来建立LSTMEncoder。表示隐藏层的特征的数量。
- src_layers: LSTMEncoder的层数。
- bidirectional: LSTM是否双向。
- dropout: 控制LSTM的dropout。

### models.py
这个代码定义了文中使用的RNN模型。

#### class SeqModel(nn.Module): 

##### \_\_init\_\_
1. 先初始化一番，把各种参数保存到self.\*\*\*里面去。
2. 建立Embedding层：如果share_vocab参数为true，源和目标会使用同一个embedding层；否则，源和目标会各自建立embedding。
3. 使用LSTM建立**self.encoder**：(使用了*encoders.py*里面的*LSTMEncoder*)参数包括emb_dim, src_hidden_dim, src_layers, bidirectional, dropout。除了LSTMEncoder层之外，还建立了一个Linear层，叫做**self.ctx_bridge**，把src_hidden_dim大小的输入转成tgt_hidden_dim大小的输出。这个代码里面没有支持使用其他类型的encoder。
4. 根据model_type来决定处理attribute的方法：
	1. "delete"模型：使用Embedding构建**self.attribute_embedding**。attr_size=emb_dim，也就是Embedding层的dimensions。
	2. "delete_retrieve"模型：使用LSTMEncoder构建**self.attribute_encoder**。还是使用了emb_dim, src_hidden_dim, src_layers, bidirectional, dropout等参数。attr_size=src_hidden_dim。
	3. "seq2seq"：attr_size=0。
5. 建立**self.c_bridge**和**self.h_bridge**两个Linear层：它们的输入维度是attr_size + src_hidden_dim，输出维度是tgt_hidden_dim。
6. 使用StackedAttentionLSTM建立**self.decoder**: (使用了*decoders.py*里面的*StackedAttentionLSTM*)。
7. 建立**self.output_projection**层：Linear层，输入维度tgt_hidden_dim，输出维度tgt_vocab_size。
8. 建立Softmax层**self.softmax**。
9. 初始化模型权重。

##### forward(self, input_src, input_tgt, srcmask, srclens, input_attr, attrlens, attrmask)
1. 跑**self.src_embedding**层：输入input_src，输出src_emb。
2. 跑**self.encoder**层：输入src_emb, srclens, srcmask，输出src_outpus, (src_h_t, src_c_t)。
3. 根据模型是不是bidirectional的来计算h_t和c_t：如果是双向的, h_t就等于把src_h_t的最后一层的前向和后向传播的结果拼起来；如果是单向的，h_t就等于src_h_t最后一层的前向传播的结果。c_t亦然。
4. 把src_outputs通过**self.ctx_bridge**从src_hidden_dim的大小转成tgt_hidden_dim。
5. 根据模型的类型继续计算h_t和c_t:
	1. "delete"模型：跑**self.attribute_embedding**层，输入是input_attr，输出a_ht。然后，把c_t赋值为c_t和a_ht拼起来，把h_t赋值为h_t和a_ht拼起来。
	2. "delete_retrieve"模型：先跑**self.src_embedding**，输入input_attr，输出attr_emb。然后，跑**self.attribute_encoder**，输入attr_emb, attrlens, attrmask，输出_, (a_ht, a_ct)。之后，执行和第3步一样的操作，判断是不是双向的。如果是双向的，把a_ht和a_ct赋值为最后一层拼起来的结果；否则，赋值为最后一层的结果。最后，把c_t赋值为c_t和a_ht拼起来，把h_t赋值为h_t和a_ht拼起来。
	3. "seq2seq"模型：不计算。
6. 跑**self.c_bridge**和**self.h_bridge**：输入c_t, h_t得到c_t, h_t。就是把他们的维度从attr_size + src_hidden_sim变成tgt_hidden_dim。
7. 对input_tgt跑**tgt_embedding**层：得到tgt_emb。
8. 跑**self.decoder**: 输入tgt_emb, (h_t, c_t), src_outputs, srcmask，输出tgt_outputs, (\_, \_)。这里因为decoder层用的是StackedAttentionLSTM，所以输出的tgt_outputs就是一个完整的sequence而不是最后一个状态的值。
9. 跑**self.output_projection**：把tgt_outputs转成decoder_logit，就是从embedding层的表示变成vocabulary的表示。
10. 跑**self.softmax**得到probs，最后输出decoder_logit和probs。decoder_logit用于训练中计算Loss。