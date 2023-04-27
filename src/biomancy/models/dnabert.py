'''
DNABERT model (https://doi.org/10.1093/bioinformatics/btab083)
'''

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class BertEmbeddings(nn.Module):
    '''
    Construct the embeddings from word, position and token_type embeddings.
    '''
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int,
        hidden_dropout_prob: float,
        type_vocab_size: int,
        ) -> None:
        '''
        Args:
            vocab_size (int): Vocabulary size of the model. 
                Defines the different tokens that can be represented by the inputs_ids passed to the forward method
            hidden_size (int): Dimensionality of the encoder layers and the pooler layer
            max_position_embeddings (int): The maximum sequence length that this model might ever be used with
            hidden_dropout_prob (float): The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler
            type_vocab_size (int): The vocabulary size of the token_type_ids
        '''
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        ) -> torch.Tensor:
        '''
        Args:
            input_ids (torch.LongTensor): Tensor of shape (batch_size, seq_len)
                Indices of input sequence tokens in the vocabulary.
            token_type_ids (Optional[torch.LongTensor]): Tensor of shape (batch_size, seq_len)
                Segment token indices to indicate first and second portions of the inputs
            position_ids (Optional[torch.LongTensor]): Tensor of shape (batch_size, seq_len)
                Indices of positions of each input sequence tokens in the position embeddings

        Returns:
            Tensor: Tensor of shape (batch_size, seq_len, hidden_size)
        '''
        input_shape = input_ids.size()

        seq_length = input_shape[1]
        device = input_ids.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        output_attentions,
        ) -> None:
        '''
        Args:
            hidden_size (int): Dimensionality of the encoder layers and the pooler layer
            num_attention_heads (int):  Number of attention heads for each attention layer
            attention_probs_dropout_prob (float): The dropout ratio for the attention probabilities
            output_attentions (bool): Should the model returns attentions weights
        '''
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )
        self.output_attentions = output_attentions

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor]:
        '''
        Args:
            hidden_states (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask (Optional[torch.LongTensor]): Tensor of shape (batch_size, seq_len)
                Mask to avoid performing attention on padding token indices, 1 for tokens that are NOT MASKED, 0 for MASKED tokens

        Returns:
            context (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_size) 
            attentions (torch.FloatTensor, optional, returned when output_attentions=True):
                Tensor of shape (batch_size, num_heads, seq_len, seq_len)
                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads
        '''
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs
    
    def _transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)


class AttentionOutput(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_dropout_prob: float,
        ) -> None:
        '''
        Args:
            hidden_size (int): Dimensionality of the encoder layers and the pooler layer
            hidden_dropout_prob (float): The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler
        '''
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
        ) -> torch.Tensor:
        '''
        Args:
            hidden_states (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_size)
            input_tensor (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Tensor: Tensor of shape (batch_size, seq_len, hidden_size) 
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class AttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        hidden_dropout_prob: float,
        output_attentions: bool,
        ) -> None:
        '''
        Args:
            hidden_size (int): Dimensionality of the encoder layers and the pooler layer
            num_attention_heads (int):  Number of attention heads for each attention layer
            attention_probs_dropout_prob (float): The dropout ratio for the attention probabilities
            hidden_dropout_prob (float): The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler
            output_attentions (bool): Should the model returns attentions weights
        '''
        super().__init__()
        self.self = SelfAttention(hidden_size = hidden_size,
                                num_attention_heads = num_attention_heads,
                                attention_probs_dropout_prob = attention_probs_dropout_prob,
                                output_attentions=output_attentions)
        
        self.output = AttentionOutput(hidden_size, hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        ) -> Tuple[torch.Tensor]:
        '''
        Args:
            hidden_states (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask (Optional[torch.LongTensor]): Tensor of shape (batch_size, seq_len)
                Mask to avoid performing attention on padding token indices, 1 for tokens that are NOT MASKED, 0 for MASKED tokens

        Returns:
            context (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_size) 
            attentions (torch.FloatTensor, optional, returned when output_attentions=True):
                Tensor of shape (batch_size, num_heads, seq_len, seq_len)
                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads
        '''
        self_outputs = self.self(hidden_states, attention_mask)
        block_output = self.output(self_outputs[0], hidden_states)
        outputs = (block_output,) + self_outputs[1:] 
        return outputs


class Intermediate(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        ) -> None:
        '''
        Args:
            hidden_size (int): Dimensionality of the encoder layers and the pooler layer
            intermediate_size (int): Dimensionality of the intermediate layer
        '''
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        # gelu activation function is used 
        self.intermediate_act_fn = F.gelu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            hidden_states (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Tensor: Tensor of shape (batch_size, seq_len, intermediate_size) 
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ModelOutput(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_dropout_prob: float,
        ) -> None:
        '''
        Args:
            hidden_size (int): Dimensionality of the encoder layers and the pooler layer
            intermediate_size (int): Dimensionality of the intermediate layer
            hidden_dropout_prob (float): The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler
        '''
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
        ) -> torch.Tensor:
        '''
        Args:
            hidden_states (torch.Tensor): Tensor of shape (batch_size, seq_len, intermediate_size)
            input_tensor (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Tensor: Tensor of shape (batch_size, seq_len, hidden_size) 
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        hidden_dropout_prob: float,
        output_attentions: bool,
        intermediate_size: int,
        ) -> None:
        '''
        Args:
            hidden_size (int): Dimensionality of the encoder layers and the pooler layer
            num_attention_heads (int):  Number of attention heads for each attention layer
            attention_probs_dropout_prob (float): The dropout ratio for the attention probabilities
            hidden_dropout_prob (float): The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler
            output_attentions (bool): Should the model returns attentions weights
            intermediate_size (int): Dimensionality of the intermediate layer
        '''
        super().__init__()
        self.attention = AttentionBlock(hidden_size = hidden_size,
                                        num_attention_heads = num_attention_heads,
                                        attention_probs_dropout_prob = attention_probs_dropout_prob,
                                        hidden_dropout_prob = hidden_dropout_prob,
                                        output_attentions = output_attentions)
        
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = ModelOutput(hidden_size, intermediate_size, hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        )-> Tuple[torch.Tensor]:
        '''
        Args:
            hidden_states (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask (Optional[torch.LongTensor]): Tensor of shape (batch_size, seq_len)
                Mask to avoid performing attention on padding token indices, 1 for tokens that are NOT MASKED, 0 for MASKED tokens

        Returns:
            layer_output (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_size) 
                Sequence of hidden-states at the output of the last layer of the encoder
            attentions (torch.FloatTensor, optional, returned when output_attentions=True):
                Tensor of shape (batch_size, num_heads, seq_len, seq_len)
                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads
        '''
        self_attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = self_attention_outputs[0]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + self_attention_outputs[1:]
        return outputs


class BertEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        hidden_dropout_prob: float,
        output_attentions: bool,
        intermediate_size: int,
        num_hidden_layers: int,
        ) -> None:
        '''
        Args:
            hidden_size (int): Dimensionality of the encoder layers and the pooler layer
            num_attention_heads (int):  Number of attention heads for each attention layer
            attention_probs_dropout_prob (float): The dropout ratio for the attention probabilities
            hidden_dropout_prob (float): The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler
            output_attentions (bool): Should the model returns attentions weights
            intermediate_size (int): Dimensionality of the intermediate layer
            num_hidden_layers (int): Number of hidden layers
        '''
        super().__init__()
        self.output_attentions = output_attentions
        self.layer = nn.ModuleList([BertLayer(hidden_size = hidden_size,
                                            num_attention_heads = num_attention_heads,
                                            attention_probs_dropout_prob = attention_probs_dropout_prob,
                                            hidden_dropout_prob = hidden_dropout_prob,
                                            output_attentions = output_attentions,
                                            intermediate_size = intermediate_size)
                                    for _ in range(num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor]:
        '''
        Args:
            hidden_states (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask (Optional[torch.LongTensor]): Tensor of shape (batch_size, seq_len)
                Mask to avoid performing attention on padding token indices, 1 for tokens that are NOT MASKED, 0 for MASKED tokens

        Returns:
            layer_output (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_size) 
                Sequence of hidden-states at the output of the last layer of the encoder
            attentions (Tuple[torch.FloatTensor], optional, returned when output_attentions=True):
                Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, seq_len, seq_len).
                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
        '''
        all_attentions = ()
        for layer_module in self.layer:
            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        outputs = (hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs 


class BertPooler(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        '''
        Args:
            hidden_size (int): Dimensionality of the encoder layers and the pooler layer
        '''
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        '''
        pool the model by simply taking the hidden state corresponding to the first token

        Args:
            hidden_states (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_size)
                Sequence of hidden-states at the output of the last layer of the encoder
        
        Returns:
            Tensor of shape (batch_size, hidden_size)
        '''
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output  


class DNABERT(nn.Module):
    '''
        DNABERT Model
    '''
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int,
        hidden_dropout_prob: float,
        type_vocab_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        output_attentions: bool,
        intermediate_size: int,
        num_hidden_layers: int,
        num_labels: int,
        ) -> None:
        '''
        Args:
            vocab_size (int): Vocabulary size of the model. 
                Defines the different tokens that can be represented by the inputs_ids passed to the forward method
            hidden_size (int): Dimensionality of the encoder layers and the pooler layer
            max_position_embeddings (int): The maximum sequence length that this model might ever be used with
            hidden_dropout_prob (float): The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler
            type_vocab_size (int): The vocabulary size of the token_type_ids
            num_attention_heads (int):  Number of attention heads for each attention layer
            attention_probs_dropout_prob (float): The dropout ratio for the attention probabilities
            output_attentions (bool): Should the model returns attentions weights
            intermediate_size (int): Dimensionality of the intermediate layer
            num_hidden_layers (int): Number of hidden layers
            num_labels (int):  Number of classes
        '''
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size = vocab_size,
                                        hidden_size = hidden_size,
                                        max_position_embeddings = max_position_embeddings,
                                        hidden_dropout_prob = hidden_dropout_prob,
                                        type_vocab_size = type_vocab_size)
        self.encoder = BertEncoder(hidden_size = hidden_size,
                                num_attention_heads = num_attention_heads,
                                attention_probs_dropout_prob = attention_probs_dropout_prob,
                                hidden_dropout_prob = hidden_dropout_prob,
                                output_attentions = output_attentions,
                                intermediate_size = intermediate_size,
                                num_hidden_layers = num_hidden_layers)
        self.pooler = BertPooler(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.apply(self._init_weights)

    @property
    def dtype(self) -> torch.dtype:
        '''
        Returns:
            dtype: The dtype of the module (assuming that all the module parameters have the same dtype)
        '''
        return _get_parameter_dtype(self)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        ) -> Tuple[torch.FloatTensor]:
        '''
        Args:
            input_ids (torch.LongTensor): Tensor of shape (batch_size, seq_len)
                Indices of input sequence tokens in the vocabulary.
            attention_mask (Optional[torch.LongTensor]): Tensor of shape (batch_size, seq_len)
                Mask to avoid performing attention on padding token indices, 1 for tokens that are NOT MASKED, 0 for MASKED tokens
            token_type_ids (Optional[torch.LongTensor]): Tensor of shape (batch_size, seq_len)
                Segment token indices to indicate first and second portions of the inputs
            position_ids (Optional[torch.LongTensor]): Tensor of shape (batch_size, seq_len)
                Indices of positions of each input sequence tokens in the position embeddings

        Returns:
            logits (torch.FloatTensor): Tensor of shape (batch_size, num_labels)
                Classification scores (before SoftMax).
            attentions (Tuple[torch.FloatTensor], optional, returned when output_attentions=True):
                Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, seq_len, seq_len)
                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
        '''
        input_shape = input_ids.size()
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
    
        extended_attention_mask = self._get_extended_attention_mask(attention_mask, input_shape, device)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + encoder_outputs[1:]  
        return outputs
   
    def _get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
        device: torch.device,
        ) -> torch.Tensor:
        '''
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored
        Args:
            attention_mask (torch.Tensor): Mask with ones indicating tokens to attend to, zeros for tokens to ignore
            input_shape (Tuple[int]): The shape of the input to the model
            device: (torch.device): The device of the input to the model

        Returns:
            Tensor: The extended attention mask, with a the same dtype as attention_mask.dtype
        '''
        # We can provide a self-attention mask of dimentions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in additino to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for masked positions,
        # this operation will create a tensor which is 0.0 for positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is effectively the same as removing these entirely.
        exteneded_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        exteneded_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask      
    
    def _init_weights(self, module: nn.Module) -> None:
        ''' 
        Initialize the weights 
        '''
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


def _get_parameter_dtype(parameter: nn.Module) -> torch.dtype:
    try:
        return next(parameter.parameters()).dtype
    except StopIteration:
        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype