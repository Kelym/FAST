import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

from utils import try_cuda
from attn import MultiHeadAttention

from env import ConvolutionalImageFeatures, BottomUpImageFeatures


def make_image_attention_layers(args, image_features_list, hidden_size):
    image_attention_size = args.image_attention_size or hidden_size
    attention_mechs = []
    for featurizer in image_features_list:
        if isinstance(featurizer, ConvolutionalImageFeatures):
            if args.image_attention_type == 'feedforward':
                attention_mechs.append(MultiplicativeImageAttention(
                    hidden_size, image_attention_size,
                    image_feature_size=featurizer.feature_dim))
            elif args.image_attention_type == 'multiplicative':
                attention_mechs.append(FeedforwardImageAttention(
                    hidden_size, image_attention_size,
                    image_feature_size=featurizer.feature_dim))
        elif isinstance(featurizer, BottomUpImageFeatures):
            attention_mechs.append(BottomUpImageAttention(
                hidden_size,
                args.bottom_up_detection_embedding_size,
                args.bottom_up_detection_embedding_size,
                image_attention_size,
                featurizer.num_objects,
                featurizer.num_attributes,
                featurizer.feature_dim
            ))
        else:
            attention_mechs.append(None)
    attention_mechs = [
        try_cuda(mech) if mech else mech for mech in attention_mechs]
    return attention_mechs

# TODO: make all attention module return logit instead of weight

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=80):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the PE once
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
                    torch.arange(0, d_model, 2).float() / d_model * \
                            (-math.log(10000.0)))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self, x):
        x = x + Variable(self.pe[:,:x.size(1)], requires_grad=False)
        return self.dropout(x)


# TODO: try variational dropout (or zoneout?)
class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                 dropout_ratio, bidirectional=False, num_layers=1, glove=None):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.use_glove = glove is not None
        if self.use_glove:
            print('Using GloVe embedding')
            self.embedding.weight.data[...] = torch.from_numpy(glove)
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=(dropout_ratio if self.num_layers > 1 else 0),
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
                                         hidden_size * self.num_directions)

    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        return try_cuda(h0), try_cuda(c0)

    def forward(self, inputs, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        batch_size = inputs.size(0)
        embeds = self.embedding(inputs)   # (batch, seq_len, embedding_size)
        if not self.use_glove:
            embeds = self.drop(embeds)
        h0, c0 = self.init_state(batch_size)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1]  # (batch, hidden_size)

        decoder_init = nn.Tanh()(self.encoder2decoder(h_t))

        ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)
        ctx = self.drop(ctx)
        # (batch, seq_len, hidden_size*num_directions), (batch, hidden_size)
        return ctx, decoder_init, c_t

class SoftDotMultiHead(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, dim, num_head):
        '''Initialize layer.'''
        super(SoftDotMultiHead, self).__init__()
        self.multi = MultiHeadAttention(num_head, dim, dim, dim)

    def forward(self, h, k, v, mask=None):
        '''Propagate h through the network.

        h: batch x dim
        k,v: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        output, attn = self.multi(h.unsqueeze(1), k, v, mask.unsqueeze(1))
        return output.squeeze(1), attn.squeeze(1)

class SoftDotAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.dim = dim
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        # TODO: attn = attn / math.sqrt(self.dim) # prevent extreme softmax

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, h), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class ContextOnlySoftDotAttention(nn.Module):
    '''Like SoftDot, but don't concatenat h or perform the non-linearity transform
    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, dim, context_dim=None):
        '''Initialize layer.'''
        super(ContextOnlySoftDotAttention, self).__init__()
        if context_dim is None:
            context_dim = dim
        self.linear_in = nn.Linear(dim, context_dim, bias=False)
        self.sm = nn.Softmax(dim=1)

    def forward(self, h, context, mask=None):
        '''Propagate h through the network.
        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        return weighted_context, attn


class FeedforwardImageAttention(nn.Module):
    def __init__(self, context_size, hidden_size, image_feature_size=2048):
        super(FeedforwardImageAttention, self).__init__()
        self.feature_size = image_feature_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.fc1_feature = nn.Conv2d(
            image_feature_size, hidden_size, kernel_size=1, bias=False)
        self.fc1_context = nn.Linear(context_size, hidden_size, bias=True)
        self.fc2 = nn.Conv2d(hidden_size, 1, kernel_size=1, bias=True)

    def forward(self, feature, context):
        batch_size = feature.shape[0]
        feature_hidden = self.fc1_feature(feature)
        context_hidden = self.fc1_context(context)
        context_hidden = context_hidden.unsqueeze(-1).unsqueeze(-1)
        x = feature_hidden + context_hidden
        x = self.fc2(F.relu(x))
        # batch_size x (width * height) x 1
        attention = F.softmax(x.view(batch_size, -1), 1).unsqueeze(-1)
        # batch_size x feature_size x (width * height)
        reshaped_features = feature.view(batch_size, self.feature_size, -1)
        x = torch.bmm(reshaped_features, attention)  # batch_size x
        return x.squeeze(-1), attention.squeeze(-1)


class MultiplicativeImageAttention(nn.Module):
    def __init__(self, context_size, hidden_size, image_feature_size=2048):
        super(MultiplicativeImageAttention, self).__init__()
        self.feature_size = image_feature_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.fc1_feature = nn.Conv2d(
            image_feature_size, hidden_size, kernel_size=1, bias=True)
        self.fc1_context = nn.Linear(context_size, hidden_size, bias=True)
        self.fc2 = nn.Conv2d(hidden_size, 1, kernel_size=1, bias=True)

    def forward(self, feature, context):
        batch_size = feature.shape[0]
        # batch_size x hidden_size x width x height
        feature_hidden = self.fc1_feature(feature)
        # batch_size x hidden_size
        context_hidden = self.fc1_context(context)
        # batch_size x 1 x hidden_size
        context_hidden = context_hidden.unsqueeze(-2)
        # batch_size x hidden_size x (width * height)
        feature_hidden = feature_hidden.view(batch_size, self.hidden_size, -1)
        # batch_size x 1 x (width x height)
        x = torch.bmm(context_hidden, feature_hidden)
        # batch_size x (width * height) x 1
        attention = F.softmax(x.view(batch_size, -1), 1).unsqueeze(-1)
        # batch_size x feature_size x (width * height)
        reshaped_features = feature.view(batch_size, self.feature_size, -1)
        x = torch.bmm(reshaped_features, attention)  # batch_size x
        return x.squeeze(-1), attention.squeeze(-1)


class BottomUpImageAttention(nn.Module):
    def __init__(self, context_size, object_embedding_size,
                 attribute_embedding_size, hidden_size, num_objects,
                 num_attributes, image_feature_size=2048):
        super(BottomUpImageAttention, self).__init__()
        self.context_size = context_size
        self.object_embedding_size = object_embedding_size
        self.attribute_embedding_size = attribute_embedding_size
        self.hidden_size = hidden_size
        self.num_objects = num_objects
        self.num_attributes = num_attributes
        self.feature_size = (image_feature_size + object_embedding_size +
                             attribute_embedding_size + 1 + 5)

        self.object_embedding = nn.Embedding(
            num_objects, object_embedding_size)
        self.attribute_embedding = nn.Embedding(
            num_attributes, attribute_embedding_size)

        self.fc1_context = nn.Linear(context_size, hidden_size)
        self.fc1_feature = nn.Linear(self.feature_size, hidden_size)
        # self.fc1 = nn.Linear(context_size + self.feature_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, bottom_up_features, context):
        # image_features: batch_size x max_num_detections x feature_size
        # object_ids: batch_size x max_num_detections
        # attribute_ids: batch_size x max_num_detections
        # no_object_mask: batch_size x max_num_detections
        # context: batch_size x context_size

        # batch_size x max_num_detections x embedding_size
        attribute_embedding = self.attribute_embedding(
            bottom_up_features.attribute_indices)
        # batch_size x max_num_detections x embedding_size
        object_embedding = self.object_embedding(
            bottom_up_features.object_indices)
        # batch_size x max_num_detections x (feat size)
        feats = torch.cat((
            bottom_up_features.cls_prob.unsqueeze(2),
            bottom_up_features.image_features,
            attribute_embedding, object_embedding,
            bottom_up_features.spatial_features), dim=2)

        # attended_feats = feats.mean(dim=1)
        # attention = None

        # batch_size x 1 x hidden_size
        x_context = self.fc1_context(context).unsqueeze(1)
        # batch_size x max_num_detections x hidden_size
        x_feature = self.fc1_feature(feats)
        # batch_size x max_num_detections x hidden_size
        x = x_context * x_feature
        x = x / torch.norm(x, p=2, dim=2, keepdim=True)
        x = self.fc2(x).squeeze(-1)  # batch_size x max_num_detections
        x.data.masked_fill_(bottom_up_features.no_object_mask, -float("inf"))
        # batch_size x 1 x max_num_detections
        attention = F.softmax(x, 1).unsqueeze(1)
        # batch_size x feat_size
        attended_feats = torch.bmm(attention, feats).squeeze(1)
        return attended_feats, attention

class WhSoftDotAttentionCompact(nn.Module):
    ''' Visual Dot Attention Layer. '''

    def __init__(self, dim, context_dim):
        '''Initialize layer.'''
        super(WhSoftDotAttentionCompact, self).__init__()
        if dim != context_dim:
            dot_dim = min(dim, context_dim)
            self.linear_in = nn.Linear(dim, dot_dim//2, bias=True)
            self.linear_in_2 = nn.Linear(context_dim, dot_dim//2, bias=True)
        self.dim = dim
        self.context_dim = context_dim
        self.sm = nn.Softmax(dim=1)

    def forward(self, h, ctx, mask=None, v=None):
        if self.dim != self.context_dim:
            target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1
            context = self.linear_in_2(ctx)
        else:
            target = h.unsqueeze(2)
            context = ctx
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn_sm = self.sm(attn)
        attn3 = attn_sm.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len
        context = v if v is not None else ctx
        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        return weighted_context, attn

class WhSoftDotAttention(nn.Module):
    ''' Visual Dot Attention Layer. '''

    def __init__(self, h_dim, v_dim=None):
        '''Initialize layer.'''
        super(WhSoftDotAttention, self).__init__()
        if v_dim is None:
            v_dim = h_dim
        self.h_dim = h_dim
        self.v_dim = v_dim
        self.linear_in_h = nn.Linear(h_dim, v_dim, bias=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, h, k, mask=None, v=None):
        '''Propagate h through the network.
        h: batch x h_dim
        k: batch x v_num x v_dim
        '''
        target = self.linear_in_h(h).unsqueeze(2)  # batch x dot_dim x 1
        attn = torch.bmm(k, target).squeeze(2)  # batch x v_num
        #attn /= math.sqrt(self.v_dim) # scaled dot product attention
        if mask is not None:
            attn.data.masked_fill_(mask, -float('inf'))
        attn_sm = self.sm(attn)
        attn3 = attn_sm.view(attn.size(0), 1, attn.size(1))  # batch x 1 x v_num
        ctx = v if v is not None else k
        weighted_context = torch.bmm(
            attn3, ctx).squeeze(1)  # batch x v_dim
        return weighted_context, attn
class VisualSoftDotAttention(nn.Module):
    ''' Visual Dot Attention Layer. '''

    def __init__(self, h_dim, v_dim, dot_dim=256):
        '''Initialize layer.'''
        super(VisualSoftDotAttention, self).__init__()
        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
        self.linear_in_v = nn.Linear(v_dim, dot_dim, bias=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, h, k, mask=None, v=None):
        '''Propagate h through the network.

        h: batch x h_dim
        k: batch x v_num x v_dim
        '''
        target = self.linear_in_h(h).unsqueeze(2)  # batch x dot_dim x 1
        context = self.linear_in_v(k)  # batch x v_num x dot_dim
        attn = torch.bmm(context, target).squeeze(2)  # batch x v_num
        attn_sm = self.sm(attn)
        attn3 = attn_sm.view(attn.size(0), 1, attn.size(1))  # batch x 1 x v_num
        ctx = v if v is not None else k
        weighted_context = torch.bmm(
            attn3, ctx).squeeze(1)  # batch x v_dim
        return weighted_context, attn

class WhSoftDotAttention(nn.Module):
    ''' Visual Dot Attention Layer. '''

    def __init__(self, h_dim, v_dim=None):
        '''Initialize layer.'''
        super(WhSoftDotAttention, self).__init__()
        if v_dim is None:
            v_dim = h_dim
        self.h_dim = h_dim
        self.v_dim = v_dim
        self.linear_in_h = nn.Linear(h_dim, v_dim, bias=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, h, k, mask=None, v=None):
        '''Propagate h through the network.
        h: batch x h_dim
        k: batch x v_num x v_dim
        '''
        target = self.linear_in_h(h).unsqueeze(2)  # batch x dot_dim x 1
        attn = torch.bmm(k, target).squeeze(2)  # batch x v_num
        #attn /= math.sqrt(self.v_dim) # scaled dot product attention
        if mask is not None:
            attn.data.masked_fill_(mask, -float('inf'))
        attn_sm = self.sm(attn)
        attn3 = attn_sm.view(attn.size(0), 1, attn.size(1))  # batch x 1 x v_num
        ctx = v if v is not None else k
        weighted_context = torch.bmm(
            attn3, ctx).squeeze(1)  # batch x v_dim
        return weighted_context, attn



class EltwiseProdScoring(nn.Module):
    '''
    Linearly mapping h and v to the same dimension, and do a elementwise
    multiplication and a linear scoring
    '''

    def __init__(self, h_dim, a_dim, dot_dim=256):
        '''Initialize layer.'''
        super(EltwiseProdScoring, self).__init__()
        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
        self.linear_in_a = nn.Linear(a_dim, dot_dim, bias=True)
        self.linear_out = nn.Linear(dot_dim, 1, bias=True)

    def forward(self, h, all_u_t, mask=None):
        '''Propagate h through the network.

        h: batch x h_dim
        all_u_t: batch x a_num x a_dim
        '''
        target = self.linear_in_h(h).unsqueeze(1)  # batch x 1 x dot_dim
        context = self.linear_in_a(all_u_t)  # batch x a_num x dot_dim
        eltprod = torch.mul(target, context)  # batch x a_num x dot_dim
        logits = self.linear_out(eltprod).squeeze(2)  # batch x a_num
        return logits


class AttnDecoderLSTM(nn.Module):
    '''
    An unrolled LSTM with attention over instructions for decoding navigation
    actions.
    '''

    def __init__(self, embedding_size, hidden_size, dropout_ratio,
                 feature_size=2048+128, image_attention_layers=None, num_head=8):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.u_begin = try_cuda(Variable(
            torch.zeros(embedding_size), requires_grad=False))
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.visual_attention_layer = VisualSoftDotAttention(
            hidden_size, feature_size)
        #self.text_attention_layer = SoftDotMultihead(hidden_size, num_head)
        self.text_attention_layer = SoftDotAttention(hidden_size)
        self.decoder2action = EltwiseProdScoring(hidden_size, embedding_size)

    def forward(self, u_t_prev, all_u_t, visual_context, h_0, c_0, ctx,
                ctx_mask=None):
        ''' Takes a single step in the decoder LSTM (allowing sampling).

        u_t_prev: batch x embedding_size
        all_u_t: batch x a_num x embedding_size
        visual_context: batch x v_num x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        '''
        attn_vision, alpha_v = self.visual_attention_layer(h_0, visual_context)
        # (batch, embedding_size+feature_size)
        concat_input = torch.cat((u_t_prev, attn_vision), 1)
        drop = self.drop(concat_input)
        h_1, c_1 = self.lstm(drop, (h_0, c_0))
        h_1_drop = self.drop(h_1)
        #attn_text, alpha_t = self.text_attention_layer(h_1_drop, ctx, ctx, ctx_mask)
        attn_text, alpha_t = self.text_attention_layer(h_1_drop, ctx, ctx_mask)
        logit = self.decoder2action(attn_text, all_u_t)
        return h_1,c_1,attn_text,attn_vision,alpha_t,logit,alpha_v

###############################################################################
# coground models
###############################################################################

class CogroundDecoderLSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout_ratio,
                 feature_size=2048+128, image_attention_layers=None,
                 visual_hidden_size=1024, num_head=8):
        super(CogroundDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.u_begin = try_cuda(Variable(
            torch.zeros(embedding_size), requires_grad=False))
        self.drop = nn.Dropout(p=dropout_ratio)
        # For now the text attention output size is hidden_size
        self.lstm = nn.LSTMCell(2*embedding_size+hidden_size, hidden_size)
        self.text_attention_layer = WhSoftDotAttention(hidden_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, dropout=0)
        self.visual_attention_layer = WhSoftDotAttention(hidden_size,
                                        visual_hidden_size)
        self.visual_mlp = nn.Sequential(
                nn.BatchNorm1d(feature_size),
                nn.Linear(feature_size, visual_hidden_size),
                nn.BatchNorm1d(visual_hidden_size),
                nn.Dropout(dropout_ratio),
                nn.ReLU())
        self.action_attention_layer = WhSoftDotAttention(hidden_size+hidden_size, visual_hidden_size)
        #self.action_attention_layer = VisualSoftDotAttention(hidden_size+hidden_size, visual_hidden_size)
        self.sm = nn.Softmax(dim=1)

    def forward(self, u_t_prev, all_u_t, visual_context, h_0, c_0, ctx,
                ctx_mask=None):
        '''
        u_t_prev: batch x embedding_size
        all_u_t: batch x a_num x embedding_size
        visual_context: batch x v_num x feature_size => panoramic view, DEP
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        '''
        ctx_pos = self.positional_encoding(ctx)
        attn_text,_alpha_text = self.text_attention_layer(h_0,ctx_pos,v=ctx,mask=ctx_mask)
        alpha_text = self.sm(_alpha_text)

        batch_size, a_size, _ = all_u_t.size()
        g_v = all_u_t.view(-1, self.feature_size)
        g_v = self.visual_mlp(g_v).view(batch_size, a_size, -1)
        attn_vision, _alpha_vision = self.visual_attention_layer(h_0, g_v, v=all_u_t)
        alpha_vision = self.sm(_alpha_vision)

        concat_input = torch.cat((attn_text, attn_vision, u_t_prev), 1)
        #drop = self.drop(concat_input)
        drop = concat_input
        h_1, c_1 = self.lstm(drop, (h_0, c_0))

        #action_selector = self.drop(torch.cat((attn_text, h_1), 1))
        action_selector = torch.cat((attn_text, h_1), 1)
        _, alpha_action = self.action_attention_layer(action_selector,g_v)
        return h_1,c_1,attn_text,attn_vision,alpha_text,alpha_action,alpha_vision

class ProgressMonitor(nn.Module):
    def __init__(self, embedding_size, hidden_size, text_len=80):
        super(ProgressMonitor, self).__init__()
        self.linear_h = nn.Linear(hidden_size + embedding_size, hidden_size)
        self.linear_pm = nn.Linear(text_len + hidden_size, 1)
        self.text_len = text_len

    def forward(self, h_t_minus_1, c_t, v_ground, text_attn):
        h_input = torch.cat((h_t_minus_1, v_ground),1)
        h_t_pm = torch.sigmoid(self.linear_h(h_input)) * torch.tanh(c_t)
        batch_size, seq_len = text_attn.size()
        if seq_len < self.text_len:
            pads = try_cuda(torch.zeros(batch_size, self.text_len - seq_len))
            pm_input = torch.cat((text_attn, pads, h_t_pm),1)
        else:
            pm_input = torch.cat((text_attn, h_t_pm),1)
        pm_output = torch.tanh(self.linear_pm(pm_input).squeeze(-1))
        return pm_output

class DeviationMonitor(nn.Module):
    def __init__(self, embedding_size, hidden_size, text_len=80):
        super(DeviationMonitor, self).__init__()
        self.linear_h = nn.Linear(hidden_size + embedding_size, hidden_size)
        self.linear_pm = nn.Linear(text_len + hidden_size, 1)
        self.text_len = text_len
        input_size = 1 + 1 + hidden_size*2 + embedding_size
        self.mlp = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.Sigmoid(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64,1)
                )

    def forward(self,last_dev,prev_h_t,h_t,c_t,v_ground,t_ground,alpha_v,alpha_t,last_logit):
        input = torch.cat((last_dev.unsqueeze(1), prev_h_t, h_t, v_ground, last_logit.unsqueeze(1)), 1)
        h = self.mlp(input).squeeze(-1)
        return h

class BacktrackButton(nn.Module):
    def __init__(self):
        self.mlp = nn.Sequential(
                nn.BatchNorm1d(4),
                nn.Linear(4, 4),
                nn.Sigmoid(),
                nn.Linear(4, 2),
                nn.ReLU(),
                )

    def forward(self, ent_logit, max_logit, mean_logit, pm):
        return self.mlp(ent_logit, max_logit, mean_logit, pm)

class SimpleCandReranker(nn.Module):
    def __init__(self, input_dim):
        super(SimpleCandReranker, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        x = self.net(x).squeeze(-1)
        return x

###############################################################################
# speaker models
###############################################################################


class SpeakerEncoderLSTM(nn.Module):
    def __init__(self, action_embedding_size, world_embedding_size,
                 hidden_size, dropout_ratio, bidirectional=False):
        super(SpeakerEncoderLSTM, self).__init__()
        assert not bidirectional, 'Bidirectional is not implemented yet'

        self.action_embedding_size = action_embedding_size
        self.word_embedding_size = world_embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.visual_attention_layer = VisualSoftDotAttention(
            hidden_size, world_embedding_size)
        self.lstm = nn.LSTMCell(
            action_embedding_size + world_embedding_size, hidden_size)
        self.encoder2decoder = nn.Linear(hidden_size, hidden_size)

    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = Variable(
            torch.zeros(batch_size, self.hidden_size), requires_grad=False)
        c0 = Variable(
            torch.zeros(batch_size, self.hidden_size), requires_grad=False)
        return try_cuda(h0), try_cuda(c0)

    def _forward_one_step(self, h_0, c_0, action_embedding,
                          world_state_embedding):
        feature, _ = self.visual_attention_layer(h_0, world_state_embedding)
        concat_input = torch.cat((action_embedding, feature), 1)
        drop = self.drop(concat_input)
        h_1, c_1 = self.lstm(drop, (h_0, c_0))
        return h_1, c_1

    def forward(self, batched_action_embeddings, world_state_embeddings):
        ''' Expects action indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        assert isinstance(batched_action_embeddings, list)
        assert isinstance(world_state_embeddings, list)
        assert len(batched_action_embeddings) == len(world_state_embeddings)
        batch_size = world_state_embeddings[0].shape[0]

        h, c = self.init_state(batch_size)
        h_list = []
        for t, (action_embedding, world_state_embedding) in enumerate(
                zip(batched_action_embeddings, world_state_embeddings)):
            h, c = self._forward_one_step(
                h, c, action_embedding, world_state_embedding)
            h_list.append(h)

        decoder_init = nn.Tanh()(self.encoder2decoder(h))

        ### TODO: The returned decoder_init and c could be problematic
        ### because no padding and unpacking????

        ctx = torch.stack(h_list, dim=1)  # (batch, seq_len, hidden_size)
        ctx = self.drop(ctx)
        return ctx, decoder_init, c  # (batch, hidden_size)


class SpeakerDecoderLSTM(nn.Module):
    def __init__(self, vocab_size, vocab_embedding_size, hidden_size,
                 dropout_ratio, glove=None, use_input_att_feed=False):
        super(SpeakerDecoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.vocab_embedding_size = vocab_embedding_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, vocab_embedding_size)
        self.use_glove = glove is not None
        if self.use_glove:
            print('Using GloVe embedding')
            self.embedding.weight.data[...] = torch.from_numpy(glove)
            self.embedding.weight.requires_grad = False
        self.drop = nn.Dropout(p=dropout_ratio)
        self.use_input_att_feed = use_input_att_feed
        if self.use_input_att_feed:
            print('using input attention feed in SpeakerDecoderLSTM')
            self.lstm = nn.LSTMCell(
                vocab_embedding_size + hidden_size, hidden_size)
            self.attention_layer = ContextOnlySoftDotAttention(hidden_size)
            self.output_l1 = nn.Linear(hidden_size*2, hidden_size)
            self.tanh = nn.Tanh()
        else:
            self.lstm = nn.LSTMCell(vocab_embedding_size, hidden_size)
            self.attention_layer = SoftDotAttention(hidden_size)
        self.decoder2action = nn.Linear(hidden_size, vocab_size)

    def forward(self, previous_word, h_0, c_0, ctx, ctx_mask=None):
        ''' Takes a single step in the decoder LSTM (allowing sampling).

        action: batch x 1
        feature: batch x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        '''
        word_embeds = self.embedding(previous_word)
        word_embeds = word_embeds.squeeze(dim=1)  # (batch, embedding_size)
        if not self.use_glove:
            word_embeds_drop = self.drop(word_embeds)
        else:
            word_embeds_drop = word_embeds

        if self.use_input_att_feed:
            h_tilde, alpha = self.attention_layer(
                self.drop(h_0), ctx, ctx_mask)
            concat_input = torch.cat((word_embeds_drop, self.drop(h_tilde)), 1)
            h_1, c_1 = self.lstm(concat_input, (h_0, c_0))
            x = torch.cat((h_1, h_tilde), 1)
            x = self.drop(x)
            x = self.output_l1(x)
            x = self.tanh(x)
            logit = self.decoder2action(x)
        else:
            h_1, c_1 = self.lstm(word_embeds_drop, (h_0, c_0))
            h_1_drop = self.drop(h_1)
            h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)
            logit = self.decoder2action(h_tilde)
        return h_1, c_1, alpha, logit


###############################################################################
# transformer models
###############################################################################

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                 dropout_ratio, bidirectional=False, num_layers=1, glove=None):
        super(TransformerEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.position_encoding = PositionalEncoding(hidden_size, dropout_ratio)
        self.use_glove = glove is not None
        self.fc = nn.Linear(embedding_size, hidden_size)
        nn.init.xavier_normal_(self.fc.weight)
        if self.use_glove:
            print('Using GloVe embedding')
            self.embedding.weight.data[...] = torch.from_numpy(glove)
            self.embedding.weight.requires_grad = False

    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = torch.zeros(batch_size,
                         self.hidden_size*self.num_layers*self.num_directions,
                         requires_grad=False)
        c0 = torch.zeros(batch_size,
                         self.hidden_size*self.num_layers*self.num_directions,
                         requires_grad=False)
        return try_cuda(h0), try_cuda(c0)

    def forward(self, inputs, lengths):
        batch_size = inputs.size(0)
        embeds = self.fc(self.embedding(inputs))
        embeds = self.position_encoding(embeds)
        max_len = max(lengths)
        embeds = embeds[:,:max_len,:]
        return (embeds, *self.init_state(batch_size))

###############################################################################
# scorer models
###############################################################################

class EmbeddingMatching(nn.Module):
    '''
    Linearly mapping h and v to the same dimension, and do a elementwise
    multiplication and a linear scoring
    '''
    # TODO experiment network structure

    def __init__(self, h_dim, a_dim, dot_dim=256):
        '''Initialize layer.'''
        super(EmbeddingMatching, self).__init__()
        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
        self.linear_in_a = nn.Linear(a_dim, dot_dim, bias=True)
        self.linear_out = nn.Linear(dot_dim, 1, bias=True)

    def forward(self, h, all_u_t, mask=None):
        '''Propagate h through the network.

        h: batch x h_dim
        all_u_t: batch x a_num x a_dim
        '''
        target = self.linear_in_h(h).unsqueeze(1)  # batch x 1 x dot_dim
        context = self.linear_in_a(all_u_t)  # batch x a_num x dot_dim
        eltprod = torch.mul(target, context)  # batch x a_num x dot_dim
        logits = self.linear_out(eltprod).squeeze(2)  # batch x a_num
        return logits

class DotScorer(nn.Module):
    def __init__(self, traj_embedding_size, instr_embedding_size):
        super(DotScorer, self).__init__()
        self.traj_embedding_size = traj_embedding_size
        self.instr_embedding_size = instr_embedding_size
        self.matching = EltwiseProdScoring(traj_embedding_size, instr_embedding_size)

    def forward(self, instr_encoding, proposals):
        return self.matching(instr_encoding, proposals)

class PairScorer(nn.Module):
    # TODO concat the two embeddings and output a logit
    def __init__(self, traj_embedding_size, instr_embedding_size):
        super(PairScorer, self).__init__()
        self.traj_embedding_size = traj_embedding_size
        self.instr_embedding_size = instr_embedding_size
        # TODO

    def forward(self, instr_encoding, proposals):
        # TODO
        pass
