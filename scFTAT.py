import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

def create_projection_matrix(m, d, seed=0, scaling=0, struct_mode=False):
    nb_full_blocks = int(m / d)
    block_list = []
    current_seed = seed
    for _ in range(nb_full_blocks):
        if struct_mode:
            q = create_products_of_givens_rotations(d, seed)
        else:
            unstructured_block = tf.random.normal((d, d), seed=current_seed)
            q, _ = tf.linalg.qr(unstructured_block)
            q = tf.transpose(q)
            block_list.append(q)
            current_seed += 1
        remaining_rows = m - nb_full_blocks * d
        if remaining_rows > 0:
            if struct_mode:
                q = create_products_of_givens_rotations(d, seed)
            else:
                unstructured_block = tf.random.normal((d, d), seed=current_seed)
                q, _ = tf.linalg.qr(unstructured_block)
                q = tf.transpose(q)
            block_list.append(q[0:remaining_rows])
        final_matrix = tf.experimental.numpy.vstack(block_list)
        current_seed += 1

        if scaling == 0:
            multiplier = tf.norm(tf.random.normal((m, d), seed=current_seed), axis=1)
        elif scaling == 1:
            multiplier = tf.math.sqrt(float(d)) * tf.ones((m))
        else:
            raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

        return tf.linalg.matmul(tf.linalg.diag(multiplier), final_matrix)


def create_products_of_givens_rotations(dim, seed):

    nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
    q = np.eye(dim, dim)
    np.random.seed(seed)
    for _ in range(nb_givens_rotations):
        random_angle = math.pi * np.random.uniform()
        random_indices = np.random.choice(dim, 2)
        index_i = min(random_indices[0], random_indices[1])
        index_j = max(random_indices[0], random_indices[1])
        slice_i = q[index_i]
        slice_j = q[index_j]
        new_slice_i = math.cos(random_angle) * slice_i + math.sin(random_angle) * slice_j
        new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j
        q[index_i] = new_slice_i
        q[index_j] = new_slice_j
    return tf.cast(tf.constant(q), dtype=tf.float32)


def softmax_kernel_transformation(data,is_query,projection_matrix,numerical_stabilizer=0.000001):

    temp1 = tf.dtypes.cast(data.shape[-1], tf.float32)
    data_normalizer = 1.0 / (tf.math.sqrt(tf.math.sqrt(temp1)))
    data = data_normalizer * data
    temp2 = tf.dtypes.cast(projection_matrix.shape[0], tf.float32)
    ratio = 1.0 / tf.math.sqrt(temp2)
    data_dash = tf.einsum("blhd,md->blhm", data, projection_matrix)
    diag_data = tf.math.square(data)
    diag_data = tf.math.reduce_sum(diag_data, axis=tf.keras.backend.ndim(data) - 1)
    diag_data = diag_data / 2.0
    diag_data = tf.expand_dims(diag_data, axis=tf.keras.backend.ndim(data) - 1)
    last_dims_t = (len(data_dash.shape) - 1,)
    attention_dims_t = (len(data_dash.shape) - 3,)
    if is_query:
        data_dash = ratio * (tf.math.exp(data_dash - diag_data - tf.math.reduce_max(data_dash, axis=last_dims_t, keepdims=True)) + 
                             numerical_stabilizer)
    else:
        data_dash = ratio * (tf.math.exp(data_dash - diag_data - tf.math.reduce_max(data_dash, axis=last_dims_t + attention_dims_t, 
                                                                                    keepdims=True)) + numerical_stabilizer)

    return data_dash


def noncausal_numerator(qs, ks, vs):

    kvs = tf.einsum("lbhm,lbhd->bhmd", ks, vs)
    return tf.einsum("lbhm,bhmd->lbhd", qs, kvs)

def noncausal_denominator(qs, ks):

    all_ones = tf.ones([ks.shape[0]])
    ks_sum = tf.einsum("lbhm,l->bhm", ks, all_ones)
    return tf.einsum("lbhm,bhm->lbh", qs, ks_sum)


def favor_attention(query,key,value,kernel_transformation,projection_matrix):

    query_prime = kernel_transformation(query, True,projection_matrix)  
    key_prime = kernel_transformation(key, False, projection_matrix)  
    query_prime = tf.transpose(query_prime, [1, 0, 2, 3])  
    key_prime = tf.transpose(key_prime, [1, 0, 2, 3])  
    value = tf.transpose(value, [1, 0, 2, 3])  
    av_attention = noncausal_numerator(query_prime, key_prime, value)
    attention_normalizer = noncausal_denominator(query_prime, key_prime)

    av_attention = tf.transpose(av_attention, [1, 0, 2, 3])
    attention_normalizer = tf.transpose(attention_normalizer, [1, 0, 2])
    attention_normalizer = tf.expand_dims(attention_normalizer,len(attention_normalizer.shape))
    return av_attention / attention_normalizer

class FeedForwardNetwork(keras.Model):

    def __init__(self,dff_size,model_size, **kwargs):
        super(FeedForwardNetwork, self).__init__(**kwargs)
        self.dense1 = keras.layers.Dense(dff_size,activation='relu')
        self.dense2 = keras.layers.Dense(model_size)
    
    def call(self,x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class FFT_Mlp(layers.Layer):

    def __init__(self, dff_size,model_size, drop=0.01, **kwargs):
        super(FFT_Mlp, self).__init__(**kwargs)
        self.fc1 = layers.Dense(dff_size)
        self.act = tf.nn.gelu
        self.fc2 = layers.Dense(model_size)
        self.drop = layers.Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SpectralGatingNetwork(layers.Layer):

    def __init__(self, model_size,maxlen,num_heads, **kwargs):
        super().__init__(**kwargs)
        self.complex_weight = self.add_weight(shape=(num_heads,maxlen,model_size, 2), initializer=tf.initializers.TruncatedNormal(stddev=.02))         

    def call(self, x, mask):                

        x_complex = tf.complex(x, tf.zeros_like(x))        
        fft_x = tf.signal.fft(x_complex)
        complex_weight = tf.complex(self.complex_weight[:, :, :, 0], self.complex_weight[:, :, :, 1])
        weighted_fft_x = complex_weight * fft_x        
        ifft_x = tf.signal.ifft(weighted_fft_x)
        real_part = tf.math.real(ifft_x)
        real_part_float32 = tf.cast(real_part, tf.float32)

        return real_part_float32

class FFT_Block(layers.Layer):

    def __init__(self, model_size,num_heads, dff_size, maxlen, drop_path=0.1, **kwargs):
        super().__init__(**kwargs)
        self.filter = SpectralGatingNetwork(model_size,maxlen,32)
        self.drop_path1 = layers.Dropout(drop_path)
        self.drop_path2 = layers.Dropout(drop_path)
        self.layernorm1 = layers.LayerNormalization(epsilon=0.000001)
        self.layernorm2 = layers.LayerNormalization(epsilon=0.000001)
        self.mlp = FFT_Mlp(dff_size,model_size)

    def call(self, x, mask,training=True):
        # x = self.norm1(x)
        x = self.filter(x, mask)
        x = tf.cast(x, tf.float32)        
        x = x + self.drop_path1(x,training = training)
        x = x + self.layernorm1(x)
        x = self.mlp(x)
        x = x + self.drop_path2(x,training = training)
        x = x + self.layernorm2(x)
        return x

class FFTEncoder(keras.Model):

    def __init__(self,num_layers,model_size,num_heads,dff_size,vocab_size,maxlen,drop_path=0.1, **kwargs):
        super(FFTEncoder, self).__init__(**kwargs)
        self.model_size = model_size
        self.num_layers = num_layers
        self.embedding = keras.layers.Embedding(vocab_size,model_size)
        self.pos_embedding = positional_embedding(maxlen,model_size)
        self.FFTencoder_layers = [FFT_Block(model_size,num_heads,dff_size,maxlen,drop_path) for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(drop_path)
    
    def call(self,x,training=True,padding_mask=None):
        x = self.embedding(x)+self.pos_embedding
        x = self.dropout(x,training=training)
        for i in range(self.num_layers):
            x = self.FFTencoder_layers[i](x,padding_mask,training=True)
        return x


def positional_embedding(maxlen,model_size):

    PE = np.zeros((maxlen,model_size))
    for i in range(maxlen):
        for j in range(model_size):
            if j%2 == 0:
                PE[i,j] = np.sin(i/10000**(j/model_size))
            else:
                PE[i,j] = np.cos(i/10000**((j-1)/model_size))
    PE = tf.constant(PE,dtype=tf.float32)
    return PE


class MultiHeadAttention(keras.Model):
    def  __init__(self, model_size, num_heads, maxlen, kernel_transformation=softmax_kernel_transformation, nb_random_features = 50, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.model_size = model_size
        self.num_heads = num_heads
        self.head_size = model_size // num_heads
        self.maxlen = maxlen
        self.pos_embedding = positional_embedding(maxlen,model_size)
        self.kernel_transformation = kernel_transformation
        self.nb_random_features = nb_random_features

        self.WQ = keras.layers.Dense(model_size, name="dense_query")
        self.WK = keras.layers.Dense(model_size, name="dense_key")
        self.WV = keras.layers.Dense(model_size, name="dense_value")

        self.dense = keras.layers.Dense(model_size)


    def call(self, query, key, value, mask):

        batch_size = tf.shape(query)[0]

        query = self.WQ(query)
        key = self.WK(key)
        value = self.WV(value)

        q = query
        k = key
    
        q = tf.reshape(q, shape=[batch_size, self.maxlen, self.model_size])
        k = tf.reshape(k, shape=[batch_size, self.maxlen, self.model_size])


        q = tf.expand_dims(q, axis=2)
        q = tf.tile(q, [1, 1, self.num_heads, 1])
        k = tf.expand_dims(k, axis=2)
        k = tf.tile(k, [1, 1, self.num_heads, 1])

        sinusoidal_pos = self.pos_embedding
        ndim = tf.rank(sinusoidal_pos)
        sinusoidal_pos = tf.expand_dims(sinusoidal_pos, axis=0)  
        sinusoidal_pos = tf.tile(sinusoidal_pos, [batch_size, 1, 1])  
        sinusoidal_pos = tf.reshape(sinusoidal_pos, [batch_size, self.maxlen, self.model_size])



        cos_pos = tf.repeat(sinusoidal_pos[..., None, 1::2], repeats=2, axis=-1)
        sin_pos = tf.repeat(sinusoidal_pos[..., None, ::2], repeats=2, axis=-1)

        q2 = tf.stack([-q[..., 1::2], q[..., ::2]], axis=-1)
        q2 = tf.reshape(q2, tf.shape(q))
        q = q * cos_pos + q2 * sin_pos
        k2 = tf.stack([-k[..., 1::2], k[..., ::2]], axis=-1)
        k2 = tf.reshape(k2, tf.shape(k))
        k = k * cos_pos + k2 * sin_pos

        q = tf.reshape(q, shape = [batch_size, self.maxlen, self.num_heads, self.model_size])
        k = tf.reshape(k, shape = [batch_size, self.maxlen, self.num_heads, self.model_size])
        value = tf.reshape(value, shape=[batch_size, -1, self.num_heads, self.head_size])

        query = q
        key = k

        dim = query.shape[3]
        seed = tf.math.ceil(tf.math.abs(tf.math.reduce_sum(query) * 1e8))
        seed = tf.dtypes.cast(seed, tf.int32)
        projection_matrix = create_projection_matrix(self.nb_random_features, dim)
        context = favor_attention(query, key, value, self.kernel_transformation, projection_matrix)
        context = tf.reshape(context, (batch_size, -1, self.model_size))

        output = self.dense(context)

        return output
    

class MyAttn(layers.Layer):

    def __init__(self, model_size, act_ratio=0.25, act_fn=tf.nn.gelu, gate_fn=tf.nn.sigmoid, **kwargs):
        super(MyAttn, self).__init__(**kwargs)
        reduce_channels = int(model_size * act_ratio)
        self.norm = layers.LayerNormalization()
        self.global_reduce = layers.Dense(reduce_channels)
        self.local_reduce = layers.Dense(reduce_channels)
        self.act_fn = act_fn
        self.channel_select = layers.Dense(model_size)
        self.spatial_select = layers.Dense(1)
        self.gate_fn = gate_fn

    def call(self, x):
        ori_x = x
        x = self.norm(x)
        x_global = tf.reduce_mean(x, axis=1, keepdims=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)  
        s_attn = self.spatial_select(tf.concat([x_local, tf.tile(x_global, [1, x.shape[1], 1])], axis=-1))
        s_attn = self.gate_fn(s_attn)  

        attn = c_attn * s_attn  
        return ori_x * attn

class MyFNN(keras.Model):

    def __init__(self, dff_size, model_size, drop=0.1, **kwargs):
        super(MyFNN, self).__init__(**kwargs)
        self.fc1 = layers.Dense(dff_size)
        self.act = tf.nn.gelu
        self.fc2 = layers.Dense(dff_size)
        self.attn = MyAttn(dff_size)
        drop = 0.1
        self.drop = layers.Dropout(drop) if drop > 0 else layers.Activation('linear')
        self.fc3 = layers.Dense(model_size)

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.attn(x)
        x = self.drop(x)
        x = self.fc3(x)
        return x

class EncoderLayer(keras.layers.Layer):

    def __init__(self,model_size,num_heads,dff_size,maxlen,rate=0.1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(model_size,num_heads,maxlen)

        self.ffn = MyFNN(dff_size,model_size)

        self.resweight1 = tf.Variable(initial_value=tf.constant([0.000001],dtype=tf.float32), trainable=True)
        self.resweight2 = tf.Variable(initial_value=tf.constant([0.000001],dtype=tf.float32), trainable=True)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
    

    def call(self,x,FFT_out,mask,training=True):

        attn_output = self.attention(x,FFT_out,FFT_out,mask)
        attn_output = self.dropout1(attn_output,training=training)

        out1 = x + tf.multiply(attn_output,self.resweight1)
        ffn_output = self.ffn(out1)

        ffn_output = self.dropout2(ffn_output,training=training)

        out2 = out1 + tf.multiply(ffn_output,self.resweight2)
        return out2


class Encoder(keras.Model):

    def __init__(self,num_layers,model_size,num_heads,dff_size,vocab_size,maxlen,rate=0.1, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.model_size = model_size
        self.num_layers = num_layers
        self.embedding = keras.layers.Embedding(vocab_size,model_size)
        self.pos_embedding = positional_embedding(maxlen,model_size)
        self.encoder_layers = [EncoderLayer(model_size,num_heads,dff_size,maxlen,rate) for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(rate)
    
    def call(self,FFT_out,x,padding_mask,training=True):
        x = self.embedding(x)+self.pos_embedding
        x = self.dropout(x,training=training)
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x,FFT_out,padding_mask,training=True)
        return x

  
class DecoderLayer(keras.layers.Layer):

    def __init__(self,model_size,num_heads,dff_size,maxlen,rate=0.1, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.mask_attention = MultiHeadAttention(model_size,num_heads,maxlen)
        self.attention = MultiHeadAttention(model_size,num_heads,maxlen)
        self.ffn = MyFNN(dff_size,model_size)

        self.resweight1 = tf.Variable(initial_value=tf.constant([0.000001],dtype=tf.float32), trainable=True)
        self.resweight2 = tf.Variable(initial_value=tf.constant([0.000001],dtype=tf.float32), trainable=True)
        self.resweight3 = tf.Variable(initial_value=tf.constant([0.000001],dtype=tf.float32), trainable=True)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)
    
    def call(self,x,enc_output,look_ahead_mask,padding_mask,training=True):
        attn_decoder = self.mask_attention(x,x,x,look_ahead_mask)
        attn_decoder = self.dropout1(attn_decoder,training=training)

        out1 = x + tf.multiply(attn_decoder,self.resweight1)
        attn_encoder_decoder = self.attention(out1,enc_output,enc_output,padding_mask)
        attn_encoder_decoder = self.dropout2(attn_encoder_decoder,training=training)

        out2 = out1 + tf.multiply(attn_encoder_decoder,self.resweight2)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output,training=training)

        out3 = out2 + tf.multiply(ffn_output,self.resweight3)

        return out3
    
class Decoder(keras.Model):

    def __init__(self,num_layers,model_size,num_heads,dff_size,vocab_size,maxlen,rate=0.1, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.model_size = model_size
        self.num_layers = num_layers
        self.embedding = keras.layers.Embedding(vocab_size,model_size)
        self.pos_embedding = positional_embedding(maxlen,model_size)
        self.decoder_layers = [DecoderLayer(model_size,num_heads,dff_size,maxlen,rate) for _ in range(num_layers)]
        self.droput = keras.layers.Dropout(rate)
    
    def call(self,enc_output,x,look_ahead_mask,padding_mask,training=True):
        x = self.embedding(x)+self.pos_embedding
        x = self.droput(x,training=training)
        for i in range(self.num_layers):
            x = self.decoder_layers[i](x,enc_output,look_ahead_mask,padding_mask,training=True)
        return x


def padding_mask(seq):
    mask = tf.cast(tf.math.not_equal(seq,0),dtype=tf.float32)
    mask = mask[:,tf.newaxis,tf.newaxis,:]
    return mask


def look_ahead_mask(size):
    ahead_mask = tf.linalg.band_part(tf.ones((size,size)),-1,0)
    ahead_mask = tf.cast(ahead_mask,dtype=tf.float32)
    return ahead_mask


def create_mask(inp,tar):
    enc_padding_mask = padding_mask(inp)
    dec_padding_mask = padding_mask(tar)
    ahead_mask = look_ahead_mask(tf.shape(tar)[1])
    combined_mask = tf.minimum(dec_padding_mask,ahead_mask)
    return enc_padding_mask,dec_padding_mask,combined_mask


class Transformer(keras.Model):
    def __init__(self,num_layers,model_size,num_heads,dff_size,vocab_size,maxlen,**kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.FFTencoder = FFTEncoder(num_layers,model_size,num_heads,dff_size,vocab_size,maxlen)
        self.encoder = Encoder(num_layers,model_size,num_heads,dff_size,vocab_size,maxlen)
        self.decoder = Decoder(num_layers,model_size,num_heads,dff_size,vocab_size,maxlen)
        self.final_dense = tf.keras.layers.Dense(vocab_size,name='final_output')
    
    def call(self,all_inputs,training=True):
        sources1 = all_inputs
        sources2 = all_inputs
        targets = all_inputs
        enc_padding_mask,dec_padding_mask,combined_mask = create_mask(sources1,targets)
        FFT_out = self.FFTencoder(sources1,padding_mask=enc_padding_mask,training=training)
        enc_output = self.encoder(FFT_out,sources2,padding_mask=enc_padding_mask,training=training)
        dec_output = self.decoder(enc_output,targets,look_ahead_mask=combined_mask,padding_mask=dec_padding_mask,training=training)
        final_output = self.final_dense(dec_output)
        return final_output