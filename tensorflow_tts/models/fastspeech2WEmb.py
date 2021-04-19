# -*- coding: utf-8 -*-
# Copyright 2020 The FastSpeech2 Authors and Minh Nguyen (@dathudeptrai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tensorflow Model modules for FastSpeech2."""

import tensorflow as tf
import numpy as np

from tensorflow.python.ops import math_ops

from tensorflow_tts.models.fastspeech import TFFastSpeech
from tensorflow_tts.models.fastspeech import get_initializer


class TFFastSpeechVariantPredictor(tf.keras.layers.Layer):
    """FastSpeech duration predictor module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.conv_layers = []
        for i in range(config.num_duration_conv_layers):
            self.conv_layers.append(
                tf.keras.layers.Conv1D(
                    config.f0_energy_predictor_filters,
                    config.f0_energy_predictor_kernel_sizes,
                    padding="same",
                    name="conv_._{}".format(i),
                )
            )
            self.conv_layers.append(tf.keras.layers.Activation(tf.nn.relu))
            self.conv_layers.append(
                tf.keras.layers.LayerNormalization(
                    epsilon=config.layer_norm_eps, name="LayerNorm_._{}".format(i)
                )
            )
            self.conv_layers.append(
                tf.keras.layers.Dropout(config.f0_energy_predictor_dropout_probs)
            )
        self.conv_layers_sequence = tf.keras.Sequential(self.conv_layers)
        self.output_layer = tf.keras.layers.Dense(1)

        if config.n_speakers > 1:
            self.decoder_speaker_embeddings = tf.keras.layers.Embedding(
                config.n_speakers,
                config.hidden_size,
                embeddings_initializer=get_initializer(config.initializer_range),
                name="speaker_embeddings",
            )
            self.speaker_fc = tf.keras.layers.Dense(
                units=config.hidden_size, name="speaker_fc"
            )

        self.config = config

    def call(self, inputs, training=False):
        """Call logic."""
        encoder_hidden_states, speaker_ids, attention_mask = inputs
        attention_mask = tf.cast(tf.expand_dims(attention_mask, 2), tf.float32)

        if self.config.n_speakers > 1:
            speaker_embeddings = self.decoder_speaker_embeddings(speaker_ids)
            speaker_features = tf.math.softplus(self.speaker_fc(speaker_embeddings))
            # extended speaker embeddings
            extended_speaker_features = speaker_features[:, tf.newaxis, :]
            encoder_hidden_states += extended_speaker_features

        # mask encoder hidden states
        masked_encoder_hidden_states = encoder_hidden_states * attention_mask

        # pass though first layer
        outputs = self.conv_layers_sequence(masked_encoder_hidden_states)
        outputs = self.output_layer(outputs)
        masked_outputs = outputs

        outputs = tf.squeeze(masked_outputs, -1)
        return outputs


class TFFastSpeech2WEmb(TFFastSpeech):
    """TF Fastspeech module."""

    def __init__(self, config, **kwargs):
        """Init layers for fastspeech."""
        super().__init__(config, **kwargs)
        self.f0_predictor = TFFastSpeechVariantPredictor(config, name="f0_predictor")
        self.energy_predictor = TFFastSpeechVariantPredictor(
            config, name="energy_predictor"
        )
        self.duration_predictor = TFFastSpeechVariantPredictor(
            config, name="duration_predictor"
        )

        self.getCorrectSize = tf.keras.layers.Dense(384, activation='relu', name='correctSize')

        # define f0_embeddings and energy_embeddings
        self.f0_embeddings = tf.keras.layers.Conv1D(
            filters=config.hidden_size,
            kernel_size=config.f0_kernel_size,
            padding="same",
            name="f0_embeddings",
        )
        self.f0_dropout = tf.keras.layers.Dropout(config.f0_dropout_rate)
        self.energy_embeddings = tf.keras.layers.Conv1D(
            filters=config.hidden_size,
            kernel_size=config.energy_kernel_size,
            padding="same",
            name="energy_embeddings",
        )
        self.energy_dropout = tf.keras.layers.Dropout(config.energy_dropout_rate)

    def _build(self):
        """Dummy input for building model."""
        # fake inputs
        input_ids = tf.convert_to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], tf.int32)
        attention_mask = tf.convert_to_tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], tf.int32
        )
        speaker_ids = tf.convert_to_tensor([0], tf.int32)
        duration_gts = tf.convert_to_tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], tf.int32)
        f0_gts = tf.convert_to_tensor(
            [[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]], tf.float32
        )
        energy_gts = tf.convert_to_tensor(
            [[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]], tf.float32
        )
        embs = tf.convert_to_tensor(
            [[[10]*768, [10]*768, [10]*768, [10]*768, [10]*768, [10]*768, [10]*768, [10]*768,[10]*768,[10]*768]*1], tf.float32
        )

        self(input_ids, attention_mask, speaker_ids, duration_gts, f0_gts, energy_gts, embs)

    def call(
        self,
        input_ids,
        attention_mask,
        speaker_ids,
        duration_gts,
        f0_gts,
        energy_gts,
        embs,
        training=False,
    ):
        """Call logic."""

        embedding_output = self.embeddings([input_ids, speaker_ids], training=training)
        #embs = tf.expand_dims(embs, axis=-1)
        #print(embs.shape)

        #first attempt
        #embedding_output = tf.concat([embedding_output, embs], -1)
        #embedding_output = self.getCorrectSize(embedding_output)

        embs = self.getCorrectSize(embs)
        embedding_output = tf.math.add(embedding_output, embs)


        encoder_output = self.encoder(
            [embedding_output, attention_mask], training=training
        )

        last_encoder_hidden_states = encoder_output[0]


        # energy predictor, here use last_encoder_hidden_states, u can use more hidden_states layers
        # rather than just use last_hidden_states of encoder for energy_predictor.
        duration_outputs = self.duration_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask]
        )  # [batch_size, length]

        f0_outputs = self.f0_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask], training=training
        )
        energy_outputs = self.energy_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask], training=training
        )

        f0_embedding = self.f0_embeddings(
            tf.expand_dims(f0_gts, 2)
        )  # [barch_size, mel_length, feature]
        energy_embedding = self.energy_embeddings(
            tf.expand_dims(energy_gts, 2)
        )  # [barch_size, mel_length, feature]

        # apply dropout both training/inference
        f0_embedding = self.f0_dropout(f0_embedding, training=True)
        energy_embedding = self.energy_dropout(energy_embedding, training=True)

        # sum features
        last_encoder_hidden_states += f0_embedding + energy_embedding

        length_regulator_outputs, encoder_masks = self.length_regulator(
            [last_encoder_hidden_states, duration_gts], training=training
        )

        # create decoder positional embedding
        decoder_pos = tf.range(
            1, tf.shape(length_regulator_outputs)[1] + 1, dtype=tf.int32
        )
        masked_decoder_pos = tf.expand_dims(decoder_pos, 0) * encoder_masks

        decoder_output = self.decoder(
            [length_regulator_outputs, speaker_ids, encoder_masks, masked_decoder_pos],
            training=training,
        )
        last_decoder_hidden_states = decoder_output[0]

        # here u can use sum or concat more than 1 hidden states layers from decoder.
        mel_before = self.mel_dense(last_decoder_hidden_states)
        mel_after = (
            self.postnet([mel_before, encoder_masks], training=training) + mel_before
        )

        outputs = (mel_before, mel_after, duration_outputs, f0_outputs, energy_outputs)
        return outputs



    def inference(
        self,
        input_ids,
        attention_mask,
        speaker_ids,
        speed_ratios,
        f0_ratios,
        energy_ratios,
        embs,
    ):
        """Call logic."""
        embedding_output = self.embeddings([input_ids, speaker_ids], training=False)
        #embs = tf.expand_dims(embs, axis=-1)

        #embedding_output = tf.concat([embedding_output, embs], -1)
        #embedding_output = self.getCorrectSize(embedding_output)
        embs = self.getCorrectSize(embs)
        embedding_output = tf.math.add(embedding_output, embs)


        encoder_output = self.encoder(
            [embedding_output, attention_mask], training=False
        )
        last_encoder_hidden_states = encoder_output[0]

        # expand ratios
        speed_ratios = tf.expand_dims(speed_ratios, 1)  # [B, 1]
        f0_ratios = tf.expand_dims(f0_ratios, 1)  # [B, 1]
        energy_ratios = tf.expand_dims(energy_ratios, 1)  # [B, 1]

        # energy predictor, here use last_encoder_hidden_states, u can use more hidden_states layers
        # rather than just use last_hidden_states of encoder for energy_predictor.
        duration_outputs = self.duration_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask]
        )  # [batch_size, length]
        duration_outputs = tf.math.exp(duration_outputs) - 1.0
        duration_outputs = tf.cast(
            tf.math.round(duration_outputs * speed_ratios), tf.int32
        )

        f0_outputs = self.f0_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask], training=False
        )
        f0_outputs *= f0_ratios

        energy_outputs = self.energy_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask], training=False
        )
        energy_outputs *= energy_ratios

        f0_embedding = self.f0_dropout(
            self.f0_embeddings(tf.expand_dims(f0_outputs, 2)), training=True
        )
        energy_embedding = self.energy_dropout(
            self.energy_embeddings(tf.expand_dims(energy_outputs, 2)), training=True
        )

        # sum features
        last_encoder_hidden_states += f0_embedding + energy_embedding

        length_regulator_outputs, encoder_masks = self.length_regulator(
            [last_encoder_hidden_states, duration_outputs], training=False
        )

        # create decoder positional embedding
        decoder_pos = tf.range(
            1, tf.shape(length_regulator_outputs)[1] + 1, dtype=tf.int32
        )
        masked_decoder_pos = tf.expand_dims(decoder_pos, 0) * encoder_masks

        decoder_output = self.decoder(
            [length_regulator_outputs, speaker_ids, encoder_masks, masked_decoder_pos],
            training=False,
        )
        last_decoder_hidden_states = decoder_output[0]

        # here u can use sum or concat more than 1 hidden states layers from decoder.
        mel_before = self.mel_dense(last_decoder_hidden_states)
        mel_after = (
            self.postnet([mel_before, encoder_masks], training=False) + mel_before
        )

        outputs = (mel_before, mel_after, duration_outputs, f0_outputs, energy_outputs)

        return outputs
