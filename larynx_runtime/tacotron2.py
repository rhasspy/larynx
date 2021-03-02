import logging
import typing

import numpy as np
import onnxruntime

from .constants import SettingsType, TextToSpeechModel, TextToSpeechModelConfig

_LOGGER = logging.getLogger("tacotron2")

# -----------------------------------------------------------------------------


class Tacotron2TextToSpeech(TextToSpeechModel):
    def __init__(self, config: TextToSpeechModelConfig):
        super(Tacotron2TextToSpeech, self).__init__(config)
        model_dir = config.model_path
        sess_options = config.session_options

        # Load models
        encoder_path = model_dir / "encoder.onnx"
        decoder_path = model_dir / "decoder_iter.onnx"
        postnet_path = model_dir / "postnet.onnx"

        _LOGGER.debug("Loading encoder from %s", encoder_path)
        self.encoder = onnxruntime.InferenceSession(
            str(encoder_path), sess_options=sess_options
        )

        _LOGGER.debug("Loading decoder from %s", decoder_path)
        self.decoder_iter = onnxruntime.InferenceSession(
            str(decoder_path), sess_options=sess_options
        )

        _LOGGER.debug("Loading postnet from %s", postnet_path)
        self.postnet = onnxruntime.InferenceSession(
            str(postnet_path), sess_options=sess_options
        )

        self.gate_threshold = 0.6
        self.max_decoder_steps = 1000

    # -------------------------------------------------------------------------

    def phonemes_to_mels(
        self, phoneme_ids: np.ndarray, settings: typing.Optional[SettingsType] = None
    ) -> np.ndarray:
        """Convert phoneme ids to mel spectrograms"""
        # Convert to tensors
        # TODO: Allow batches
        text = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        text_lengths = np.array([text.shape[1]], dtype=np.int64)

        # Infer mel spectrograms
        mel = infer(
            text,
            text_lengths,
            self.encoder,
            self.decoder_iter,
            self.postnet,
            gate_threshold=self.gate_threshold,
            max_decoder_steps=self.max_decoder_steps,
        )

        return mel


# -----------------------------------------------------------------------------


def infer(
    text,
    text_lengths,
    encoder,
    decoder_iter,
    postnet,
    gate_threshold=0.6,
    max_decoder_steps=1000,
):
    # Encoder
    memory, processed_memory, _ = encoder.run(
        None, {"sequences": text, "sequence_lengths": text_lengths}
    )

    # Decoder
    mel_lengths = np.zeros([memory.shape[0]], dtype=np.int32)
    not_finished = np.ones([memory.shape[0]], dtype=np.int32)
    mel_outputs, gate_outputs, alignments = (np.zeros(1), np.zeros(1), np.zeros(1))
    first_iter = True

    (
        decoder_input,
        attention_hidden,
        attention_cell,
        decoder_hidden,
        decoder_cell,
        attention_weights,
        attention_weights_cum,
        attention_context,
        memory,
        processed_memory,
        mask,
    ) = init_decoder_inputs(memory, processed_memory, text_lengths)

    while True:
        (
            mel_output,
            gate_output,
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
        ) = decoder_iter.run(
            None,
            {
                "decoder_input": decoder_input,
                "attention_hidden": attention_hidden,
                "attention_cell": attention_cell,
                "decoder_hidden": decoder_hidden,
                "decoder_cell": decoder_cell,
                "attention_weights": attention_weights,
                "attention_weights_cum": attention_weights_cum,
                "attention_context": attention_context,
                "memory": memory,
                "processed_memory": processed_memory,
                "mask": mask,
            },
        )

        if first_iter:
            mel_outputs = np.expand_dims(mel_output, 2)
            gate_outputs = np.expand_dims(gate_output, 2)
            alignments = np.expand_dims(attention_weights, 2)
            first_iter = False
        else:
            mel_outputs = np.concatenate(
                (mel_outputs, np.expand_dims(mel_output, 2)), 2
            )
            gate_outputs = np.concatenate(
                (gate_outputs, np.expand_dims(gate_output, 2)), 2
            )
            alignments = np.concatenate(
                (alignments, np.expand_dims(attention_weights, 2)), 2
            )

        dec = (
            np.less_equal(sigmoid(gate_output), gate_threshold)
            .astype(np.int32)
            .squeeze(1)
        )
        not_finished = not_finished * dec
        mel_lengths += not_finished

        if np.sum(not_finished) == 0:
            _LOGGER.debug("Stopping after %s decoder steps(s)", mel_outputs.shape[2])
            break

        if mel_outputs.shape[2] >= max_decoder_steps:
            _LOGGER.warning("Reached max decoder steps (%s)", max_decoder_steps)
            break

        decoder_input = mel_output

    # Postnet
    mel_outputs_postnet = postnet.run(None, {"mel_outputs": mel_outputs})[0]

    return mel_outputs_postnet


# -----------------------------------------------------------------------------


def get_mask_from_lengths(lengths):
    max_len = np.max(lengths)
    ids = np.arange(0, max_len, dtype=lengths.dtype)
    return ids > np.expand_dims(lengths, 1)


def init_decoder_inputs(memory, processed_memory, memory_lengths):

    dtype = memory.dtype
    bs = memory.shape[0]
    seq_len = memory.shape[1]
    attention_rnn_dim = 1024
    decoder_rnn_dim = 1024
    encoder_embedding_dim = 512
    n_mel_channels = 80

    attention_hidden = np.zeros((bs, attention_rnn_dim), dtype=dtype)
    attention_cell = np.zeros((bs, attention_rnn_dim), dtype=dtype)
    decoder_hidden = np.zeros((bs, decoder_rnn_dim), dtype=dtype)
    decoder_cell = np.zeros((bs, decoder_rnn_dim), dtype=dtype)
    attention_weights = np.zeros((bs, seq_len), dtype=dtype)
    attention_weights_cum = np.zeros((bs, seq_len), dtype=dtype)
    attention_context = np.zeros((bs, encoder_embedding_dim), dtype=dtype)
    mask = get_mask_from_lengths(memory_lengths)
    decoder_input = np.zeros((bs, n_mel_channels), dtype=dtype)

    return (
        decoder_input,
        attention_hidden,
        attention_cell,
        decoder_hidden,
        decoder_cell,
        attention_weights,
        attention_weights_cum,
        attention_context,
        memory,
        processed_memory,
        mask,
    )


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
