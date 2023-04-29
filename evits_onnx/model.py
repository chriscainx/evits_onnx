import numpy as np
import onnxruntime as ort
from importlib.resources import files
from .txtdecoder import text_to_interspersed_sequence

class Synthesizer:
    """
    Synthesizer
    """

    def __init__(self, dec_path=None, dp_path=None, enc_p_path=None, flow_path=None):
        if dec_path is None:
            dec_path = str(files("evits_onnx").joinpath("dec.onnx"))
        if dp_path is None:
            dp_path = str(files("evits_onnx").joinpath("dp.onnx"))
        if enc_p_path is None:
            enc_p_path = str(files("evits_onnx").joinpath("enc_p.onnx"))
        if flow_path is None:
            flow_path = str(files("evits_onnx").joinpath("flow.onnx"))
        self.dec = ort.InferenceSession(dec_path)
        self.dp = ort.InferenceSession(dp_path)
        self.enc_p = ort.InferenceSession(enc_p_path)
        self.flow = ort.InferenceSession(flow_path)
        np.float = np.float32

    def _sequence_mask(self, length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = np.arange(max_length, dtype=length.dtype)
        return x[np.newaxis, :] < length[:, np.newaxis]

    def _generate_path(self, duration, mask):
        """
        duration: [b, 1, t_x]
        mask: [b, 1, t_y, t_x]
        """
        b, _, t_y, t_x = mask.shape
        cum_duration = np.cumsum(duration, axis=-1)

        cum_duration_flat = cum_duration.reshape(b * t_x)
        path = self._sequence_mask(cum_duration_flat, t_y)
        path = path.reshape(b, t_x, t_y)
        path = np.logical_xor(
            path, np.pad(path, ((0, 0), (1, 0), (0, 0)), "constant")[:, :-1]
        )
        path = np.expand_dims(path, 1).transpose(0, 1, 3, 2) * mask
        return path.astype(mask.dtype)

    def tts(self, txt, emo):
        np.float = np.float32
        noise_scale = 0.667
        length_scale = 1
        noise_scale_w = 0.8
        max_len = None
        x = np.array(text_to_interspersed_sequence(txt))[np.newaxis, :]
        x_lengths = np.array([x.shape[1]])

        x, m_p, logs_p, x_mask = self.enc_p.run(
            None, {"x": x, "x_lengths": x_lengths, "emotion": emo}
        )
        zinput = (
            np.random.randn(x.shape[0], 2, x.shape[2]).astype(np.float32)
            * noise_scale_w
        )

        logw = self.dp.run(None, {"x": x, "x_mask": x_mask, "zin": zinput})[
            0
        ]
        w = np.exp(logw) * x_mask * length_scale
        w_ceil = np.ceil(w)
        y_lengths = np.clip(np.sum(w_ceil, axis=(1, 2)), a_min=1, a_max=None).astype(
            np.int64
        )
        y_mask = self._sequence_mask(y_lengths, None)[:, np.newaxis].astype(
            x_mask.dtype
        )
        attn_mask = np.expand_dims(x_mask, axis=2) * np.expand_dims(y_mask, axis=-1)
        attn = self._generate_path(w_ceil, attn_mask)

        attn_squeezed = np.squeeze(attn, axis=1)  # [b, 1, t, t] -> [b, t, t]
        m_p_t = np.transpose(m_p, axes=(0, 2, 1))  # [b, t, d] -> [b, d, t]
        m_p = np.matmul(attn_squeezed, m_p_t).transpose(
            0, 2, 1
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p_t = np.transpose(logs_p, axes=(0, 2, 1))  # [b, t, d] -> [b, d, t]
        logs_p = np.matmul(attn_squeezed, logs_p_t).transpose(
            0, 2, 1
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        z_p = (
            m_p
            + np.random.randn(*m_p.shape).astype(np.float32)
            * np.exp(logs_p)
            * noise_scale
        )

        z = self.flow.run(None, {"z_p": z_p, "y_mask": y_mask})[0]
        z_in = (z * y_mask)[:, :, :max_len]

        o = self.dec.run(None, {"z_in": z_in})[0]
        return o[0, 0]
