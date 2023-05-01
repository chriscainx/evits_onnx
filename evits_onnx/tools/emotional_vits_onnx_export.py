import emotional_vits_onnx_model
import torch
import numpy as np

net_g = emotional_vits_onnx_model.SynthesizerTrn(
    n_vocab=50,
    spec_channels=513,
    n_speakers=0,
    segment_size=32,
    filter_channels=768,
    hidden_channels=192,
    inter_channels=192,
    kernel_size=3,
    n_heads=2,
    n_layers=6,
    n_layers_q=3,
    p_dropout=0.1,
    resblock='1',
    resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]],
    resblock_kernel_sizes=[3,7,11],
    upsample_initial_channel=512,
    upsample_kernel_sizes=[16,16,4,4],
    upsample_rates=[8,8,2,2],
    use_spectral_norm=False)
_ = net_g.eval()
_ = emotional_vits_onnx_model.load_checkpoint("../G_32000.pth", net_g)

stn_tst = torch.LongTensor([0,20,0,21,0,22,0])
emo = np.load("../../../emo_vecs/select/e_1_0190.npy")
with torch.no_grad():
    x_tst = stn_tst.unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
    emo = emo
    o = net_g(x_tst, x_tst_lengths, emo=emo)