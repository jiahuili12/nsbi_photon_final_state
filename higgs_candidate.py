# 从 Delphes ROOT 直接重建 2b2γ 的 m_bb 与 m_gg (可用于快速 sanity check)
import uproot, awkward as ak, numpy as np, matplotlib.pyplot as plt, vector
plt.style.use("science.mplstyle")

# === 改成你的 ROOT 路径（signal / background 各一个）===
FILE_SIG = "/path/to/tag_1_pythia8_events_delphes.root"
FILE_BKG = "/path/to/tag_1_pythia8_events_delphes.root"

def load_bb_gg_masses(root_path):
    with uproot.open(root_path) as f:
        t = f["Delphes"]
        # 分支名按 Delphes 默认 + 默认卡片
        jets_pt   = t["Jet.PT"].arrays(entry_stop=None)
        jets_eta  = t["Jet.Eta"].arrays()
        jets_phi  = t["Jet.Phi"].arrays()
        jets_m    = t["Jet.Mass"].arrays()
        jets_btag = t["Jet.BTag"].arrays()  # 1 or 0

        pho_pt   = t["Photon.PT"].arrays()
        pho_eta  = t["Photon.Eta"].arrays()
        pho_phi  = t["Photon.Phi"].arrays()

    # 选择 b-tag jet 与 photon，并做基本切选（与你 03a_read_delphes.py 一致或更宽松）:
    jets = ak.zip({"pt":jets_pt, "eta":jets_eta, "phi":jets_phi, "m":jets_m, "b":jets_btag}, with_name="Momentum4D")
    phos = ak.zip({"pt":pho_pt, "eta":pho_eta, "phi":pho_phi, "m":ak.zeros_like(pho_pt)}, with_name="Momentum4D")

    # 基本 kinematic cuts（与 03a 脚本尽量一致）
    jets = jets[(jets.pt>30) & (abs(jets.eta)<2.4)]
    phos = phos[(phos.pt>30) & (abs(phos.eta)<2.4)]

    # 只保留 b-tag 的 jet
    bjets = jets[jets["b"]==1]

    # 事件级：至少 2 个 bjet 且至少 2 个 photon
    mask = (ak.num(bjets)>=2) & (ak.num(phos)>=2)
    bjets = bjets[mask]
    phos  = phos[mask]

    # 取 pT 排序后的前 2 个
    b2   = ak.firsts(ak.combinations(ak.sort(bjets, axis=1, key="pt"), 2, fields=["b0","b1"]))
    g2   = ak.firsts(ak.combinations(ak.sort(phos, axis=1, key="pt"), 2, fields=["g0","g1"]))

    # 计算不变质量
    m_bb = (b2.b0 + b2.b1).mass
    m_gg = (g2.g0 + g2.g1).mass

    # 可选：复用你在 03a 中的质量窗 cuts（若想和训练前处理保持一致）
    # keep = (abs(m_bb-125)<25) & (abs(m_gg-125)<3)
    # m_bb, m_gg = m_bb[keep], m_gg[keep]

    return ak.to_numpy(m_bb), ak.to_numpy(m_gg)

sig_mbb, sig_mgg = load_bb_gg_masses(FILE_SIG)
bkg_mbb, bkg_mgg = load_bb_gg_masses(FILE_BKG)

# 1D 直方图（各自）
fig, ax = plt.subplots(1,2, figsize=(10,4), dpi=140)
ax[0].hist(sig_mgg, bins=60, histtype="step", label="Signal", density=True)
ax[0].hist(bkg_mgg, bins=60, histtype="step", label="Background", density=True)
ax[0].axvline(125, ls="--", c="r", alpha=.5); ax[0].set_xlabel(r"$m_{\gamma\gamma}$ [GeV]"); ax[0].legend()

ax[1].hist(sig_mbb, bins=60, histtype="step", label="Signal", density=True)
ax[1].hist(bkg_mbb, bins=60, histtype="step", label="Background", density=True)
ax[1].axvline(125, ls="--", c="r", alpha=.5); ax[1].set_xlabel(r"$m_{bb}$ [GeV]"); ax[1].legend()
plt.tight_layout(); plt.show()

# 2D 分布（m_bb vs m_gg）
fig, axs = plt.subplots(1,2, figsize=(11,4.2), dpi=140, constrained_layout=True)
h1 = axs[0].hist2d(sig_mbb, sig_mgg, bins=[60,60])
axs[0].set_xlabel(r"$m_{bb}$ [GeV]"); axs[0].set_ylabel(r"$m_{\gamma\gamma}$ [GeV]"); axs[0].set_title("Signal")
axs[0].axvline(125, ls="--", c="r", alpha=.6); axs[0].axhline(125, ls="--", c="r", alpha=.6)
fig.colorbar(h1[3], ax=axs[0])

h2 = axs[1].hist2d(bkg_mbb, bkg_mgg, bins=[60,60])
axs[1].set_xlabel(r"$m_{bb}$ [GeV]"); axs[1].set_title("Background")
axs[1].axvline(125, ls="--", c="r", alpha=.6); axs[1].axhline(125, ls="--", c="r", alpha=.6)
fig.colorbar(h2[3], ax=axs[1])
plt.show()
