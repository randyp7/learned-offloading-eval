# net/nrv2x_lite.py
from dataclasses import dataclass
import math
import numpy as np

K_BOLTZ = 1.380649e-23

# ----- Config structs -----
@dataclass
class NRRadio:
    carrier_hz: float
    p_tx_w: float
    p_rx_w: float
    noise_fig_db: float
    bw_hz: float
    ant_gain_db: float
    

@dataclass
class NRParams:
    scs_khz: int                 # 15/30/60
    slot_symbols: int            # e.g., 14
    slot_duration_s: float       # e.g., 0.5 ms for 30 kHz
    prbs: int                    # number of PRBs allocated to this link
    overhead_data: float         # fraction for payload after PSSCH/PSCCH/DMRS
    harq_reps: int               # repetitions per TB (>=1)
    # Channel model / mapping
    scenario: str                # "UMi","UMa","RMa" (for pathloss slopes)
    los_model: str               # "UMi","UMa" (LOS probability function)
    shadow_sigma_db_los: float
    shadow_sigma_db_nlos: float
    cqi_thresh_db: list          # len 15 thresholds for CQI1..15

CQI_TABLE = [
    (-6, 0.15), (-4, 0.23), (-2, 0.38), (0, 0.60),
    (2, 0.90), (4, 1.35), (6, 1.91), (8, 2.73),
    (10, 3.88), (12, 5.12), (14, 6.02), (16, 6.72),
    (18, 7.40), (20, 7.80), (22, 7.98)  # bits/s/Hz (rough)
]

def _snr_to_eff(snr_db: float) -> tuple[int, float]:
    cqi = 1
    eff = CQI_TABLE[0][1]
    for i, (thr, e) in enumerate(CQI_TABLE, start=1):
        if snr_db >= thr:
            cqi, eff = i, e
    return min(cqi, 15), eff

def _noise_dbm(bw_hz: float, nf_db: float) -> float:
    kT_dbm_hz = -174.0  # dBm/Hz
    return kT_dbm_hz + 10*math.log10(bw_hz) + nf_db

def _umi_pl_db(d_m: float, f_hz: float) -> float:
    # simple 38.901 UMi NLOS-ish approximation
    d = max(d_m, 1.0)
    f_ghz = f_hz / 1e9
    return 36.7*math.log10(d) + 22.7 + 26*math.log10(f_ghz)

def one_nr_sidelink_hop(d_m, payload_bytes, tx: NRRadio, rx: NRRadio, nr=None, rng=None):
    bw = tx.bw_hz #float(tx.get("bw_hz", 20e6))
    pt_w = tx.p_tx_w #float(tx.get("p_tx_w", 0.2))
    gt = tx.ant_gain_db #float(tx.get("ant_gain_db", 0.0))
    gr = rx.ant_gain_db #float(rx.get("ant_gain_db", 0.0))
    nf = rx.noise_fig_db #float(rx.get("noise_figure_db", 7.0))
    p_rx_w = rx.p_rx_w #float(rx.get("p_rx_w", 0.6))
    f = tx.carrier_hz #float(tx.get("freq_hz", 5.9e9))

    pl = _umi_pl_db(d_m, f)             # path loss (dB)
    ptx_dbm = 10*math.log10(pt_w*1e3)   # W -> mW -> dBm
    noise_dbm = _noise_dbm(bw, nf)
    snr_db = ptx_dbm + gt + gr - pl - noise_dbm

    cqi, eff_bphz = _snr_to_eff(snr_db)
    rate_bps = eff_bphz * bw
    # floors/caps to avoid inf/nan
    rate_bps = max(rate_bps, 2e6)       # ≥ 2 Mbps
    tx_time_s = (payload_bytes * 8) / rate_bps
    tx_time_s = min(tx_time_s, 5.0)     # cap at 5 s per hop

    return {
        "rate_mbps": rate_bps / 1e6,
        "tx_time_s": tx_time_s,
        "snr_db": snr_db,
        "cqi": cqi,
        "e_tx_j": tx_time_s * pt_w,
        "e_rx_j": tx_time_s * p_rx_w,
    }



# CQI -> spectral efficiency (bits/s/Hz), from 3GPP TS 38.214 tables
CQI_TO_EFF = {
    1: 0.1523, 2: 0.2344, 3: 0.3770, 4: 0.6016, 5: 0.8770,
    6: 1.1758, 7: 1.4766, 8: 1.9141, 9: 2.4063, 10: 2.7305,
    11: 3.3223, 12: 3.9023, 13: 4.5234, 14: 5.1152, 15: 5.5547
}

# ----- Helpers -----
def _bw_hz_from_prb(scs_khz: int, prbs: int) -> float:
    # BW = PRB * 12 subcarriers * SCS (Hz)
    return prbs * 12 * (scs_khz * 1e3)

def _noise_floor_dbm(bw_hz: float, noise_fig_db: float) -> float:
    # N = -174 dBm/Hz + 10 log10(BW) + NF
    return -174.0 + 10.0 * math.log10(bw_hz) + noise_fig_db

def _los_prob(d_m: float, model: str) -> float:
    d = max(1.0, float(d_m))
    if model == "UMi":
        # 38.901 UMi street canyon LOS probability
        return min(18.0/d, 1.0) * (1.0 - math.exp(-d/36.0)) + math.exp(-d/36.0)
    elif model == "UMa":
        # 38.901 UMa
        if d <= 18: return 1.0
        return math.exp(-(d - 18.0)/63.0) * (1.0 + (d/36.0)) + (1.0 - math.exp(-(d - 18.0)/63.0)) * math.exp(-d/36.0)
    else:
        return 0.5  # fallback
def _pl_ci_db(d_m: float, fc_hz: float, n_los: float, n_nlos: float, los: bool) -> float:
    # 38.901 Close-In (CI) model: PL = FSPL(1m) + 10 n log10(d)
    fc_ghz = fc_hz / 1e9
    fspl1m = 32.4 + 20.0 * math.log10(fc_ghz)  # dB
    n = n_los if los else n_nlos
    return fspl1m + 10.0 * n * math.log10(max(1.0, d_m))

def _snr_db(tx_w: float, d_m: float, rx: NRRadio, n_los: float, n_nlos: float,
            p_los: float, sigma_los: float, sigma_nlos: float, bw_hz: float, rng: np.random.Generator) -> (float, bool):
    los = rng.random() < p_los
    pl = _pl_ci_db(d_m, rx.carrier_hz, n_los, n_nlos, los)
    sh_sigma = sigma_los if los else sigma_nlos
    sh = rng.normal(0.0, sh_sigma) if sh_sigma > 0 else 0.0
    tx_dbm = 10.0 * math.log10(max(1e-12, tx_w)) + 30.0
    pr_dbm = tx_dbm - pl + sh
    n_dbm  = _noise_floor_dbm(bw_hz, rx.noise_fig_db)
    return pr_dbm - n_dbm, los

def _cqi_from_snr(snr_db: float, thresh_db: list) -> int:
    # thresholds map CQI1..15; CQI=0 if below CQI1
    cqi = 0
    for i, th in enumerate(thresh_db, start=1):
        if snr_db >= th: cqi = i
    return cqi

# ----- Core: one NR sidelink hop -----
def one_nr_sidelink_hop_old(
    d_m: float,
    payload_bytes: int,
    tx: NRRadio, rx: NRRadio,
    nr: NRParams,
    rng: np.random.Generator
):
    # Effective bandwidth for the allocation
    bw_hz = _bw_hz_from_prb(nr.scs_khz, nr.prbs)
    # Scenario pathloss slopes (very typical CI exponents; tune per calibration)
    if nr.scenario == "UMi":
        n_los, n_nlos = 2.0, 2.9
    elif nr.scenario == "UMa":
        n_los, n_nlos = 2.2, 3.2
    else:  # RMa or fallback
        n_los, n_nlos = 2.1, 3.1

    p_los = _los_prob(d_m, nr.los_model)
    snr, los_flag = _snr_db(tx.p_tx_w, d_m, rx, n_los, n_nlos, p_los,
                            nr.shadow_sigma_db_los, nr.shadow_sigma_db_nlos, bw_hz, rng)

    cqi = _cqi_from_snr(snr, nr.cqi_thresh_db)
    eff = CQI_TO_EFF.get(cqi, 0.0)  # bits/s/Hz

    if eff <= 0.0:
        # Outage: return very large time/zero rate (still report energies ~ RX listening)
        return {
            "snr_db": snr, "los": los_flag, "cqi": cqi, "eff": eff,
            "slots": math.inf, "tx_time_s": math.inf, "rate_mbps": 0.0,
            "e_tx_j": math.inf, "e_rx_j": math.inf
        }

    # Bits per slot for our allocation
    bits_per_slot = eff * bw_hz * nr.slot_duration_s * nr.overhead_data
    need_bits = max(1, 8 * payload_bytes)
    slots = math.ceil(need_bits / bits_per_slot)
    # HARQ repetitions multiply airtime/energy (delivering one TB reliably)
    slots *= max(1, nr.harq_reps)

    tx_time = slots * nr.slot_duration_s
    rate_mbps = (need_bits / tx_time) / 1e6

    e_tx = tx.p_tx_w * tx_time
    e_rx = rx.p_rx_w * tx_time

    return {
        "snr_db": snr, "los": los_flag, "cqi": cqi, "eff": eff,
        "slots": slots, "tx_time_s": tx_time, "rate_mbps": rate_mbps,
        "e_tx_j": e_tx, "e_rx_j": e_rx
    }
