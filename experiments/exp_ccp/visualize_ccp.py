"""CCP visualisation -- predefined subject selection (largest gain / median / largest loss)."""
import sys, csv, importlib.util
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'experiments' / 'exp_e12_eggo_m' / 'e130'))
from neuroscan_3d_v5 import UNet3D_v5
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset, REGIONS
import ccp_core as C
from run_ccp_oracle import run_ccp, CONFIGS, THRESH, WT

spec = importlib.util.spec_from_file_location("t", str(ROOT/'experiments/exp_e12_eggo_m/e130/train_e130_multimodal_baseline.py'))
t = importlib.util.module_from_spec(spec); _a=sys.argv; sys.argv=["e"]; spec.loader.exec_module(t); sys.argv=_a

rows = list(csv.DictReader(open(HERE/'CCP_per_subject.csv')))
d = np.array([float(r['dice_O4_ccp_R1'])-float(r['dice_baseline']) for r in rows])
order = np.argsort(d)
picks = {'largest_gain': order[-1], 'median': order[len(order)//2], 'largest_loss': order[0]}
print('selected (predefined criteria):', {k: rows[v]['subject_id'] for k,v in picks.items()})

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ck = torch.load(str(ROOT/'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'), map_location=dev, weights_only=False)
m = UNet3D_v5(4,3).to(dev).eval(); m.load_state_dict(ck['model_state'])
for p in m.parameters(): p.requires_grad_(False)
ds = BraTSMultimodalDataset(str(ROOT/'Dataset'/'Training'),'val',val_split=0.1,patch_size=t.PATCH)
sid2idx = {}
for i in range(len(ds)):
    sid2idx[Path(ds.subject_dirs[i]).name] = i

for label, ri in picks.items():
    sid = rows[ri]['subject_id']; i = sid2idx[sid]
    img, tgt, _ = ds[i]
    probs = t.sliding_window_predict(m, img.unsqueeze(0).to(dev), t.PATCH, t.SW_OVERLAP, 3, dev, True)
    E = torch.from_numpy(probs[WT]).to(torch.float64).to(dev)
    Y = tgt[WT].to(torch.float64).to(dev)
    al,be,ga = CONFIGS['R1']
    q4, stages, _ = run_ccp(E, al, be, ga, True, True)
    # slice with most GT
    z = int(Y.sum(dim=(1,2)).argmax().item())
    mri = img[2,z].numpy()  # t2f
    Eb=(E>=THRESH).float(); Qb=(q4>=THRESH).float()
    panels = [
        (mri,'MRI (t2f)','gray'), (E[z].cpu(),'baseline P(WT)','viridis'),
        (Eb[z].cpu(),'baseline binary','gray'), (q4[z].cpu(),'CCP q_final','viridis'),
        (Qb[z].cpu(),'CCP binary','gray'), ((q4-E)[z].cpu(),'q_CCP - E','bwr'),
        (Y[z].cpu(),'ground truth','gray'),
        ((Eb*(1-Y))[z].cpu(),'baseline FP','Reds'), ((Y*(1-Eb))[z].cpu(),'baseline FN','Blues'),
    ] + [(stages[k][z].cpu(), f'q^({k})','viridis') for k in range(5)]
    fig, ax = plt.subplots(2, 7, figsize=(26,8))
    for a_,(dat,ttl,cm) in zip(ax.ravel(), panels):
        arr=np.asarray(dat)
        if cm=='bwr':
            v=max(1e-9,np.abs(arr).max()); a_.imshow(arr,cmap=cm,vmin=-v,vmax=v)
        else: a_.imshow(arr,cmap=cm)
        a_.set_title(ttl,fontsize=9); a_.axis('off')
    for a_ in ax.ravel()[len(panels):]: a_.axis('off')
    db=float(rows[ri]['dice_baseline']); dcp=float(rows[ri]['dice_O4_ccp_R1'])
    fig.suptitle(f'{label}: {sid}  slice z={z}   WT Dice {db:.4f} -> {dcp:.4f} ({100*(dcp-db):+.3f}pp)  [R1]',fontsize=12)
    plt.tight_layout(); out=HERE/f'ccp_viz_{label}_{sid}.png'; plt.savefig(out,dpi=100); plt.close()
    print('wrote', out.name)
