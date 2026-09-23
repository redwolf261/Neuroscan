"""
Verification script for E42's Lanczos implementation, run BEFORE trusting
any result computed with it on the real model. Two checks:
  1. Lanczos-recovered eigenvalue spectrum vs. direct numpy eigendecomposition
     on a small synthetic symmetric matrix (via a matvec-only interface,
     mimicking exactly how the real HVP-based Lanczos in lanczos_hvp.py
     is used -- never given direct matrix access).
  2. HVP vs. finite-difference directional derivative on the REAL model,
     restricted to a small parameter subset that genuinely participates in
     the loss graph (verified separately not to be an unused/zero-gradient
     parameter before using it as the test case).
"""
import sys
import numpy as np
import torch

sys.path.insert(0, ".")
from lanczos_hvp import lanczos_tridiagonalize, eigs_from_tridiagonal, flatten, unflatten_like


def verify_lanczos_on_synthetic_matrix():
    print("=== Verification 1: Lanczos vs direct eigendecomposition, synthetic matrix ===")
    np.random.seed(0)
    d = 200
    A = np.random.randn(d, d)
    A = (A + A.T) / 2  # symmetric
    true_eigs = np.sort(np.linalg.eigvalsh(A))[::-1]

    A_t = torch.tensor(A, dtype=torch.float32)

    class FakeParam:
        """Wraps a flat vector as a single 'parameter' so the real
        lanczos_tridiagonalize/hvp_fn interface can be used unchanged."""
        def __init__(self, t):
            self.data = t
            self.shape = t.shape

        def numel(self):
            return self.data.numel()

    fake_param = torch.zeros(d)

    def loss_fn():
        # loss whose Hessian IS A: L(x) = 0.5 * x^T A x, evaluated at the
        # CURRENT value of fake_param (a leaf tensor, requires_grad tracked
        # externally by the caller re-wrapping it each call -- see hvp_fn below)
        x = fake_param
        return 0.5 * torch.dot(x, A_t @ x)

    class DummyHVP:
        def __call__(self, loss_fn_unused, v_list):
            v = v_list[0]
            return [A_t @ v], 0.0

    hvp_fn = DummyHVP()
    params = [fake_param]

    alphas, betas = lanczos_tridiagonalize(hvp_fn, loss_fn, params, n_iter=60, seed=0, device="cpu")
    lanczos_eigs = eigs_from_tridiagonal(alphas, betas)

    print(f"  True top-5 eigenvalues:    {true_eigs[:5].round(4)}")
    print(f"  Lanczos top-5 eigenvalues: {lanczos_eigs[:5].round(4)}")
    print(f"  True bottom-5 eigenvalues:    {true_eigs[-5:].round(4)}")
    print(f"  Lanczos bottom-5 eigenvalues: {lanczos_eigs[-5:].round(4)}")

    top_match = np.allclose(true_eigs[:5], lanczos_eigs[:5], atol=1e-2)
    print(f"  Top-5 eigenvalues match within 1e-2: {top_match}")
    return top_match


def verify_hvp_on_real_model():
    print("\n=== Verification 2: HVP vs finite-difference, real model ===")
    project_root = "../../.."
    sys.path.insert(0, project_root)
    from neuroscan_3d_v3 import UNet3D_v3
    from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss
    from Dataset.brats_dataset import BraTSDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load("../e39/runs/lambda_0.25/checkpoints/epoch_30.pth", map_location=device, weights_only=False)
    model = UNet3D_v3(1, 1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_dataset = BraTSDataset(root_dir=f"{project_root}/Dataset/Training", split="val",
                                val_split=0.1, target_shape=(64, 64, 64), normalize=True)
    imgs, masks = [], []
    for i in range(4):
        img, mask, _ = val_dataset[i]
        imgs.append(img)
        masks.append(mask)
    images = torch.stack(imgs).to(device)
    masks_t = torch.stack(masks).to(device)

    focal_fn = FocalTverskyLoss()
    evidential_fn = EvidentialBetaLoss(weight=0.5)
    boundary_criterion = torch.nn.BCEWithLogitsLoss()
    MU = 0.1
    params = [p for p in model.parameters() if p.requires_grad]
    names = [n for n, _ in model.named_parameters()]

    def loss_main():
        out = model(images)
        probs, alpha, beta = out["probs"], out["alpha"], out["beta"]
        boundary_logit = out["boundary_logit"]
        seg = 0.5 * focal_fn(probs, masks_t) + 0.5 * evidential_fn(alpha, beta, masks_t)
        bnd = boundary_criterion(boundary_logit, masks_t)
        return seg + MU * bnd

    def hvp(loss_fn, v_list):
        loss = loss_fn()
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
        gv = sum(torch.sum(g * v) for g, v in zip(grads, v_list))
        hvp_list = torch.autograd.grad(gv, params, retain_graph=False, allow_unused=True)
        hvp_list = [h if h is not None else torch.zeros_like(p) for h, p in zip(hvp_list, params)]
        return hvp_list

    target_idx = names.index("boundary_head.bias")  # verified earlier to genuinely participate in L_0's graph
    target_param = params[target_idx]

    torch.manual_seed(0)
    v_list = [torch.zeros_like(p) for p in params]
    v_list[target_idx] = torch.randn_like(target_param)
    v_list[target_idx] = v_list[target_idx] / v_list[target_idx].norm()

    hv_analytic = hvp(loss_main, v_list)
    hv_analytic_target = hv_analytic[target_idx].detach().clone()

    eps = 1e-3
    with torch.no_grad():
        target_param.add_(eps * v_list[target_idx])
    grad_plus = torch.autograd.grad(loss_main(), [target_param])[0].detach().clone()

    with torch.no_grad():
        target_param.add_(-2 * eps * v_list[target_idx])
    grad_minus = torch.autograd.grad(loss_main(), [target_param])[0].detach().clone()

    with torch.no_grad():
        target_param.add_(eps * v_list[target_idx])

    fd_hvp = (grad_plus - grad_minus) / (2 * eps)
    rel_err = (fd_hvp - hv_analytic_target).norm().item() / (hv_analytic_target.norm().item() + 1e-12)
    print(f"  analytic HVP: {hv_analytic_target.tolist()}")
    print(f"  finite-diff HVP: {fd_hvp.tolist()}")
    print(f"  relative error: {rel_err:.2e}")
    passed = rel_err < 1e-2
    print(f"  PASS (rel_err < 1e-2): {passed}")
    return passed


if __name__ == "__main__":
    ok1 = verify_lanczos_on_synthetic_matrix()
    ok2 = verify_hvp_on_real_model()
    print(f"\n=== Overall: Lanczos verified={ok1}, HVP verified={ok2} ===")
    if not (ok1 and ok2):
        raise SystemExit("VERIFICATION FAILED -- do not proceed to the main E42 pipeline until this passes.")
    print("Both verifications passed. Safe to proceed.")
