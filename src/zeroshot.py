"""Zero-shot point-cloud classification with CLIP, following PointCLIP.

Each point cloud is rendered into depth images from several viewpoints. The images are encoded
with the CLIP image encoder and compared with text embeddings of prompts such as
"a depth map of a chair.". The predicted class is the one with the highest mean similarity.

References: Zhang et al., PointCLIP (CVPR 2022); Zhu et al., PointCLIP V2 (ICCV 2023).
Rendering: nearest-point depth buffer, point splatting (max-pooling) and Gaussian smoothing.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

PRETTY = {"night_stand": "nightstand", "flower_pot": "flower pot", "tv_stand": "tv stand",
          "range_hood": "range hood", "glass_box": "glass box", "xbox": "Xbox", "sink": "sink"}

PROMPTS = [
    "a depth map of a {}.",
    "a point cloud depth map of a {}.",
    "a grayscale 3D rendering of a {}.",
    "a 3D model of a {}.",
    "a silhouette of a {}.",
    "a photo of a {}.",
]


def view_matrices(n_azim=8, elevations=(20.0,), add_top=True, up_axis=2):
    """Camera frames [V,3,3] (rows = image-right, image-up, toward-camera) on a ring around the up axis."""
    up = np.eye(3)[up_axis]
    dirs = []
    for e in elevations:
        for k in range(n_azim):
            a, el = 2 * math.pi * k / n_azim, math.radians(e)
            h = [i for i in range(3) if i != up_axis]
            c = np.zeros(3)
            c[h[0]], c[h[1]], c[up_axis] = math.cos(el) * math.cos(a), math.cos(el) * math.sin(a), math.sin(el)
            dirs.append(c)
    if add_top:
        dirs.append(up * 1.0)
    mats = []
    for c in dirs:
        ref = up if abs(np.dot(c, up)) < 0.99 else np.eye(3)[(up_axis + 1) % 3]
        right = np.cross(ref, c)
        right /= np.linalg.norm(right)
        vup = np.cross(c, right)
        mats.append(np.stack([right, vup, c]))
    return torch.tensor(np.stack(mats), dtype=torch.float32)


def render_depth(pts, mats, res=128, out_size=224, splat=3, blur_sigma=1.0, background="white"):
    """pts [B,N,3] (unit sphere) -> images [B*V,3,out,out] ready for CLIP (normalised)."""
    B, N, _ = pts.shape
    V = mats.shape[0]
    p = torch.einsum("vij,bnj->bvni", mats.to(pts.device), pts)  # [B,V,N,3]
    u = ((p[..., 0] * 0.9 + 1) / 2 * (res - 1)).round().long().clamp(0, res - 1)
    v = ((1 - (p[..., 1] * 0.9 + 1) / 2) * (res - 1)).round().long().clamp(0, res - 1)
    near = (p[..., 2] + 1) / 2  # 1 = closest to the camera
    flat = (v * res + u).reshape(B * V, N)
    img = torch.full((B * V, res * res), -1.0, device=pts.device)
    img = img.scatter_reduce(1, flat, near.reshape(B * V, N), reduce="amax", include_self=True)
    img = img.view(B * V, 1, res, res)
    if splat > 1:
        img = F.max_pool2d(img, splat, stride=1, padding=splat // 2)
    inten = torch.where(img >= 0, 0.25 + 0.75 * img.clamp(min=0), torch.zeros_like(img))  # near = bright
    if blur_sigma > 0:
        k = int(2 * math.ceil(2 * blur_sigma) + 1)
        x = torch.arange(k, device=pts.device, dtype=torch.float32) - k // 2
        g = torch.exp(-x ** 2 / (2 * blur_sigma ** 2))
        g = (g / g.sum()).view(1, 1, 1, k)
        inten = F.conv2d(F.conv2d(inten, g, padding=(0, k // 2)), g.transpose(2, 3), padding=(k // 2, 0))
    if background == "white":  # dark object on a white background
        inten = 1 - inten
    img = F.interpolate(inten, size=out_size, mode="bilinear", align_corners=False).clamp(0, 1)
    return img.repeat(1, 3, 1, 1)


class CLIPZeroShot:
    def __init__(self, arch="ViT-B/32", device="cuda", classes=None, prompts=PROMPTS,
                 n_azim=8, elevations=(20.0,), add_top=True, background="white"):
        import clip
        self.model, _ = clip.load(arch, device=device, jit=False)
        self.model.eval()
        self.device = device
        self.size = self.model.visual.input_resolution
        self.mats = view_matrices(n_azim, elevations, add_top)
        self.background = background
        self.set_classes(classes, prompts)

    @torch.no_grad()
    def set_classes(self, classes, prompts=PROMPTS):
        import clip
        self.classes = list(classes)
        embs = []
        for c in self.classes:
            name = PRETTY.get(c, c.replace("_", " "))
            t = clip.tokenize([p.format(name) for p in prompts]).to(self.device)
            e = F.normalize(self.model.encode_text(t).float(), dim=-1).mean(0)  # prompt ensemble
            embs.append(F.normalize(e, dim=0))
        self.text = torch.stack(embs)  # [C,D]

    @torch.no_grad()
    def images(self, pts):
        return render_depth(pts.to(self.device).float(), self.mats, out_size=self.size, background=self.background)

    @torch.no_grad()
    def logits(self, pts):
        """pts [B,N,3] -> class logits [B,C] (mean over views of cosine similarity x 100)."""
        B, V = pts.shape[0], self.mats.shape[0]
        img = (self.images(pts) - CLIP_MEAN.to(self.device)) / CLIP_STD.to(self.device)
        f = F.normalize(self.model.encode_image(img.to(self.model.dtype)).float(), dim=-1)
        sim = 100.0 * f @ self.text.T  # [B*V,C]
        return sim.view(B, V, -1).mean(1)
