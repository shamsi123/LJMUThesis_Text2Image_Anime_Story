"""Narrative Graph Prompting -- training-free identity propagation.

Given a sequence of prompts and their independently rendered frames,
construct a graph where edges link consecutive frames whose prompts share
character/scene cues, then bias frame_t towards frame_{t-1} via image-space
blending (a stand-in for cross-attention biasing). Replace `propagate_identity`
with your own implementation if you have stronger NGP code."""
import re, numpy as np
from PIL import Image

def _scene_tokens(p):
    return set(re.findall(r"[A-Za-z][A-Za-z\-]+", p.lower()))


class NGPGraph:
    def __init__(self, prompts):
        self.prompts = prompts
        self.edges = []
        for i in range(len(prompts) - 1):
            a, b = _scene_tokens(prompts[i]), _scene_tokens(prompts[i + 1])
            sim = len(a & b) / max(1, len(a | b))
            if sim >= 0.3:
                self.edges.append((i, i + 1, sim))

    @classmethod
    def from_prompts(cls, prompts):
        return cls(prompts)


def propagate_identity(graph: NGPGraph, frames, alpha=0.4):
    """Blend each frame with its predecessor along graph edges.
    `frames` is a list of HxWx3 numpy uint8 arrays."""
    out = [f.copy() for f in frames]
    for i, j, sim in graph.edges:
        a = float(min(0.6, alpha + sim * 0.2))
        out[j] = (a * out[i].astype(np.float32) + (1 - a) * out[j].astype(np.float32)).astype(np.uint8)
    return out
