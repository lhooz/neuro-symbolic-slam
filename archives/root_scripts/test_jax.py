import sys
sys.path.insert(0, '/Users/lhooz/.openclaw/workspace')
from src.snn_slam_twin import build_asymmetric_cann_weights, build_asymmetric_cann_weights_y
import jax.numpy as jnp

print("Testing vectorized asymmetric weights...")

try:
    Wx = build_asymmetric_cann_weights()
    print("Wx shape:", Wx.shape)
    print("Wx max:", float(Wx.max()))
except Exception as e:
    print("ERROR building Wx:", e)
