"""Hardware specifications, model configurations, and real-world catalogs.

All internal values use base units: bytes, FLOPS/s, bytes/s.
Conversions (GB, TFLOPS, etc.) happen only at display boundaries.

Last updated: February 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GB = 1 << 30          # 1 GiB in bytes
TB = 1 << 40          # 1 TiB in bytes
TFLOPS = 1e12         # 1 TFLOPS in FLOPS/s
BYTES_PER_FP16 = 2
BYTES_PER_INT8 = 1


# ---------------------------------------------------------------------------
# HardwareSpec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MemoryTier:
    """One memory tier in a hardware hierarchy.

    ``bandwidth`` is the tier's local service bandwidth.
    ``link_bandwidth_to_parent`` is transfer bandwidth to ``parent`` tier.
    """

    name: str
    capacity: int
    bandwidth: float
    parent: str | None = None
    link_bandwidth_to_parent: float = 0.0


@dataclass(frozen=True)
class HardwareSpec:
    """Specification for a single accelerator (GPU / TPU / SoC).

    All capacities in bytes, all throughputs in bytes/s or FLOPS/s.
    """

    name: str

    # Memory
    hbm_capacity: int            # bytes
    hbm_bandwidth: float         # bytes/s
    cpu_ram_capacity: int        # bytes (0 for unified-memory devices)
    cpu_gpu_bandwidth: float     # bytes/s (PCIe / UB / etc.)

    # Compute
    fp16_flops: float            # FLOPS/s (tensor-core FP16/BF16)
    int8_flops: float            # FLOPS/s (INT8 tensor ops)
    fp32_flops: float            # FLOPS/s (FP32 CUDA cores)

    # On-chip SRAM (L2 / scratchpad) — informational
    sram_capacity: int           # bytes

    # Interconnect (multi-GPU, informational for now)
    interconnect_bandwidth: float  # bytes/s (NVLink / Infinity Fabric / etc.)

    # Whether memory is unified (Apple, some TPUs)
    unified_memory: bool = False

    # Disk / NVMe (cold storage tier)
    disk_capacity: int = 0         # bytes (NVMe/SSD, 0 = none)
    disk_bandwidth: float = 0.0    # bytes/s (NVMe sequential read)
    tiers: tuple[MemoryTier, ...] = ()

    def __post_init__(self) -> None:
        if self.tiers:
            self._validate_tiers(self.tiers)
            return

        # hbm <- cpu <- disk (disk parent falls back to hbm when cpu is absent).
        built_tiers: list[MemoryTier] = [
            MemoryTier(
                name="hbm",
                capacity=self.hbm_capacity,
                bandwidth=self.hbm_bandwidth,
                parent=None,
                link_bandwidth_to_parent=0.0,
            )
        ]
        has_cpu = self.cpu_ram_capacity > 0 and self.cpu_gpu_bandwidth > 0
        if has_cpu:
            built_tiers.append(
                MemoryTier(
                    name="cpu",
                    capacity=self.cpu_ram_capacity,
                    bandwidth=self.cpu_gpu_bandwidth,
                    parent="hbm",
                    link_bandwidth_to_parent=self.cpu_gpu_bandwidth,
                )
            )
        if self.disk_capacity > 0 and self.disk_bandwidth > 0:
            built_tiers.append(
                MemoryTier(
                    name="disk",
                    capacity=self.disk_capacity,
                    bandwidth=self.disk_bandwidth,
                    parent="cpu" if has_cpu else "hbm",
                    link_bandwidth_to_parent=self.disk_bandwidth,
                )
            )

        built_tuple = tuple(built_tiers)
        self._validate_tiers(built_tuple)
        object.__setattr__(self, "tiers", built_tuple)

    @staticmethod
    def _validate_tiers(tiers: tuple[MemoryTier, ...]) -> None:
        if not tiers:
            raise ValueError("HardwareSpec.tiers cannot be empty")
        names = {t.name for t in tiers}
        if len(names) != len(tiers):
            raise ValueError("HardwareSpec.tiers contains duplicate tier names")
        if "hbm" not in names:
            raise ValueError("HardwareSpec.tiers must include an 'hbm' tier")
        for t in tiers:
            if t.parent is not None and t.parent not in names:
                raise ValueError(
                    f"Tier '{t.name}' references unknown parent '{t.parent}'"
                )

    def get_tier(self, name: str) -> MemoryTier | None:
        for tier in self.tiers:
            if tier.name == name:
                return tier
        return None

    def transfer_bandwidth_to_hbm(self, tier_name: str) -> float:
        """Effective transfer bandwidth from ``tier_name`` to HBM.

        Follows parent links up to ``hbm`` and takes the bottleneck link.
        Returns 0 when the tier does not exist or no path is available.
        """
        if tier_name == "hbm":
            return self.hbm_bandwidth
        tier_map = {t.name: t for t in self.tiers}
        start = tier_map.get(tier_name)
        if start is None:
            return 0.0

        cur = start
        bottleneck = float("inf")
        visited: set[str] = set()
        while cur.name != "hbm":
            if cur.name in visited:
                return 0.0
            visited.add(cur.name)
            if cur.parent is None or cur.link_bandwidth_to_parent <= 0:
                return 0.0
            bottleneck = min(bottleneck, cur.link_bandwidth_to_parent)
            parent = tier_map.get(cur.parent)
            if parent is None:
                return 0.0
            cur = parent
        return bottleneck if bottleneck != float("inf") else 0.0

    # --- computed properties ---------------------------------------------------

    @property
    def critical_batch_size_fp16(self) -> float:
        """Batch size at which MLP transitions from memory-bound to compute-bound.

        B_crit = FP16_FLOPS / HBM_BW  (arithmetic intensity crossover).
        """
        return self.fp16_flops / self.hbm_bandwidth

    def to_prompt_string(self) -> str:
        """Human-readable summary suitable for LLM prompts."""
        base = (
            f"{self.name}: "
            f"HBM {self.hbm_capacity / GB:.0f} GB @ {self.hbm_bandwidth / GB:.0f} GB/s, "
            f"FP16 {self.fp16_flops / TFLOPS:.0f} TFLOPS, "
            f"INT8 {self.int8_flops / TFLOPS:.0f} TFLOPS, "
            f"SRAM {self.sram_capacity / (1 << 20):.0f} MB, "
            f"{'unified' if self.unified_memory else f'CPU {self.cpu_ram_capacity / GB:.0f} GB @ {self.cpu_gpu_bandwidth / GB:.0f} GB/s'}"
        )
        if self.disk_capacity > 0:
            base += f", Disk {self.disk_capacity / GB:.0f} GB @ {self.disk_bandwidth / GB:.1f} GB/s"
        return base

    def to_tensor(self) -> np.ndarray:
        """Flat float32 feature vector (12 dims) for RL observation space.

        Log-scaled where appropriate so values span similar magnitudes.
        """
        return np.array([
            np.log2(self.hbm_capacity),
            np.log2(self.hbm_bandwidth),
            np.log2(self.cpu_ram_capacity + 1),
            np.log2(self.cpu_gpu_bandwidth + 1),
            np.log2(self.fp16_flops),
            np.log2(self.int8_flops),
            np.log2(self.fp32_flops),
            np.log2(self.sram_capacity + 1),
            np.log2(self.interconnect_bandwidth + 1),
            float(self.unified_memory),
            np.log2(self.disk_capacity + 1),
            np.log2(self.disk_bandwidth + 1),
        ], dtype=np.float32)


# ---------------------------------------------------------------------------
# ModelConfig (dense only)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    """Transformer model architecture parameters (dense models only)."""

    name: str
    num_layers: int
    d_model: int
    num_heads: int          # query heads
    num_kv_heads: int       # key/value heads (GQA); equals num_heads for MHA
    d_ff: int               # FFN intermediate dimension
    vocab_size: int
    head_dim: int | None = None   # defaults to d_model // num_heads
    bytes_per_param: int = 2      # 2 = bf16/fp16

    def __post_init__(self) -> None:
        if self.head_dim is None:
            object.__setattr__(self, "head_dim", self.d_model // self.num_heads)

    # --- computed properties ---------------------------------------------------

    @property
    def _head_dim(self) -> int:
        return self.head_dim if self.head_dim is not None else self.d_model // self.num_heads

    @property
    def num_params(self) -> int:
        """Total parameter count (approximate, excludes biases and norms)."""
        attn_per_layer = self.attn_params_per_layer
        mlp_per_layer = self.mlp_params_per_layer
        per_layer = attn_per_layer + mlp_per_layer
        embedding = self.vocab_size * self.d_model  # often tied
        return per_layer * self.num_layers + embedding

    @property
    def params_per_layer(self) -> int:
        """Parameters per transformer layer (attention + FFN)."""
        return self.attn_params_per_layer + self.mlp_params_per_layer

    @property
    def attn_params_per_layer(self) -> int:
        """Attention projection parameters per layer.

        Q: d_model * num_heads * head_dim
        K: d_model * num_kv_heads * head_dim
        V: d_model * num_kv_heads * head_dim
        O: num_heads * head_dim * d_model
        """
        h = self._head_dim
        q_params = self.d_model * self.num_heads * h
        k_params = self.d_model * self.num_kv_heads * h
        v_params = self.d_model * self.num_kv_heads * h
        o_params = self.num_heads * h * self.d_model
        return q_params + k_params + v_params + o_params

    @property
    def mlp_params_per_layer(self) -> int:
        """MLP (FFN) parameters per layer.

        SwiGLU: 3 * d_model * d_ff  (gate + up + down)
        """
        return 3 * self.d_model * self.d_ff

    @property
    def param_bytes(self) -> int:
        """Total model size in bytes."""
        return self.num_params * self.bytes_per_param

    @property
    def kv_bytes_per_token(self) -> int:
        """KV cache bytes per token (full precision, all layers).

        Per layer: 2 (K+V) * num_kv_heads * head_dim * bytes_per_param
        Total:     num_layers * per_layer
        """
        per_layer = 2 * self.num_kv_heads * self._head_dim * self.bytes_per_param
        return per_layer * self.num_layers

    def kv_bytes_per_token_per_layer(self) -> int:
        """KV cache bytes per token for a single layer."""
        return 2 * self.num_kv_heads * self._head_dim * self.bytes_per_param

    def kv_cache_bytes(self, seq_len: int) -> int:
        """Total KV cache size for a given sequence length."""
        return self.kv_bytes_per_token * seq_len


# ---------------------------------------------------------------------------
# Hardware catalog — 16 real accelerators (as of Feb 2026)
# All FP16/INT8 values are dense (no 2:4 sparsity) unless noted.
# Datacenter GPUs get typical NVMe: 2 TB capacity, 5 GB/s sequential read.
# Edge / unified-memory devices get disk=0 (no local NVMe).
# ---------------------------------------------------------------------------
def get_hardware_specs() -> dict[str, HardwareSpec]:
    """Return a catalog of real hardware specs keyed by name."""
    specs = {}

    # --- NVIDIA ----------------------------------------------------------------

    specs["A100_80GB"] = HardwareSpec(
        name="A100_80GB",
        hbm_capacity=80 * GB,
        hbm_bandwidth=2.0 * TB,       # 2039 GB/s HBM2e
        cpu_ram_capacity=512 * GB,
        cpu_gpu_bandwidth=64 * GB,     # PCIe Gen4 x16
        fp16_flops=312 * TFLOPS,
        int8_flops=624 * TFLOPS,
        fp32_flops=19.5 * TFLOPS,
        sram_capacity=40 * (1 << 20),  # 40 MB L2
        interconnect_bandwidth=600 * GB,  # NVLink 3
        disk_capacity=2 * TB,
        disk_bandwidth=5 * GB,
    )

    specs["H100_SXM"] = HardwareSpec(
        name="H100_SXM",
        hbm_capacity=80 * GB,
        hbm_bandwidth=3.35 * TB,       # 3350 GB/s HBM3
        cpu_ram_capacity=512 * GB,
        cpu_gpu_bandwidth=128 * GB,    # PCIe Gen5 x16
        fp16_flops=990 * TFLOPS,
        int8_flops=1979 * TFLOPS,
        fp32_flops=67 * TFLOPS,
        sram_capacity=50 * (1 << 20),  # 50 MB L2
        interconnect_bandwidth=900 * GB,  # NVLink 4
        disk_capacity=2 * TB,
        disk_bandwidth=5 * GB,
    )

    specs["H200"] = HardwareSpec(
        name="H200",
        hbm_capacity=141 * GB,
        hbm_bandwidth=4.8 * TB,        # 4800 GB/s HBM3e
        cpu_ram_capacity=512 * GB,
        cpu_gpu_bandwidth=128 * GB,
        fp16_flops=990 * TFLOPS,       # same Hopper die as H100
        int8_flops=1979 * TFLOPS,
        fp32_flops=67 * TFLOPS,
        sram_capacity=50 * (1 << 20),
        interconnect_bandwidth=900 * GB,
        disk_capacity=2 * TB,
        disk_bandwidth=5 * GB,
    )

    specs["B200"] = HardwareSpec(
        name="B200",
        hbm_capacity=192 * GB,
        hbm_bandwidth=8.0 * TB,        # 8000 GB/s HBM3e
        cpu_ram_capacity=512 * GB,
        cpu_gpu_bandwidth=128 * GB,
        fp16_flops=2250 * TFLOPS,      # 5th-gen Tensor Cores, FP16/BF16
        int8_flops=4500 * TFLOPS,
        fp32_flops=80 * TFLOPS,
        sram_capacity=126 * (1 << 20), # 126 MB L2
        interconnect_bandwidth=1800 * GB,  # NVLink 5
        disk_capacity=2 * TB,
        disk_bandwidth=5 * GB,
    )

    specs["B300"] = HardwareSpec(
        name="B300",
        hbm_capacity=288 * GB,         # HBM3e 12-high stacks
        hbm_bandwidth=8.0 * TB,        # 8000 GB/s
        cpu_ram_capacity=512 * GB,
        cpu_gpu_bandwidth=128 * GB,    # PCIe Gen5 x16
        fp16_flops=2500 * TFLOPS,
        int8_flops=5000 * TFLOPS,
        fp32_flops=75 * TFLOPS,
        sram_capacity=126 * (1 << 20),
        interconnect_bandwidth=1800 * GB,  # NVLink 5
        disk_capacity=2 * TB,
        disk_bandwidth=5 * GB,
    )

    specs["L40S"] = HardwareSpec(
        name="L40S",
        hbm_capacity=48 * GB,
        hbm_bandwidth=864 * GB,        # 864 GB/s GDDR6
        cpu_ram_capacity=256 * GB,
        cpu_gpu_bandwidth=64 * GB,
        fp16_flops=362 * TFLOPS,
        int8_flops=733 * TFLOPS,
        fp32_flops=91.6 * TFLOPS,
        sram_capacity=96 * (1 << 20),  # 96 MB L2
        interconnect_bandwidth=0,       # no NVLink on L40S
        disk_capacity=2 * TB,
        disk_bandwidth=5 * GB,
    )

    specs["RTX_5090"] = HardwareSpec(
        name="RTX_5090",
        hbm_capacity=32 * GB,          # 32 GB GDDR7
        hbm_bandwidth=1.79 * TB,       # 1792 GB/s
        cpu_ram_capacity=192 * GB,
        cpu_gpu_bandwidth=64 * GB,     # PCIe Gen5 x16
        fp16_flops=209.5 * TFLOPS,    # FP16/BF16 dense tensor (5th-gen)
        int8_flops=838 * TFLOPS,      # INT8 dense tensor
        fp32_flops=104.8 * TFLOPS,    # FP32 shader (CUDA cores)
        sram_capacity=96 * (1 << 20),  # 96 MB L2
        interconnect_bandwidth=0,       # no NVLink on consumer cards
        disk_capacity=2 * TB,
        disk_bandwidth=5 * GB,
    )

    # --- AMD ------------------------------------------------------------------

    specs["MI300X"] = HardwareSpec(
        name="MI300X",
        hbm_capacity=192 * GB,
        hbm_bandwidth=5.3 * TB,        # 5300 GB/s HBM3
        cpu_ram_capacity=512 * GB,
        cpu_gpu_bandwidth=128 * GB,    # PCIe Gen5
        fp16_flops=1307 * TFLOPS,
        int8_flops=2615 * TFLOPS,
        fp32_flops=163 * TFLOPS,
        sram_capacity=256 * (1 << 20), # 256 MB Infinity Cache
        interconnect_bandwidth=896 * GB,  # Infinity Fabric
        disk_capacity=2 * TB,
        disk_bandwidth=5 * GB,
    )

    specs["MI325X"] = HardwareSpec(
        name="MI325X",
        hbm_capacity=256 * GB,
        hbm_bandwidth=6.0 * TB,        # 6000 GB/s HBM3e
        cpu_ram_capacity=512 * GB,
        cpu_gpu_bandwidth=128 * GB,
        fp16_flops=1307 * TFLOPS,      # same CDNA 3 die as MI300X
        int8_flops=2615 * TFLOPS,
        fp32_flops=163 * TFLOPS,
        sram_capacity=256 * (1 << 20),
        interconnect_bandwidth=896 * GB,
        disk_capacity=2 * TB,
        disk_bandwidth=5 * GB,
    )

    specs["MI350X"] = HardwareSpec(
        name="MI350X",
        hbm_capacity=288 * GB,
        hbm_bandwidth=8.0 * TB,        # 8000 GB/s HBM3e
        cpu_ram_capacity=512 * GB,
        cpu_gpu_bandwidth=128 * GB,
        fp16_flops=2307 * TFLOPS,      # CDNA 4
        int8_flops=4614 * TFLOPS,
        fp32_flops=144 * TFLOPS,
        sram_capacity=256 * (1 << 20),
        interconnect_bandwidth=896 * GB,
        disk_capacity=2 * TB,
        disk_bandwidth=5 * GB,
    )

    # --- Google TPU -----------------------------------------------------------

    specs["TPU_v5e"] = HardwareSpec(
        name="TPU_v5e",
        hbm_capacity=16 * GB,
        hbm_bandwidth=820 * GB,
        cpu_ram_capacity=0,
        cpu_gpu_bandwidth=0,
        fp16_flops=197 * TFLOPS,
        int8_flops=394 * TFLOPS,
        fp32_flops=49 * TFLOPS,
        sram_capacity=128 * (1 << 20), # 128 MiB VMEM
        interconnect_bandwidth=1600 * GB,  # ICI
        unified_memory=True,
    )

    specs["TPU_v6e"] = HardwareSpec(
        name="TPU_v6e",
        hbm_capacity=32 * GB,
        hbm_bandwidth=1640 * GB,
        cpu_ram_capacity=0,
        cpu_gpu_bandwidth=0,
        fp16_flops=918 * TFLOPS,       # 4.7x TPU v5e (Trillium)
        int8_flops=1836 * TFLOPS,
        fp32_flops=230 * TFLOPS,       # estimated
        sram_capacity=128 * (1 << 20),
        interconnect_bandwidth=3200 * GB,
        unified_memory=True,
    )

    specs["TPU_v7"] = HardwareSpec(
        name="TPU_v7",
        hbm_capacity=192 * GB,
        hbm_bandwidth=7.37 * TB,       # 7370 GB/s HBM3e (Ironwood)
        cpu_ram_capacity=0,
        cpu_gpu_bandwidth=0,
        fp16_flops=2300 * TFLOPS,
        int8_flops=4600 * TFLOPS,
        fp32_flops=575 * TFLOPS,       # estimated (no FP32 MXU)
        sram_capacity=256 * (1 << 20), # estimated
        interconnect_bandwidth=1200 * GB,  # ICI per chip
        unified_memory=True,
    )

    # --- Intel ----------------------------------------------------------------

    specs["Gaudi_3"] = HardwareSpec(
        name="Gaudi_3",
        hbm_capacity=128 * GB,
        hbm_bandwidth=3.7 * TB,        # 3700 GB/s HBM2e
        cpu_ram_capacity=512 * GB,
        cpu_gpu_bandwidth=128 * GB,    # PCIe Gen5
        fp16_flops=1835 * TFLOPS,
        int8_flops=1835 * TFLOPS,      # same rate as BF16 on Gaudi
        fp32_flops=28.7 * TFLOPS,      # vector FP32
        sram_capacity=96 * (1 << 20),  # 48 MB per die x2
        interconnect_bandwidth=1200 * GB,  # 24x 200GbE
        disk_capacity=2 * TB,
        disk_bandwidth=5 * GB,
    )

    # --- Apple Silicon --------------------------------------------------------

    specs["M4_Max"] = HardwareSpec(
        name="M4_Max",
        hbm_capacity=128 * GB,         # unified, max config
        hbm_bandwidth=546 * GB,        # 546 GB/s (40-core GPU)
        cpu_ram_capacity=0,
        cpu_gpu_bandwidth=0,
        fp16_flops=18.4 * TFLOPS,      # 40-core GPU
        int8_flops=36.8 * TFLOPS,      # estimated 2x FP16
        fp32_flops=18.4 * TFLOPS,      # Apple GPU: FP16 = FP32 rate
        sram_capacity=48 * (1 << 20),
        interconnect_bandwidth=0,
        unified_memory=True,
    )

    # --- Qualcomm -------------------------------------------------------------

    specs["Snapdragon_X_Elite"] = HardwareSpec(
        name="Snapdragon_X_Elite",
        hbm_capacity=32 * GB,          # LPDDR5X unified
        hbm_bandwidth=136 * GB,        # 136 GB/s LPDDR5X-8448
        cpu_ram_capacity=0,
        cpu_gpu_bandwidth=0,
        fp16_flops=9.2 * TFLOPS,       # Adreno GPU FP16
        int8_flops=45 * TFLOPS,        # Hexagon NPU INT8
        fp32_flops=4.6 * TFLOPS,       # Adreno GPU FP32
        sram_capacity=12 * (1 << 20),
        interconnect_bandwidth=0,
        unified_memory=True,
    )

    return specs


# ---------------------------------------------------------------------------
# Model catalog — 14 dense models (MoE deferred to weeks 5+)
# ---------------------------------------------------------------------------
def get_model_configs() -> dict[str, ModelConfig]:
    """Return a catalog of model configs keyed by name."""
    models = {}

    # =========================================================================
    # Reference / small models
    # =========================================================================

    models["Tiny-1B"] = ModelConfig(
        name="Tiny-1B",
        num_layers=22,
        d_model=2048,
        num_heads=32,
        num_kv_heads=4,
        d_ff=5632,
        vocab_size=32000,
    )

    # =========================================================================
    # Meta LLaMA 3.1 / 3.2 / 3.3  (dense, open-weight)
    # =========================================================================

    models["LLaMA-3.1-8B"] = ModelConfig(
        name="LLaMA-3.1-8B",
        num_layers=32,
        d_model=4096,
        num_heads=32,
        num_kv_heads=8,        # GQA
        d_ff=14336,
        vocab_size=128256,
    )

    models["LLaMA-3.1-70B"] = ModelConfig(
        name="LLaMA-3.1-70B",
        num_layers=80,
        d_model=8192,
        num_heads=64,
        num_kv_heads=8,        # GQA
        d_ff=28672,
        vocab_size=128256,
    )

    models["LLaMA-3.2-3B"] = ModelConfig(
        name="LLaMA-3.2-3B",
        num_layers=28,
        d_model=3072,
        num_heads=24,
        num_kv_heads=8,        # GQA
        d_ff=8192,
        vocab_size=128256,
    )

    models["LLaMA-3.3-70B"] = ModelConfig(
        name="LLaMA-3.3-70B",
        num_layers=80,
        d_model=8192,
        num_heads=64,
        num_kv_heads=8,
        d_ff=28672,
        vocab_size=128256,
    )

    # =========================================================================
    # Qwen 2.5  (dense, strong baselines)
    # =========================================================================

    models["Qwen2.5-14B"] = ModelConfig(
        name="Qwen2.5-14B",
        num_layers=48,
        d_model=5120,
        num_heads=40,
        num_kv_heads=8,        # GQA
        d_ff=13824,
        vocab_size=152064,
    )

    models["Qwen2.5-72B"] = ModelConfig(
        name="Qwen2.5-72B",
        num_layers=80,
        d_model=8192,
        num_heads=64,
        num_kv_heads=8,        # GQA
        d_ff=29568,
        vocab_size=152064,
    )

    # =========================================================================
    # Qwen 3  (dense, hybrid thinking/non-thinking, Apr 2025)
    # =========================================================================

    models["Qwen3-4B"] = ModelConfig(
        name="Qwen3-4B",
        num_layers=36,
        d_model=2560,
        num_heads=32,
        num_kv_heads=8,        # GQA
        d_ff=9728,
        vocab_size=151936,
        head_dim=128,          # decoupled from d_model/num_heads
    )

    models["Qwen3-8B"] = ModelConfig(
        name="Qwen3-8B",
        num_layers=36,
        d_model=4096,
        num_heads=32,
        num_kv_heads=8,        # GQA (4 Q heads per KV head)
        d_ff=12288,
        vocab_size=151936,
        head_dim=128,
    )

    models["Qwen3-32B"] = ModelConfig(
        name="Qwen3-32B",
        num_layers=64,
        d_model=5120,
        num_heads=64,
        num_kv_heads=8,        # GQA
        d_ff=25600,
        vocab_size=151936,
        head_dim=128,
    )

    # =========================================================================
    # Google Gemma 3  (dense, multimodal, Mar 2025)
    # =========================================================================

    models["Gemma-3-4B"] = ModelConfig(
        name="Gemma-3-4B",
        num_layers=34,
        d_model=2560,
        num_heads=8,
        num_kv_heads=4,        # GQA
        d_ff=10240,
        vocab_size=262208,
        head_dim=256,          # NOT d_model//num_heads (would be 320)
    )

    models["Gemma-3-27B"] = ModelConfig(
        name="Gemma-3-27B",
        num_layers=62,
        d_model=5376,
        num_heads=32,
        num_kv_heads=16,       # GQA
        d_ff=21504,
        vocab_size=262208,
        head_dim=128,          # NOT d_model//num_heads (would be 168)
    )

    # =========================================================================
    # Microsoft Phi  (dense, small but capable)
    # =========================================================================

    models["Phi-4-14B"] = ModelConfig(
        name="Phi-4-14B",
        num_layers=40,
        d_model=5120,
        num_heads=40,
        num_kv_heads=10,       # GQA
        d_ff=17920,
        vocab_size=100352,
    )

    models["Phi-4-mini-3.8B"] = ModelConfig(
        name="Phi-4-mini-3.8B",
        num_layers=32,
        d_model=3072,
        num_heads=24,
        num_kv_heads=8,        # GQA
        d_ff=8192,
        vocab_size=200064,
    )

    return models
