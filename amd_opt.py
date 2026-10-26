"""
AMD / ROCm-specific setup for nanoGPT on gfx1101 (RX 7800 XT).

Import this module BEFORE torch, from the very top of train.py:

    import amd_opt        # must precede 'import torch'
    import torch

Why: PyTorch's ROCm SDPA dispatch reads TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL
from the process environment. Without it, scaled_dot_product_attention on RDNA3
has NO flash and NO memory-efficient kernel available, so it silently falls back
to the MATH backend, which materialises the full B x heads x T x T attention
matrix for every layer. That is both ~19x slower and ~4x more VRAM hungry.

Everything here is opt-outable with NANOAMD=0 -- the env vars above, the SDPA backend
lockdown, and the VRAM cap alike. configure() returns early and leaves the process alone.

Scope: this is amdgpu-specific machinery. configure() checks torch.version.hip / a gfx*
gcnArchName first, so an NVIDIA or other build is left untouched rather than being
clamped at 8 GiB by a rule written to protect a Linux desktop session.

Device safety
-------------
Device 1 (RX 7800 XT, gfx1101, PCI 0000:06:00.0) is ALSO the GPU driving the
desktop: its DP-3 output is connected, while every output on device 0
(0000:03:00.0, the 7900 XTX) is disconnected. That fact is detected at runtime
from /sys/class/drm, not assumed.

Why the cap is not optional: when a GPU runs out of VRAM, amdgpu does not fail
cleanly -- it EVICTS, moving pages into system RAM (GTT). With an LLM already
holding most of the 62 GiB of host RAM, that eviction lands on a machine with no
free RAM, and the compositor is what gets pushed out. The session dies. A 12 GiB
cap on a 16 GiB display GPU was enough to do this twice.

So: if the GPU you picked has a connected display output, the ceiling is
_DISPLAY_CEILING (8 GiB by default) and it is CLAMPED, even if
NANOAMD_MAX_VRAM_GIB asks for more. Getting past the clamp takes an explicit
NANOAMD_ALLOW_DISPLAY_OVERRIDE=1. If the GPU has no connected outputs (the 24 GB
card, or the desktop is closed), there is no cap unless you ask for one. If the box
cannot tell -- /sys/class/drm unreadable, or a display is live on some card but this
GPU's BDF could not be resolved -- it assumes the worst and clamps.

Choose the GPU with NANOAMD_DEVICE (default 1). Set NANOAMD_DEVICE=0 to use the
7900 XTX once the LLM on it has been stopped. That default is this box's second card,
so it is only applied when the kfd topology says that index exists: on a one-GPU box
pinning to 1 would hide the only GPU and make torch report zero devices.

Measured headroom: micro-batch 8 at T=1024 peaks at 7.18 GiB allocated /
7.23 GiB reserved, which fits under the 8 GiB ceiling. Micro-batch 12 needs
9.6 GiB and micro-batch 16 needs more than 12 -- on a display GPU, do not.
"""
import os
import subprocess

_ENABLE = os.environ.get('NANOAMD', '1') != '0'

# GiB this process may take. Unset = decide automatically from whether the target
# GPU has a connected display (8 GiB if yes, no cap if no). 0 = explicitly no cap.
_MAX_VRAM_GIB = os.environ.get('NANOAMD_MAX_VRAM_GIB', '')
# Hard ceiling for a GPU that draws the desktop. Clamped unless explicitly overridden.
_DISPLAY_CEILING = float(os.environ.get('NANOAMD_DISPLAY_CEILING', '8'))
_OVERRIDE = os.environ.get('NANOAMD_ALLOW_DISPLAY_OVERRIDE', '0') == '1'
# Physical GPU to use: 1 = RX 7800 XT (display GPU), 0 = RX 7900 XTX (headless).
_DEVICE = os.environ.get('NANOAMD_DEVICE', '1')


def _rocm_gpu_node_count():
    """GPU nodes in the kfd topology; 0 when this is not a ROCm box at all."""
    try:
        nodes = os.listdir('/sys/class/kfd/kfd/topology/nodes')
    except OSError:
        return 0
    n = 0
    for node in nodes:
        try:
            with open(f'/sys/class/kfd/kfd/topology/nodes/{node}/gpu_id') as f:
                if f.read().strip() not in ('', '0'):
                    n += 1
        except OSError:
            continue
    return n

def _looks_like_rocm_box():
    """Cheap pre-torch check: ROCm cannot drive a GPU without /dev/kfd, so a box with
    neither that node nor a GPU in the kfd topology is not one this module has opinions
    for. Skip the block rather than setting OMP_NUM_THREADS=4 and
    TORCHINDUCTOR_COMPILE_THREADS=4 on someone else's machine."""
    return os.path.exists('/dev/kfd') or _rocm_gpu_node_count() > 0


# --- environment variables: must be set before torch/HIP is loaded ---
if _ENABLE and not _looks_like_rocm_box():
    print('[amd_opt] no /dev/kfd and no GPU in the kfd topology: not a ROCm box, '
          'leaving the process environment alone')
elif _ENABLE:
    # 1. Unlock AOTriton's flash / memory-efficient attention kernels for RDNA3.
    os.environ.setdefault('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', '1')
    # 2. The caching allocator fragments badly when the per-step working set is a
    #    large fraction of 16 GB; expandable segments reclaim most of that loss.
    os.environ.setdefault('PYTORCH_HIP_ALLOC_CONF', 'expandable_segments:True')
    # 3. Keep ROCm from over-subscribing CPU threads while the LLM on GPU0 is also
    #    running; 16 logical cores, one process.
    os.environ.setdefault('OMP_NUM_THREADS', '4')

    # 4. Inductor's FX-graph cache defaults to /tmp/torchinductor_$USER. On this machine
    #    /tmp is tmpfs (31 GiB, charged to host RAM, cleared by every boot), so a run
    #    re-pays its entire compile after a reboot even when only log_interval changed.
    #    Keep it on the NVMe instead: survives reboots, and costs no host RAM.
    _cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', 'torchinductor')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', _cache_dir)
    try:
        os.makedirs(_cache_dir, exist_ok=True)
    except OSError:
        print(f'[amd_opt] could not create {_cache_dir}; inductor will use /tmp')

    # 5. torch's default here is cpu_count//2 = 16 async compile worker processes, each
    #    one importing torch (~0.5-1.5 GiB resident). That is roughly 10 GiB of host RAM
    #    asked for at once, in a box where the LLM already holds ~52 GiB and zram is
    #    full. Four workers costs about the same wall-clock time and far less RAM.
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '4')

    # Pin to the requested GPU: HIP_VISIBLE_DEVICES=1 => RX 7800 XT, index 0 in-process.
    # Set ONLY HIP_VISIBLE_DEVICES. Setting ROCR_VISIBLE_DEVICES as well double-filters
    # the device list (ROCR renumbers to 0, then HIP looks for index 1) and PyTorch
    # then sees zero GPUs.
    # Without any pinning, torch.cuda.device_count()==2 and 'cuda' means the 7900 XTX
    # that the LLM is resident on -- stock nanoGPT will OOM or fight it for VRAM.
    #
    # But only pin an index that actually exists here. NANOAMD_DEVICE defaults to 1
    # because this box has two cards; on a one-GPU box (or a box with no kfd topology,
    # i.e. not ROCm) pinning to 1 means torch sees zero devices and train.py dies on
    # device='cuda' with a confusing error.
    if 'HIP_VISIBLE_DEVICES' not in os.environ and 'ROCR_VISIBLE_DEVICES' not in os.environ:
        _n_gpus = _rocm_gpu_node_count()
        _want = int(_DEVICE) if _DEVICE.lstrip('-').isdigit() else 0
        if _n_gpus == 0:
            print('[amd_opt] no kfd topology: this does not look like a ROCm box, '
                  'so no device is pinned')
        elif _want >= _n_gpus:
            print(f'[amd_opt] NANOAMD_DEVICE={_DEVICE} but only {_n_gpus} ROCm GPU(s) '
                  f'present; not pinning rather than hiding every device')
        else:
            os.environ['HIP_VISIBLE_DEVICES'] = _DEVICE


def _env_summary():
    keys = ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', 'PYTORCH_HIP_ALLOC_CONF',
            'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'OMP_NUM_THREADS',
            'TORCHINDUCTOR_CACHE_DIR', 'TORCHINDUCTOR_COMPILE_THREADS']
    return {k: os.environ.get(k) for k in keys}


def configure(dtype='bfloat16', tunable=False, strict_flash=True, verbose=True):
    """Called after torch is imported. Returns the device string to use."""
    import torch

    if not torch.cuda.is_available():
        if verbose:
            print('[amd_opt] no HIP device visible, nothing to configure')
        return 'cpu'

    # NANOAMD=0 is documented as opting out of everything here, so it has to gate this
    # function as well as the env block above -- the VRAM cap and the SDPA lockdown are
    # exactly the two things a reader would expect that switch to turn off.
    if not _ENABLE:
        if verbose:
            print('[amd_opt] NANOAMD=0: no env changes, no SDPA lockdown, no VRAM cap')
        return 'cuda'

    props = torch.cuda.get_device_properties(0)
    arch = (getattr(props, 'gcnArchName', '') or '').split(':')[0]
    hip = getattr(torch.version, 'hip', None)
    if verbose:
        print(f"[amd_opt] using {props.name} ({arch or 'no gcnArchName'}), "
              f"{props.total_memory/2**30:.1f} GiB, {torch.cuda.device_count()} visible device(s)")
        print(f"[amd_opt] env: {_env_summary()}")

    # Everything below is amdgpu-specific. On an NVIDIA build torch.version.hip is None
    # and there is no gcnArchName, so the display/cap logic has nothing to key on and
    # would clamp an 80 GB card at 8 GiB while warning about a desktop it cannot see.
    if not (hip or arch.startswith('gfx')):
        if verbose:
            print(f"[amd_opt] {props.name}: no HIP in this torch build, so this module leaves "
                  "the process alone (no cap, no SDPA lockdown, no TunableOp)")
        return 'cuda'

    # Is this the GPU that draws the desktop? Decided from sysfs, every run.
    target = _visible_physical_index()
    displays = _connected_display_outputs()
    bdf, _market = _gpu_identity(target)
    connector = displays.get(bdf) if (displays and bdf) else None
    requested = _parse_gib(_MAX_VRAM_GIB)

    cap, reason = _vram_cap(bdf, connector, displays, requested, _DISPLAY_CEILING, _OVERRIDE)

    if reason.startswith('display'):
        print(f"[amd_opt] WARNING: {bdf} (NANOAMD_DEVICE={_DEVICE}) has {connector} CONNECTED.")
        print(f"[amd_opt]   This GPU draws your desktop. Overflowing its VRAM makes amdgpu")
        print(f"[amd_opt]   evict pages into system RAM, and with the LLM holding most of")
        print(f"[amd_opt]   host RAM the compositor is what gets pushed out.")
        if reason == 'display_clamped':
            print(f"[amd_opt] ignoring NANOAMD_MAX_VRAM_GIB={requested:.1f}: clamped to "
                  f"{cap:.1f} GiB because {bdf} has a connected display")
        elif reason == 'display_override':
            print(f"[amd_opt] NANOAMD_ALLOW_DISPLAY_OVERRIDE=1: raising the cap to {cap:.1f} GiB "
                  f"on a GPU that draws the desktop. This is how the desktop gets killed; "
                  f"nothing here will stop it.")
    elif reason == 'headless':
        if verbose:
            print(f"[amd_opt] {bdf or 'this GPU'} has no connected outputs: headless, "
                  "no VRAM cap unless requested")
    else:
        print(f"[amd_opt] could not determine whether {bdf or _DEVICE} drives a display; "
              f"assuming it does and capping at {cap:.1f} GiB")

    # Hard VRAM ceiling for this process.
    if cap and cap > 0:
        total_gib = props.total_memory / 2**30
        frac = min(0.98, cap / total_gib)
        torch.cuda.set_per_process_memory_fraction(frac)
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            free_gib = free_b / 2**30
        except Exception:
            free_gib = float('nan')
        if verbose:
            print(f"[amd_opt] VRAM cap: {frac*100:.0f}% of {total_gib:.1f} GiB "
                  f"({cap:.1f} GiB max for this process); {free_gib:.2f} GiB free now")
        if free_gib == free_gib and free_gib < cap + 1.0:
            print(f"[amd_opt] NOTE: only {free_gib:.1f} GiB free on this GPU right now; "
                  f"{total_gib - free_gib:.1f} GiB is held by something else. Stop before "
                  f"this process needs more than {free_gib:.1f} GiB or the card overflows.")
    elif verbose:
        print('[amd_opt] no VRAM cap set for this process')

    # TF32 is an NVIDIA concept; on ROCm the useful knob is reduced-precision
    # accumulation inside fp16/bf16 GEMMs, which hipBLASLt honours.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if dtype in ('float16', 'bfloat16'):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    if strict_flash:
        _lock_in_flash(torch, verbose)

    if dtype == 'float32':
        fp32 = _FP32_TFLOPS.get(arch)
        bf16 = _PEAK_TFLOPS.get(arch)
        if fp32 and bf16:
            print(f"[amd_opt] WARNING dtype='float32' on {arch}: measured {fp32:.1f} TFLOP/s "
                  f"fp32 vs {bf16:.1f} TFLOP/s bf16 ({bf16/fp32:.1f}x slower). Use bfloat16.")
        else:
            print("[amd_opt] WARNING dtype='float32' on RDNA3 with this ROCm build: fp32 matmul "
                  "is a small fraction of bf16 (measured 26x slower on gfx1101, 3.3x on gfx1100).")

    if tunable:
        _enable_tunable(torch, dtype, verbose)

    return 'cuda'


# Practical bf16/fp16 GEMM throughput -- the number MFU should be divided by.
# gfx1101 (7800 XT): measured here, tools/matmul_probe.py, 47.2 TFLOP/s median of 9
#   interleaved rounds; 61 % of its 76.7 TFLOP/s published dual-rate rate.
# gfx1100 (7900 XTX): measured 76 TFLOP/s on this machine's card with the same probe;
#   62 % of the 122.8 TFLOP/s dual-rate figure TechPowerUp lists for it. That number is
#   96 CU x 2 (dual-rate fp16) at the 2.5 GHz boost clock: an upper bound, not an
#   achievable dense-GEMM rate.
# Re-measure after any ROCm upgrade; these move more than the hardware does.
_PEAK_TFLOPS = {'gfx1100': 76.0, 'gfx1101': 47.2, 'gfx1102': 41.9}
# Published dual-rate fp16 peaks, for quoting alongside a measured MFU if you want both.
# Derived as CU x 128 lanes x 2 (FMA) x 2 (dual-rate fp16) x boost clock:
#   gfx1100: 96  CU @ 2.484 GHz -> 122.8 ; gfx1101: 60 CU @ 2.495 GHz -> 76.7
# gfx1102: 32 CU @ 2.65 GHz -> 43.4. Measured/spec is 62 % and 61 % respectively on
# this ROCm build -- the gap is the stack, not the silicon.
_SPEC_TFLOPS = {'gfx1100': 122.8, 'gfx1101': 76.7, 'gfx1102': 43.4}

# Measured fp32 matmul throughput on the same probe, because the fp32 penalty is NOT
# consistent across these cards on this ROCm build:
#   gfx1100: ~30 % of bf16 (~23 TFLOP/s) -- roughly what a 61.4 TFLOP/s fp32 spec predicts
#   gfx1101: 1.81 TFLOP/s = 3.8 % of bf16 -- against a 38 TFLOP/s spec, i.e. that card's
#            fp32 GEMM path is broken in this build, not merely slower.
# Consequence: never train fp32 on gfx1101. On gfx1100 an fp32 op is merely 3.3x the cost
# of a bf16 one, which is why the fp32 upcast in cross_entropy is worth measuring rather
# than assuming it is a disaster. gfx1100 is your reported ~30% of bf16; gfx1102 is a
# derived guess (32 CU), not a measurement.
_FP32_TFLOPS = {'gfx1100': 23.0, 'gfx1101': 1.81, 'gfx1102': 10.9}


def recommended_peak_tflops():
    """Peak TFLOP/s to divide MFU by, so the number means something on AMD."""
    import torch
    try:
        props = torch.cuda.get_device_properties(0)
    except Exception:
        return 312.0
    arch = getattr(props, 'gcnArchName', '').split(':')[0]
    if arch in _PEAK_TFLOPS:
        return _PEAK_TFLOPS[arch]
    for needle, val in (('7900 XTX', 76.0), ('7800 XT', 47.2), ('7600', 41.9)):
        if needle in props.name:
            return val
    return 312.0  # A100, nanoGPT's original assumption


def _lock_in_flash(torch, verbose):
    """Make flash the only permitted SDPA backend, so a silent fallback is impossible."""
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from torch.nn.functional import scaled_dot_product_attention as sdpa

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)

    probe = torch.randn(2, 4, 256, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            out = sdpa(probe, probe, probe, is_causal=True)
            out.sum().backward()
        torch.cuda.synchronize()
        if verbose:
            print('[amd_opt] AOTriton flash attention: AVAILABLE (fused, no T*T materialisation)')
    except Exception as e:
        torch.backends.cuda.enable_math_sdp(True)
        print('[amd_opt] WARNING: flash attention is NOT available on this build.')
        print(f'[amd_opt]   -> {type(e).__name__}: {str(e)[:160]}')
        print('[amd_opt]   -> SDPA will fall back to the slow MATH backend. Set')
        print('[amd_opt]      TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 before launching.')


def _enable_tunable(torch, dtype, verbose):
    """ROCm TunableOp: autotunes hipBLASLt GEMM algo selection for the exact shapes used."""
    try:
        import torch.cuda.tunable as tun
        if not hasattr(tun, 'enable'):
            raise AttributeError('torch.cuda.tunable unavailable in this build')
        tun.enable(True)
        if hasattr(tun, 'set_rotating_buffer_size'):
            tun.set_rotating_buffer_size(40)
        tun.set_max_tuning_iterations(100)
        tun.set_filename(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      f'tunable_{dtype}.json'))
        if hasattr(tun, 'set_tuning_enable'):
            tun.set_tuning_enable(True)
        if verbose:
            print('[amd_opt] TunableOp enabled (first run pays a tuning cost, then '
                  'results are cached to the json file)')
    except Exception as e:
        print(f'[amd_opt] TunableOp could not be enabled: {e}')


# --- device identity: sysfs + amd-smi only, never a GPU allocation ---

def _parse_gib(value):
    """'' -> None (decide automatically); '8' -> 8.0; '0' -> 0.0 (explicitly uncapped)."""
    if value is None or str(value).strip() == '':
        return None
    try:
        return float(value)
    except ValueError:
        print(f"[amd_opt] ignoring unparseable NANOAMD_MAX_VRAM_GIB={value!r}")
        return None


def _vram_cap(bdf, connector, displays, requested, ceiling, override):
    """(GiB ceiling, reason) for this process; 0 means no cap.

    The ordering is the point, because the failure this module exists to prevent is
    amdgpu evicting VRAM into system RAM and pushing out the compositor:

      display            - the target GPU has a live connector: clamp to `ceiling`
                           (a smaller request is honoured, a larger one is refused below)
      display_clamped    - a request this rule refuses -- larger than `ceiling`, or 0
                           ("uncapped") on the GPU that draws the desktop -- ignored in
                           favour of `ceiling`
      display_override   - a larger request was honoured because
                           NANOAMD_ALLOW_DISPLAY_OVERRIDE=1 was set
      headless           - sysfs is readable and no connector is live on this GPU: the
                           card is not drawing anyone's desktop, so cap only if asked
      unknown            - sysfs could not be read, or some card on the box has a display
                           but this GPU's BDF could not be resolved (no amd-smi). We cannot
                           tell whether it is the display GPU, so assume the worst.

    The non-ROCm case never gets here: configure() returns before any of this.
    """
    if connector:
        # NANOAMD_MAX_VRAM_GIB=0 means "uncapped" (see _parse_gib). On the GPU that draws
        # the desktop that is not available without also setting ALLOW_DISPLAY_OVERRIDE:
        # an uncap that arrives by accident is exactly the failure this module exists to
        # stop, so here 0 is reported as a refused request rather than silently obeyed.
        if requested is None:
            return ceiling, 'display'
        if requested <= 0.0:
            return (0.0, 'display_override') if override else (ceiling, 'display_clamped')
        if requested < ceiling:
            return requested, 'display'
        return (requested, 'display_override') if override else (ceiling, 'display_clamped')
    if displays is None or (displays and bdf is None):
        return (requested if requested is not None else ceiling), 'unknown'
    return (requested if requested is not None else 0.0), 'headless'


def _visible_physical_index():
    """Physical ROCm index of the GPU torch is looking at."""
    visible = os.environ.get('HIP_VISIBLE_DEVICES') or os.environ.get('ROCR_VISIBLE_DEVICES')
    if visible:
        first = visible.split(',')[0].strip()
        if first.isdigit():
            return int(first)
    try:
        return int(_DEVICE)
    except ValueError:
        return 0


def _connected_display_outputs(drm_dir='/sys/class/drm'):
    """{PCI BDF: connector} for every GPU with a live display output.

    Returns None when the question cannot be answered -- either drm_dir is
    unreadable, or it lists no DRM cards at all (a container with only renderD* nodes).
    An empty dict means something else and is not the same thing: DRM cards are visible
    and none of them has a connector live, i.e. nobody is drawing a desktop.

    drm_dir is overridable so tools/selfcheck.py can build a fake sysfs tree.
    """
    out = {}
    try:
        entries = sorted(os.listdir(drm_dir))
    except OSError:
        return None
    if not any(e.startswith('card') and '-' not in e for e in entries):
        return None
    for entry in entries:
        # connector dirs look like cardN-DP-3; skip cardN, renderD*, controlD*, Writeback
        if not entry.startswith('card') or '-' not in entry:
            continue
        card, connector = entry.split('-', 1)
        if 'Writeback' in connector:
            continue
        try:
            with open(f'{drm_dir}/{entry}/status') as f:
                status = f.read().strip()
            bdf = os.path.basename(os.path.realpath(f'{drm_dir}/{card}/device'))
        except OSError:
            continue
        if status == 'connected':
            out[bdf] = connector
    return out


def _gpu_identity(physical_index):
    """(PCI BDF, market name) for a ROCm-indexed GPU, read from amd-smi text output."""
    exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv', 'bin', 'amd-smi')
    if not os.path.exists(exe):
        exe = 'amd-smi'
    bdf = name = None
    for flags in (['static', '-b'], ['static']):
        try:
            res = subprocess.run([exe] + flags, capture_output=True, text=True, timeout=15)
        except Exception:
            continue
        idx = None
        for line in res.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith('GPU:'):
                try:
                    idx = int(stripped.split(':', 1)[1])
                except ValueError:
                    idx = None
            elif idx == physical_index and stripped.startswith('BDF:'):
                bdf = stripped.split(':', 1)[1].strip()
            elif idx == physical_index and stripped.startswith('MARKET_NAME:'):
                name = stripped.split(':', 1)[1].strip()
    return bdf, name
