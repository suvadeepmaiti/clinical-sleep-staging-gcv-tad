"""Model complexity and CPU latency of the teacher and the students (writes efficiency.json, tables/efficiency.tex).

Params: trainable parameters. FP32 size: params x 4 bytes. INT8: serialized state_dict after PyTorch dynamic
quantization (Linear/LSTM/GRU weights int8, convolutions stay FP32). MACs: torch FlopCounterMode / 2.
Latency: median wall time on CPU, batch 1, one thread, per newly scored 30-s epoch:
  student  one epoch + GRU state (causal streaming);
  teacher  encoder on the new epoch + BiLSTM head over the cached 21-epoch window (embeddings of the other 20
           epochs reused); the centre epoch is scored 10 epochs (5 min) after it is recorded.
Weights are random (pretrained=False): size and speed do not depend on the values.
"""
import io
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
from teachers_multi import SleepGPTMultiTeacher  # noqa: E402
from models import SleepStudentNet  # noqa: E402

WINDOW = 21


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def serialized(m):
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    return buf.tell()


def macs(fn):
    # FlopCounterMode undercounts under inference_mode, so count under no_grad.
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        fn()
    return counter.get_total_flops() / 2


@torch.inference_mode()
def latency_ms(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        times.append(1e3 * (time.perf_counter() - t))
    return statistics.median(times)


def student_row(name, channels):
    m = SleepStudentNet(n_channels=channels).eval()
    x, h = torch.randn(1, channels, 3000), torch.zeros(1, 1, 64)
    q = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear, torch.nn.GRU}, dtype=torch.qint8)
    return {"model": name, "params": n_params(m), "fp32_bytes": 4 * n_params(m), "int8_bytes": serialized(q),
            "macs": macs(lambda: m(x, h)), "latency_ms": latency_ms(lambda: m(x, h), 200, 2000),
            "latency_int8_ms": latency_ms(lambda: q(x, h), 200, 2000)}


def teacher_row():
    m = SleepGPTMultiTeacher(pretrained=False).eval()
    x, e = torch.randn(1, 4, 3000), torch.randn(1, WINDOW, m.context.lstm.input_size)
    step = lambda: (m.embed(x), m.forward_embeddings(e))  # noqa: E731
    # Attention reads qkv.weight directly, so those layers stay FP32.
    spec = {n for n, mod in m.named_modules() if isinstance(mod, (torch.nn.Linear, torch.nn.LSTM)) and not n.endswith("qkv")}
    q = torch.ao.quantization.quantize_dynamic(m, spec, dtype=torch.qint8)
    qstep = lambda: (q.embed(x), q.forward_embeddings(e))  # noqa: E731
    return {"model": "teacher", "params": n_params(m), "fp32_bytes": 4 * n_params(m), "int8_bytes": serialized(q),
            "macs": macs(step), "latency_ms": latency_ms(step, 5, 50), "latency_int8_ms": latency_ms(qstep, 5, 50)}


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    rows = [teacher_row(), student_row("student_1ch", 1), student_row("student_2ch", 2)]
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor())
    meta = {"cpu": cpu, "threads": 1, "torch": torch.__version__}
    (repo_paths.ANALYSIS / "efficiency.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))

    t = rows[0]
    names = {"teacher": "Teacher", "student_1ch": "Student, 1 ch.", "student_2ch": "Student, 2 ch."}
    count = lambda n: f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n:,}"  # noqa: E731
    macs_str = lambda n: f"{n / 1e9:.1f}G" if n >= 1e9 else f"{n / 1e6:.1f}M"  # noqa: E731
    size = lambda b: f"{b / 1e6:.1f} MB" if b >= 1e6 else f"{b / 1e3:.1f} KB"  # noqa: E731
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Model complexity and CPU cost per 30-s epoch (batch 1, one thread of an " + cpu.replace("(R)", "")
             + r", PyTorch " + torch.__version__.split("+")[0] + r"). FP32: parameters $\times$ 4 bytes; INT8: dynamic "
             r"quantization of linear and recurrent layers (serialized). Teacher latency: encoder on the new epoch plus BiLSTM over "
             r"the cached 21-epoch window; it scores an epoch 10 epochs (5 min) later, the student immediately.}",
             r"\label{tab:efficiency}", r"\footnotesize", r"\setlength{\tabcolsep}{2pt}",
             r"\begin{tabular}{lrrrrrr}", r"\hline",
             r"Model & Params & FP32 & INT8 & Ratio & MACs & Latency \\", r"\hline"]
    for r in rows:
        lines.append(rf"{names[r['model']]} & {count(r['params'])} & {size(r['fp32_bytes'])} & {size(r['int8_bytes'])} & "
                     rf"{t['params'] / r['params']:,.0f}$\times$ & {macs_str(r['macs'])} & {r['latency_ms']:.2f} ms \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (repo_paths.REPO / "paper" / "tables" / "efficiency.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps({"meta": meta, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
