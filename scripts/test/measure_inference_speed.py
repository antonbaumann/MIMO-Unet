from argparse import ArgumentParser
from models.ensemble import EnsembleModule
import torch
import numpy as np

def main(
    model_checkpoint_paths: str,
    monte_carlo_steps: int = 0,
    device: str = "cuda",
):
    model = EnsembleModule(
        checkpoint_paths=model_checkpoint_paths,
        monte_carlo_steps=monte_carlo_steps,
    )
    model.to(device)

    dummy_input = torch.randn(1, 3, 128, 160, dtype=torch.float).to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1000
    timings=np.zeros((repetitions,1))

    #GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)

    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.mean(timings)
    std_syn = np.std(timings)
    print(f"Mean inference time: {mean_syn} ms")
    print(f"Std inference time: {std_syn} ms")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint_paths", nargs="+", type=str, required=True)
    parser.add_argument("--monte_carlo_steps", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    main(
        model_checkpoint_paths=args.model_checkpoint_paths,
        monte_carlo_steps=args.monte_carlo_steps,
        device=args.device,
    )