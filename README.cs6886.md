# CS6886 Project — Conv2DTranspose Support in hls4ml

This document explains how to reproduce the Conv2DTranspose experiments, run the automated conv2dtranspose sweeps, and execute the Mini U-Net pipeline in both synthetic and trained-weight modes. All instructions assume an Ubuntu 22.04 (or newer) host with an Intel-compatible GPU/FPGA development environment.

## 1. Environment Setup

I ran everything on Python 3.13.5, torch==2.9.0+cpu and oneAPI DPC++/C++ Compiler 2025.0.4 (2025.0.4.20241205)

1. **Create and activate a Python virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Clone and install the repo**

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   git clone https://github.com/porridgewithraisins/hls4ml.git
   cd hls4ml
   pip install -e .
   pip install Pillow
   ```

3. **Install Intel oneAPI Base Toolkit (Ubuntu)**

   ```bash
   sudo apt update
   sudo apt install intel-basekit
   ```

4. **Install Intel FPGA Add-on for oneAPI (Ubuntu)**

   ```bash
   wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/ff129224-aa19-48f7-96d4-ad12d2d427f9/intel-fpga-support-for-compiler-2025.0.0.591_offline.sh
   bash intel-fpga-support-for-compiler-2025.0.0.591_offline.sh
   ```

   Follow the graphical prompts and accept the defaults.

5. **Source oneAPI environment variables (for every new shell)**

   ```bash
   source /opt/intel/oneapi/setvars.sh
   ```

   It is idempotent, don't worry about running it twice.

## 2. Conv2DTranspose Regression Sweep

Run the regression harness manually to regenerate the oneAPI project, launch the emulator, and produce synthesis reports.

1. **Execute the PyTorch regression script**

   ```bash
   python3 test_pytorch_conv2dtranspose.py --strategy resource --reuse-factor 1 --parallelization-factor 1
   ```

   Adjust the flags (`--strategy`, `--reuse-factor`, `--parallelization-factor`) to explore different configurations.

2. **Configure and build the project**

   ```bash
   mkdir -p hls4ml_prj/build
   cd hls4ml_prj/build
   cmake -DUSER_INCLUDE_PATHS=/opt/intel/oneapi/compiler/latest/linux/include/ ..
   make fpga_emu
   cd ..
   ```

3. **Run the emulator and post-processing**

   ```bash
   ./build/myproject.fpga_emu
   python3 ../sanitycheck.py
   cd build
   rm -f myproject.report
   make report
   cd ..
   ```

4. **Review outputs**

   - Emulator results: `hls4ml_prj/tb_data/results.log`
   - Reports: `hls4ml_prj/build/myproject.report.prj/reports/report.html`
   - Resource JSON: `hls4ml_prj/build/myproject.report.prj/reports/resources/json`

## 3. Mini U-Net Pipeline (Random Initialization)

This mode uses randomly initialized weights and is useful for sanity checks and knob sweeps.

1. **Generate the HLS project and run the emulator**

   ```bash
   python3 test_unet.py --precision ap_fixed<18,8> \
   	 --strategy resource --reuse-factor 1 --parallelization-factor 1 \
   	 --io-type io_parallel --output-dir hls4ml_prj_unet
   ```

   This command writes NHWC `.dat` files to `hls4ml_prj_unet/tb_data` and emits a oneAPI project in `hls4ml_prj_unet/`.

2. **Build and launch the emulator**

   ```bash
   mkdir -p hls4ml_prj_unet/build
   cd hls4ml_prj_unet/build
   cmake -DUSER_INCLUDE_PATHS=/opt/intel/oneapi/compiler/latest/linux/include/ ..
   make fpga_emu report
   ./myproject.fpga_emu
   cd ../..
   ```

3. **Inspect reports**
   ```
   xdg-open hls4ml_prj_unet/build/myproject.report.prj/reports/report.html
   ```

## 4. Mini U-Net Pipeline (Trained Weights)

1. **Train the Mini U-Net**

   ```bash
   python3 train_mini_unet.py --epochs 20 --batch-size 16 --output checkpoints/mini_unet.pt
   ```

   Adjust epochs and dataset paths as needed.

2. **Regenerate the HLS project with trained weights**

   ```bash
   python3 test_unet.py --load checkpoints/mini_unet.pt --precision ap_fixed<18,8> \
   	 --strategy resource --reuse-factor 1 --parallelization-factor 1 \
   	 --io-type io_parallel --output-dir hls4ml_prj_unet
   ```

3. **Create PyTorch reference overlays and logits**

   ```bash
   mkdir -p unet-results/torch
   python3 generate_unet_torch_outputs.py --checkpoint checkpoints/mini_unet.pt \
   	 --image-dir path/to/sample_images --output-dir unet-results/torch
   ```

4. **Convert sample images into HLS testbench format**

   ```bash
   python3 write_unet_tb_from_images.py --image-dir path/to/sample_images \
   	 --output-dir hls4ml_prj_unet/tb_data
   ```

5. **Rebuild and run the emulator**

   ```bash
   cd hls4ml_prj_unet/build
   cmake -DUSER_INCLUDE_PATHS=/opt/intel/oneapi/compiler/latest/linux/include/ ..
   make fpga_emu
   ./myproject.fpga_emu
   cd ../..
   ```

6. **Parse HLS outputs and generate overlays/metrics**

   ```bash
   mkdir -p unet-results/hls
   python3 generate_unet_hls_outputs.py --results-log hls4ml_prj_unet/tb_data/results.log \
   	 --output-dir unet-results/hls
   ```

   This step recreates overlays, masks, and MAE/MaxAbs summaries for direct comparison with the PyTorch references.

## 5. Notes and Troubleshooting

- Always activate the virtual environment and source oneAPI variables before running scripts.
- The trained-weight flow currently exhibits mismatches due to outstanding BatchNorm folding issues in the larger Mini U-Net; emulator outputs are still useful for qualitative inspection.
- Adjust `--strategy`, `--reuse-factor`, `--parallelization-factor`, and `--precision` flags to explore resource/latency trade-offs.
- Clean old builds with `rm -rf hls4ml_prj` or `rm -rf hls4ml_prj_unet` if you encounter stale artifacts before regenerating projects.

With these steps, you can repeat the conv2dtranspose regressions, inspect resource and latency trade-offs, and evaluate the Mini U-Net pipeline in both synthetic and trained modes.
