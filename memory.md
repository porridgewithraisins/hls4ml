# hls4ml Conv2DTranspose Implementation Journey

## What we are doing

Adding Conv2DTranspose support for hls4ml, initially from the pytorch frontend to the vivado backend. hls4ml basically takes your pytorch class, you define a converter that digs into the layers and converts it to a c++ template (literally python f strings). Then vivado_writer is able to write this to a c++ thing and then we can compile and run that in csim. We can of course also vivado that later on. For now, our dev loop is pytorch -> csim only though.

## Implementation Checklist & Progress

### ✅ Phase 1: Initial Setup & Prototype

- [x] Create basic layer class in `layers.py`
- [x] Add minimal PyTorch converter
- [x] Setup basic test file with small input
- [x] Verify PyTorch outputs for reference
- [x] Add dummy backend initialization
- [x] Generate initial HLS code structure

### ✅ Phase 2: Infrastructure Setup

- [x] Complete layer attributes in `Conv2DTranspose`
- [x] Implement proper PyTorch tensor conversion
- [x] Add to accumulation layers in fpga_backend
- [x] Create initial C++ templates
- [x] Setup proper weight handling

### ✅ Phase 3: Weight Precision & Debug

- [x] Debug empty weight files issue
- [x] Discover UnspecifiedPrecisionType problem
- [x] Add to infer_precision pass
- [x] Verify weight file generation
- [x] Confirm proper weight values

### ✅ Phase 4: End-to-End Pipeline

- [x] Complete C++ compilation pipeline
- [x] Add proper include paths
- [x] Generate test data
- [x] Run C++ testbench
- [x] Verify non-zero outputs

### ✅ Phase 5: Correctness

- [x] Implement correct Conv2DTranspose algorithm
- [x] Match PyTorch's behavior (multi-channel, padding)
- [x] Validate against PyTorch reference for large randomized cases
- [ ] Optimize for HLS synthesis (throughput, resource)
- [ ] Add performance optimizations

## Development Workflow

### 1. Prototype Phase

```bash
# Initial minimal implementation
1. Created Conv2DTranspose class with basic attributes
2. Added simple PyTorch converter
3. Created test file with small example
4. Ran test to verify PyTorch reference output
```

### 2. Infrastructure Phase

```bash
# Building proper infrastructure
1. Expanded layer class with full attributes
2. Added proper tensor handling
3. Setup backend initialization
4. Created template structure
```

### 3. Debug & Fix Phase

```bash
# When we hit the weight precision issue
1. Added debug logging to WeightVariable.update_precision()
2. Created comparison test with Conv2D
3. Identified optimizer pass issue
4. Fixed by adding to infer_precision
```

### 4. End-to-End Testing

```bash
# Getting full pipeline working
1. Generate HLS project
2. Compile C++ code
3. Run testbench
4. Compare outputs
```

## Development Process Learned

1. **Incremental Development**

   - Start with minimal working example
   - Add complexity gradually
   - Test each component individually

2. **Pattern Matching**

   - Follow existing layer patterns (Conv2D)
   - Match data formats (channels_first)
   - Reuse existing infrastructure

3. **Debug Strategy**

   - Compare with working layers
   - Add strategic debug logging
   - Use comparison tests
   - Follow data through pipeline

4. **Testing Process**
   ```
   Python Test → HLS Generation → Weight Files → C++ Compilation → Runtime Test
   ```

## Project Structure & Key Files

### Core Implementation Files

- `hls4ml/model/layers.py`: Layer class definitions

  - Added `Conv2DTranspose` class with `_expected_attributes`
  - Key attributes: weights, bias, input/output shapes
  - Changed to channels_first format to match Conv2D

- `hls4ml/converters/pytorch/conv2dtranspose.py`: PyTorch converter

  - Handles conversion from PyTorch's Conv2DTranspose
  - Sets weight_data and bias_data from PyTorch tensors
  - Matches Conv2D's channels_first pattern

- `hls4ml/backends/vivado/vivado_backend.py`: Backend initialization

  - Added `init_conv2dtranspose()` method
  - Sets strategy, partitions, implementation
  - Doesn't set weight types (handled by optimizer)

- `hls4ml/backends/vivado/passes/convolution_templates.py`: HLS C++ templates
  - Defines template for Conv2DTranspose implementation
  - Currently has naive implementation
  - Removed problematic conv_implementation line from config

### Support Files

- `hls4ml/backends/fpga/fpga_backend.py`: Added Conv2DTranspose to accum_layers
- `hls4ml/model/optimizer/passes/infer_precision.py`: Weight precision inference
  - Added Conv2DTranspose to `_infer_precision` method
  - Uses same logic as Conv2D for precision inference

## Development Process Discovered

1. **Layer Definition**

   - Add layer class in `layers.py`
   - Define expected attributes
   - Match existing patterns (e.g., Conv2D for convolution layers)

2. **PyTorch Converter**

   - Create converter in `converters/pytorch/`
   - Handle tensor conversion and reshaping
   - Ensure weight format matches backend expectations

3. **Backend Integration**

   - Add initialization in `vivado_backend.py`
   - Add to accumulation layers if needed
   - Create HLS C++ templates

4. **Weight Precision**

   - Add to `infer_precision.py` optimizer pass
   - Match similar layer types (Conv2D in this case)

5. **Testing Flow**

   ```python
   # First test with prototype/dummy implementation
   python test_pytorch_conv2dtranspose.py
   # Check PyTorch reference output
   # Check HLS code generation
   # Check weight file generation
   ```

6. **C++ Compilation & Testing**
   ```bash
   cd hls4ml_prj
   # Compile C++ testbench
   g++ -o myproject_test myproject_test.cpp firmware/myproject.cpp -Ifirmware/ap_types/ -Ifirmware -std=c++11 -D WEIGHTS_DIR='"firmware/weights"'
   # Run tests
   ./myproject_test
   ```

## Project Structure Generated

After running conversion, an `hls4ml_prj/` directory is created with:

```
hls4ml_prj/
├── build_lib.sh
├── build_prj.tcl
├── firmware/
│   ├── ap_types/
│   ├── defines.h
│   ├── myproject.cpp       # Main implementation
│   ├── myproject.h
│   ├── nnet_utils/
│   ├── parameters.h
│   └── weights/           # Generated weight files
├── myproject_bridge.cpp
├── myproject_test         # Compiled testbench
├── myproject_test.cpp     # Testbench source
├── project.tcl
├── tb_data/              # Test data & results
└── vivado_synth.tcl
```

In oneAPI case, it is

hls4ml_prj
├── build
│   ├── myproject.fpga_emu # csim executable
│   └── myproject.report.prj
│       ├── board_spec.xml
│       ├── include
│       ├── ip
│       ├── ip_include.tcl
│       ├── ipinterfaces.xml
│       ├── kernel_hdl
│       ├── kernel_list_file.txt
│       ├── kernel_system_import.tcl
│       ├── kernel_system.qip
│       ├── kernel_system.sv
│       ├── kernel_system.tcl
│       ├── logs
│       ├── Myproject_interface_structs.sv
│       ├── myproject_report.bc.xml
│       ├── myproject_report_di_hw.tcl
│       ├── myproject_report_di_inst.sv
│       ├── myproject_report_di.sv
│       ├── myproject_report_sys_hw.tcl
│       ├── myproject_report_sys.sv
│       ├── opencl.ipx
│       ├── reports # contains reports
│       ├── sim
│       └── sys_description.txt
├── src
│   ├── exception_handler.hpp
│   ├── firmware
│   │   ├── defines.h
│   │   ├── myproject.cpp # implementation
│   │   ├── myproject.h
│   │   ├── nnet_utils
│   │   ├── parameters.h
│   │   └── weights # contains the weights
│   ├── myproject_bridge.cpp
│   └── myproject_test.cpp # test bench for csim
└── tb_data
    ├── results.log
    ├── tb_input_features.dat
    └── tb_output_predictions.dat

28 directories, 43 files


## Key Discoveries & Solutions

1. **Weight Precision Issue**

   - Problem: Weights had UnspecifiedPrecisionType, causing empty files
   - Root cause: Conv2DTranspose not recognized by infer_precision_types pass
   - Solution: Added to \_infer_precision method list
   - Verification: Weight files now contain actual values

2. **Data Format**

   - Changed from channels_last to channels_first
   - Matches Conv2D pattern for consistency
   - Required for proper HLS implementation

3. **Pipeline Verification**
   - PyTorch output shape matches expectations: (1, 3, 4, 4)
   - HLS code generates successfully
   - C++ testbench compiles and runs
   - Outputs now track PyTorch within fixed-point tolerance once the redundant input transpose is removed

## Current Status

✅ Complete:

- Layer definition & attributes
- PyTorch converter
- Backend initialization
- Template generation
- Weight precision inference
- C++ compilation pipeline
- Correct multi-channel Conv2DTranspose behavior (matches PyTorch within quantization error)

🔄 In Progress:

- Throughput/resource optimization for the finalized kernel
- Evaluating precision/performance trade-offs for deployment

## Next Steps

We now have the Intel oneAPI Base Toolkit on PATH for this repo (via direnv) and the FPGA Add-On installed, so the oneAPI backend builds are unblocked. Recent progress:

> Note: How to build
> mkdir build && cd build && cmake -DUSER_INCLUDE_PATHS=/opt/intel/oneapi/compiler/2025.0/opt/oclfpga/include/ ..

- ✅ `make fpga_emu` runs cleanly; emulator outputs match PyTorch (`MAE≈5.1e-3`, `MSE≈3.9e-5`, `MaxAbs≈1.9e-2`).
- ✅ `make report` succeeds; baseline Agilex7 estimates show ~326k ALUTs (34%), ~423k FFs (22%), 0 DSPs, and 3.8k MLABs. Inner accumulation loop limits the II to 3 because of 11.5k-bit shifts. The reports live in the build directory's myproject.report.prj/reports directory. In there you can read either the report.html, or the structured json in resources/json/*. Particularly summary.ndjson.

Outstanding work:

1. Iterate on optimization knobs (reuse, IOType, precision) using the oneAPI project as the baseline while keeping the parity tests in place.

## Data Layout Lessons

- **PyTorch staging (`test_conv2dtranspose_simple.py`, `test_pytorch_conv2dtranspose.py`)** writes reference tensors in NCHW but explicitly saves testbench data as NHWC before invoking the converter; that keeps file order consistent with the generated `ap_fixed` arrays.
- **Optimizer insertion (`hls4ml/model/optimizer/passes/convert_to_channels_last.py`)** previously added a `Transpose` node (`transpose_input_for_x`), yielding `config3_perm = {1,2,0}` in `hls4ml_prj/firmware/parameters.h` and an `nnet::transpose` call in `hls4ml_prj/firmware/myproject.cpp`. With NHWC data already on disk, this extra hop double-permuted channels and caused the 19/29/14/25 mismatch.
- **Template expectations (`hls4ml/backends/vivado/passes/convolution_templates.py`)** assume channels-last indexing: weights flatten as `[fh][fw][ff][cc]` (see generated `hls4ml_prj/firmware/weights/w2.h`), and the loop body accesses inputs via `(ih * in_width + iw) * n_chan + cc`.
- **Fix applied**: set `config['Model']['ChannelsLastConversion'] = 'internal'` in both test drivers so the optimizer keeps internal tensors in NHWC without inserting front-end transposes; regenerated `myproject.cpp` now streams `x` directly into the convolution loops, matching PyTorch within quantization noise.
- **Takeaway**: align disk layouts, optimizer settings, and template indexing. When channels-last data is supplied externally, disable redundant transposes and keep an eye on `parameters.h` for unexpected permutation configs.

## Optimization

- **I/O topology (docs/api/concepts.rst, docs/api/configuration.rst):** Choose between `io_parallel` for minimum latency and `io_stream` when the design needs FIFO-based DATAFLOW decoupling. For streaming, enable the Vivado FIFO-depth optimizer (`config['Flows'] = ['vivado:fifo_depth_optimization']`) to shrink buffers after co-sim.
- **Reuse & strategy knobs (`config_from_pytorch_model`, `hls4ml_prj/firmware/parameters.h`):** Lower `ReuseFactor` and `Strategy='Latency'` give peak throughput, while higher reuse or `Strategy='Resource'` shares multipliers. `distributed_arithmetic` is available (docs/advanced/da.rst) when reuse must be 1 and DSPs are scarce.
- **Pipeline style:** Align `Model.PipelineStyle` (`auto`/`pipeline`/`dataflow`) with the chosen IO; DATAFLOW plus `io_stream` keeps our Conv2DTranspose busy without global control, while `pipeline` can target a specific II if reuse choices permit.
- **Precision management (docs/advanced/precision.rst):** Default `ap_fixed<16,6>` works, but we can widen `model_default_t`/`result_t` via `HLSConfig['Model']['Precision']` or layer overrides in `config['LayerName'][...]`. For bit-exact flows, enable the `bit_exact` optimizer; otherwise, manual mixed precision keeps resource usage low while respecting the ≈0.019 error budget.
- **Higher-level model pruning (docs/advanced/model_optimization.rst):** hls4ml’s Optimization API can reduce filter counts or enforce DSP budgets before conversion—useful if the final mini U-Net needs to fit a tight part once Conv2DTranspose is present.
- **Verification harness:** Maintain `test_pytorch_conv2dtranspose.py` as a regression across knob combinations (reuse sweep, precision variants, io_parallel vs io_stream) so any template tweak still matches PyTorch within quantization tolerance.

We will be using the oneAPI backend exclusively going forward.

### 2025-11-03 – oneAPI Conv2DTranspose accumulator fix

- Converted the Conv2DTranspose template to accumulate per output pixel using a small local buffer; this removed the 11.5k-bit scatter shift register and cut Agilex7 ALUT usage from ~206k to ~64.6k while lowering FF count to ~167k.
- Ensured all MAC paths cast to the layer’s `output_elem_t` so the emulator MAE returned to ≈5.1e-3 (matching the pre-optimization baseline) with MSE ≈3.9e-5 and MaxAbs ≈1.9e-2.
- Verified the optimized kernel through `./iterate.sh --agent`; both fpga_emu and report builds succeed with the improved resource profile.
- Next up: surface reuse/parallelization/IO knobs in the template and add regression sweeps so users can choose between latency and area trade-offs without regressing accuracy.

## TODO: Next optimization

- Evaluate remaining resource gap versus Conv2D by inspecting the output scatter loop. Current design keeps `parallelization_factor` MAC lanes alive at once, so each output pixel holds a `n_filt`-wide accumulator. Explore time-multiplexing filters (higher reuse factor) to shrink the local accumulator array or serialize writes when users request `Strategy='Resource'`.
- Mirror Conv2D’s knob handling: honor `ReuseFactor`, `Strategy`, `io_type`, and `implementation` in the config generator, then gate pragmas accordingly (e.g. skip `#pragma unroll pfc` when reuse > 1, emit DATAFLOW wrappers for `io_stream`).
- Build a regression sweep in `test_pytorch_conv2dtranspose.py` that iterates over representative knob combinations, regenerates the project, and asserts MAE stays within ±1e-2 while reporting resource deltas.

### 2025-11-03 – oneAPI Conv2DTranspose knob plumbing

- Registered Conv2DTranspose with the oneAPI backend’s reusable attributes so `Strategy`, `ReuseFactor`, and `ParallelizationFactor` flow straight from the user config.
- Refactored the generated kernel to respect those knobs: the bias/init/write loops only unroll when either latency mode or multi-filter parallelization is requested, and the MAC nesting now chunks filters according to the configured parallelization factor while keeping reuse-driven II hints intact.
- Added `io_type` metadata to the conv2dtranspose config struct; `io_parallel` remains the default but `io_stream` is now propagated for future streaming support.
- Verified via `./iterate.sh --agent` that the default build still lands at MAE≈5.09e-3 with identical resource estimates (~68.5k ALUTs / ~175k FFs) so baseline behavior is unchanged pending knob sweeps.

## oneAPI Backend Build Flow

See iterate.sh in hls4ml root folder

## Phase 5 Progress

- **Scope**: Validate multi-channel Conv2DTranspose correctness end-to-end and document the layout pipeline.
- **Method**: Exercised both `test_conv2dtranspose_simple.py` (deterministic two-channel case) and `test_pytorch_conv2dtranspose.py` (7→5 channels with random weights) after each template tweak, rebuilding `hls4ml_prj` and comparing `Predictions` vs. `Quantized predictions` from `./myproject_test`.
- **Findings**:
   - The framework inserted `transpose_input_for_*` nodes because `ChannelsLastConversion` defaulted to `'full'`, so NHWC testbench data was transposed twice before hitting the convolution loops.
   - Setting `config['Model']['ChannelsLastConversion'] = 'internal'` removed the redundant transpose while keeping internal tensors channels-last; regenerated `myproject.cpp` now mirrors PyTorch indexing.
   - After the change, MAE between PyTorch outputs and HLS quantized results is ≈5e-3 for the large random case, with max error ≈1.9e-2—fully attributable to fixed-point quantization.
- **Result**: Functional parity is achieved; remaining work is performance tuning and precision exploration.

## Phase 6: Template Optimization

### Initial Baseline-to-Optimized Iteration

After establishing functional correctness in Phase 5, we turned to reducing resource usage while maintaining the same throughput characteristics. The oneAPI backend synthesis report for our baseline sequential implementation showed significant FPGA resource consumption (Agilex7, 5ns clock target):

**Baseline metrics (sequential loops, no pragmas):**
- ALUTs: 291,565 (49.6% of available)
- Flip-flops: 333,997 (36.5%)
- Initiation Interval (II): 3 cycles on innermost loop
- Bottleneck: 11,520-bit accumulator shifts dominating area

**Challenge:** Aggressive full-unroll pragmas on nested loops caused design explosion—the report linker hung due to creating thousands of parallel MAC units for the 7-channel × 3×3 filter × 5-output configuration.

**Solution:** Applied selective optimizations inspired by the existing Conv2D im2col implementation (`nnet_conv2d_resource.h`), which uses controlled unrolling via `parallelization_factor`:

1. **Full unroll on bias init loop**: Small bounded iteration (180 elements), safe to unroll completely for initialization.
2. **Computed parallelization factor**: `pfc = MIN(n_filt, parallelization_factor)` to limit filter-loop unrolling; with default `parallelization_factor=1` and `n_filt=5`, only one filter computed per iteration (no explosion).
3. **Initiation interval constraint**: Added `[[intel::initiation_interval(reuse_factor)]]` pragma on the filter width loop to respect throughput requirements without over-pipelining.
4. **Strategic register placement**: Applied `[[intel::fpga_register]]` attribute only to the input value register (`in_val`) to reduce repeated memory accesses, avoiding wholesale register insertion that inflated state.

**Optimized metrics:**
- ALUTs: 206,344 (39.6%, **29% reduction**)
- Flip-flops: 303,975 (26.6%, **9% reduction**)
- II: 3 cycles (unchanged, matching reuse_factor constraint)
- Key wins:
  - `layer2_out` barrel shifter: 46k FFs → 23k FFs (50% reduction from input register reuse)
  - Private memory copies: 10 → 9 (reduced loop concurrency)

**Validation:** Emulator still passes with MAE=5.09e-3, matching Phase 5 functional correctness. Synthesis report generation completes quickly without linker hangs, confirming the controlled unrolling strategy scales.

**Takeaway:** For Conv2DTranspose and similar accumulation-heavy layers, selective pragma application guided by `parallelization_factor` and `reuse_factor` achieves significant resource savings without sacrificing throughput or correctness. Full unrolling should be reserved for provably small loops (e.g., bias init), while nested convolution loops require bounded unroll factors to avoid combinatorial explosion.

### Next Steps: User-Configurable Optimization Strategies

Now that we have a working optimization baseline, we need to design a system where users can tune Conv2DTranspose behavior through hls4ml's configuration knobs. Key questions to address:

1. **Knob exposure**: How do users set `ReuseFactor`, `Strategy`, `parallelization_factor`, etc. in the Python test scripts? What's the API surface (`config_from_pytorch_model`, `HLSConfig`, layer-specific overrides)?

2. **Strategy selection**: Should we generate different template variants based on knob combinations? For example:
   - `Strategy='Latency'` + low `ReuseFactor` → aggressive unrolling (if safe)
   - `Strategy='Resource'` + high `ReuseFactor` → time-multiplexed MACs
   - How does `parallelization_factor` interact with these?

3. **Available knobs inventory**: What configuration parameters are relevant for Conv2DTranspose?
   - IO style (`io_parallel` vs `io_stream`)
   - Precision overrides (`model_default_t`, layer-specific types)
   - Pipeline/DATAFLOW directives
   - Reuse and parallelization factors
   - Any oneAPI-specific attributes?

4. **Template generation logic**: Do we need conditional code generation in `convolution_templates.py` that inspects the config and emits different pragma combinations? Or can we parameterize a single template effectively?

5. **Testing matrix**: How do we validate that knob combinations don't break correctness or cause synthesis failures? Extend `test_pytorch_conv2dtranspose.py` with a parameter sweep?

Before implementing further optimizations, we should map out the configuration space and design a coherent strategy for template generation that respects user intent while maintaining the optimization gains achieved so far.

## Knob Exploration

- **Baseline (`io_parallel`, `ReuseFactor=1`, `ParallelizationFactor=1`)**: Reference configuration stays at ALUT≈95k, FF≈234k, DSP=1, MLAB≈5.9k with MAE ≈5.09e-3.
- **Reuse sweep (`ReuseFactor=5`)**: Valid factor but raised control/state cost (ALUT≈96k, FF≈283k) while leaving accuracy unchanged; additional feedback and barrel shifter depth are the main culprits.
- **IO topology check (`io_stream`, `ReuseFactor=1`)**: After wiring `Model.IOType`, streaming mode compiles cleanly yet keeps resource totals aligned with baseline, suggesting pipes dominate regardless of interface.
- **Parallelization lane bump (`ParallelizationFactor=2`)**: Best improvement so far—ALUT≈71.6k and FF≈175.8k with DSP usage at 2, preserving MAE; unrolling two filter lanes balances logic sharing vs. throughput.

### Future Sweeps

1. Combine `io_stream` with `ParallelizationFactor=2` to see if decoupled I/O shaves pipe control overhead.
2. Explore higher filter parallelization (3–5 lanes) while monitoring DSP pressure and register growth.
3. Revisit reuse factors ≥5 alongside the parallel lanes to trade MAC duplication for tighter control logic.
4. Evaluate precision downgrades (e.g., ap_fixed<14,4>) once structural knobs settle, keeping MAE within tolerance.
