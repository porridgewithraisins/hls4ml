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

1. Profile resource usage and latency in `hls4ml_prj/firmware/myproject.cpp`; explore loop tiling/unrolling tweaks.
2. Evaluate wider fixed-point or mixed precision via `HLSConfig['Model']['Precision']` overrides before synthesis.
3. Add regression tests that cover multi-channel stress cases (`test_pytorch_conv2dtranspose.py`) to guard against layout regressions.

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

## Testing Commands

```bash
# Run Python test (generates HLS project)
python test_pytorch_conv2dtranspose.py

# Compile C++ testbench
cd hls4ml_prj
g++ -o myproject_test myproject_test.cpp firmware/myproject.cpp -Ifirmware/ap_types/ -Ifirmware -std=c++11 -D WEIGHTS_DIR='"firmware/weights"'

# Run C++ testbench
./myproject_test
```

## Phase 5 Progress

- **Scope**: Validate multi-channel Conv2DTranspose correctness end-to-end and document the layout pipeline.
- **Method**: Exercised both `test_conv2dtranspose_simple.py` (deterministic two-channel case) and `test_pytorch_conv2dtranspose.py` (7→5 channels with random weights) after each template tweak, rebuilding `hls4ml_prj` and comparing `Predictions` vs. `Quantized predictions` from `./myproject_test`.
- **Findings**:
   - The framework inserted `transpose_input_for_*` nodes because `ChannelsLastConversion` defaulted to `'full'`, so NHWC testbench data was transposed twice before hitting the convolution loops.
   - Setting `config['Model']['ChannelsLastConversion'] = 'internal'` removed the redundant transpose while keeping internal tensors channels-last; regenerated `myproject.cpp` now mirrors PyTorch indexing.
   - After the change, MAE between PyTorch outputs and HLS quantized results is ≈5e-3 for the large random case, with max error ≈1.9e-2—fully attributable to fixed-point quantization.
- **Result**: Functional parity is achieved; remaining work is performance tuning and precision exploration.
