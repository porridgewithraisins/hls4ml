#ifndef NNET_CONV2DTRANSPOSE_STREAM_H_
#define NNET_CONV2DTRANSPOSE_STREAM_H_

#include "nnet_common.h"
#include "nnet_stream.h"
#include "nnet_types.h"

namespace nnet {

template <typename InputPipe, typename OutputPipe, typename CONFIG_T, bool UNROLL_BIAS_LOOP, bool UNROLL_FILTER_LOOP,
          bool UNROLL_WRITE_LOOP>
struct conv2dtranspose_stream_body {
    static void run(typename CONFIG_T::weight_t weights, typename CONFIG_T::bias_t biases) {
        auto input_buf = InputPipe::read();
        using output_t = typename ExtractPipeType<OutputPipe>::value_type;
        output_t output_buf;

        using output_elem_t = typename output_t::value_type;
        constexpr int pfc =
            (CONFIG_T::n_filt > CONFIG_T::parallelization_factor) ? CONFIG_T::parallelization_factor : CONFIG_T::n_filt;
        constexpr int filters_per_iter = (pfc > 0) ? pfc : 1;
        constexpr int max_filters = CONFIG_T::n_filt;

        for (int oh = 0; oh < CONFIG_T::out_height; oh++) {
            for (int ow = 0; ow < CONFIG_T::out_width; ow++) {
                output_elem_t acc[CONFIG_T::n_filt];

                if constexpr (UNROLL_BIAS_LOOP) {
#pragma unroll
                    for (int ff = 0; ff < CONFIG_T::n_filt; ff++) {
                        acc[ff] = static_cast<output_elem_t>(biases[ff]);
                    }
                } else {
                    for (int ff = 0; ff < CONFIG_T::n_filt; ff++) {
                        acc[ff] = static_cast<output_elem_t>(biases[ff]);
                    }
                }

                for (int fh = 0; fh < CONFIG_T::filt_height; fh++) {
                    int ih_raw = oh + CONFIG_T::pad_top - fh;
                    bool row_in_range = (ih_raw >= 0) && ((ih_raw % CONFIG_T::stride_height) == 0);
                    int ih = 0;
                    if (row_in_range) {
                        ih = ih_raw / CONFIG_T::stride_height;
                        row_in_range = ih < CONFIG_T::in_height;
                    }
                    if (!row_in_range) {
                        continue;
                    }

                    for (int fw = 0; fw < CONFIG_T::filt_width; fw++) {
                        int iw_raw = ow + CONFIG_T::pad_left - fw;
                        bool col_in_range = (iw_raw >= 0) && ((iw_raw % CONFIG_T::stride_width) == 0);
                        int iw = 0;
                        if (col_in_range) {
                            iw = iw_raw / CONFIG_T::stride_width;
                            col_in_range = iw < CONFIG_T::in_width;
                        }
                        if (!col_in_range) {
                            continue;
                        }

                        int in_base = (ih * CONFIG_T::in_width + iw) * CONFIG_T::n_chan;

                        [[intel::initiation_interval(CONFIG_T::reuse_factor)]]
                        for (int cc = 0; cc < CONFIG_T::n_chan; cc++) {
                            output_elem_t in_val = static_cast<output_elem_t>(input_buf[in_base + cc]);

                            for (int ff_base = 0; ff_base < max_filters; ff_base += filters_per_iter) {
                                if constexpr (UNROLL_FILTER_LOOP) {
#pragma unroll
                                    for (int lane = 0; lane < filters_per_iter; lane++) {
                                        int ff = ff_base + lane;
                                        if (ff >= max_filters) {
                                            continue;
                                        }
                                        int w_idx = ((((fh * CONFIG_T::filt_width) + fw) * CONFIG_T::n_filt) + ff) *
                                                        CONFIG_T::n_chan +
                                                    cc;
                                        acc[ff] += static_cast<output_elem_t>(in_val * weights[w_idx]);
                                    }
                                } else {
                                    for (int lane = 0; lane < filters_per_iter; lane++) {
                                        int ff = ff_base + lane;
                                        if (ff >= max_filters) {
                                            continue;
                                        }
                                        int w_idx = ((((fh * CONFIG_T::filt_width) + fw) * CONFIG_T::n_filt) + ff) *
                                                        CONFIG_T::n_chan +
                                                    cc;
                                        acc[ff] += static_cast<output_elem_t>(in_val * weights[w_idx]);
                                    }
                                }
                            }
                        }
                    }
                }

                if constexpr (UNROLL_WRITE_LOOP) {
#pragma unroll
                    for (int ff = 0; ff < CONFIG_T::n_filt; ff++) {
                        int out_idx = (oh * CONFIG_T::out_width + ow) * CONFIG_T::n_filt + ff;
                        output_buf[out_idx] = acc[ff];
                    }
                } else {
                    for (int ff = 0; ff < CONFIG_T::n_filt; ff++) {
                        int out_idx = (oh * CONFIG_T::out_width + ow) * CONFIG_T::n_filt + ff;
                        output_buf[out_idx] = acc[ff];
                    }
                }
            }
        }

        OutputPipe::write(output_buf);
    }
};

} // namespace nnet

#endif
