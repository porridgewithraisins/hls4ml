#ifndef NNET_CONV2DTRANSPOSE_H_
#define NNET_CONV2DTRANSPOSE_H_

#include "nnet_common.h"
#include "nnet_conv2d.h"

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T, bool UNROLL_BIAS, bool UNROLL_FILTERS, bool UNROLL_WRITES>
void conv_2dtranspose_cl(const data_T &data, res_T &res, const typename CONFIG_T::weight_t &weights,
                         const typename CONFIG_T::bias_t &biases) {
    using output_elem_t = typename res_T::value_type;

    constexpr int total_pf = (CONFIG_T::parallelization_factor > 0) ? CONFIG_T::parallelization_factor : 1;
    constexpr int max_filters = CONFIG_T::n_filt;
    constexpr int spatial_capacity = CONFIG_T::out_height * CONFIG_T::out_width;
    constexpr int spatial_pf = (total_pf < spatial_capacity) ? total_pf : spatial_capacity;
    constexpr int raw_pfc = (spatial_pf < CONFIG_T::out_width) ? spatial_pf : CONFIG_T::out_width;
    constexpr int safe_pfc = (raw_pfc < 1) ? 1 : raw_pfc;
    constexpr int raw_pfr = (safe_pfc > 0) ? (spatial_pf / safe_pfc) : spatial_pf;
    constexpr int capped_pfr = (raw_pfr < CONFIG_T::out_height) ? raw_pfr : CONFIG_T::out_height;
    constexpr int pfr = (capped_pfr < 1) ? 1 : capped_pfr;
    constexpr int spatial_tiles = safe_pfc * pfr;
    constexpr int filter_lane_candidates = (spatial_tiles > 0) ? (total_pf / spatial_tiles) : total_pf;
    constexpr int filters_per_iter_parallel =
        (filter_lane_candidates < 1) ? 1 : ((filter_lane_candidates > max_filters) ? max_filters : filter_lane_candidates);
    constexpr int total_macs = max_filters * CONFIG_T::n_chan;
    constexpr int reuse_parallel_macs =
        (CONFIG_T::reuse_factor > 0) ? ((total_macs + CONFIG_T::reuse_factor - 1) / CONFIG_T::reuse_factor) : total_macs;
    constexpr int reuse_filter_lanes =
        (CONFIG_T::n_chan > 0) ? ((reuse_parallel_macs + CONFIG_T::n_chan - 1) / CONFIG_T::n_chan) : reuse_parallel_macs;
    constexpr int bounded_reuse_lanes =
        (reuse_filter_lanes < 1) ? 1 : ((reuse_filter_lanes > max_filters) ? max_filters : reuse_filter_lanes);
    constexpr int filters_per_iter =
        (filters_per_iter_parallel < bounded_reuse_lanes) ? filters_per_iter_parallel : bounded_reuse_lanes;

HeightLoop:
#pragma unroll pfr
    for (int oh = 0; oh < CONFIG_T::out_height; oh++) {
    WidthLoop:
#pragma unroll safe_pfc
        for (int ow = 0; ow < CONFIG_T::out_width; ow++) {
            output_elem_t acc[CONFIG_T::n_filt];

            if constexpr (UNROLL_BIAS) {
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
                        output_elem_t in_val = static_cast<output_elem_t>(data[in_base + cc]);

                        for (int ff_base = 0; ff_base < max_filters; ff_base += filters_per_iter) {
                            if constexpr (UNROLL_FILTERS) {
#pragma unroll
                                for (int lane = 0; lane < filters_per_iter; lane++) {
                                    int ff = ff_base + lane;
                                    if (ff >= max_filters) {
                                        continue;
                                    }
                                    int w_idx =
                                        ((((fh * CONFIG_T::filt_width) + fw) * CONFIG_T::n_filt) + ff) * CONFIG_T::n_chan +
                                        cc;
                                    acc[ff] += static_cast<output_elem_t>(in_val * weights[w_idx]);
                                }
                            } else {
                                for (int lane = 0; lane < filters_per_iter; lane++) {
                                    int ff = ff_base + lane;
                                    if (ff >= max_filters) {
                                        continue;
                                    }
                                    int w_idx =
                                        ((((fh * CONFIG_T::filt_width) + fw) * CONFIG_T::n_filt) + ff) * CONFIG_T::n_chan +
                                        cc;
                                    acc[ff] += static_cast<output_elem_t>(in_val * weights[w_idx]);
                                }
                            }
                        }
                    }
                }
            }

            if constexpr (UNROLL_WRITES) {
#pragma unroll
                for (int ff = 0; ff < CONFIG_T::n_filt; ff++) {
                    int out_idx = (oh * CONFIG_T::out_width + ow) * CONFIG_T::n_filt + ff;
                    res[out_idx] = acc[ff];
                }
            } else {
                for (int ff = 0; ff < CONFIG_T::n_filt; ff++) {
                    int out_idx = (oh * CONFIG_T::out_width + ow) * CONFIG_T::n_filt + ff;
                    res[out_idx] = acc[ff];
                }
            }
        }
    }
}

} // namespace nnet

#endif
