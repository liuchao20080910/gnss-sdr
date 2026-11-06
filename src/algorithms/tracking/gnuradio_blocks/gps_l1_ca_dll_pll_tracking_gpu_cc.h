/*!
 * \file gps_l1_ca_dll_pll_tracking_gpu_cc.h
 * \brief Implementation of a code DLL + carrier PLL tracking block, GPU ACCELERATED
 * \author Javier Arribas, 2015. jarribas(at)cttc.es
 *
 * Code DLL + carrier PLL according to the algorithms described in:
 * K.Borre, D.M.Akos, N.Bertelsen, P.Rinder, and S.H.Jensen,
 * A Software-Defined GPS and Galileo Receiver. A Single-Frequency Approach,
 * Birkhauser, 2007
 *
 * -----------------------------------------------------------------------------
 *
 * GNSS-SDR is a Global Navigation Satellite System software-defined receiver.
 * This file is part of GNSS-SDR.
 *
 * Copyright (C) 2010-2020  (see AUTHORS file for a list of contributors)
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * -----------------------------------------------------------------------------
 */

#ifndef GNSS_SDR_GPS_L1_CA_DLL_PLL_TRACKING_GPU_CC_H
#define GNSS_SDR_GPS_L1_CA_DLL_PLL_TRACKING_GPU_CC_H

#include "cuda_multicorrelator.h"
#include "gnss_block_interface.h"
#include "gnss_synchro.h"
#include "tracking_2nd_DLL_filter.h"
#include "tracking_FLL_PLL_filter.h"
#include "bit_synchronizer.h"
#include "exponential_smoother.h"
#include "tow_to_trk.h"
#if 0
#include "tracking_loop_filter.h"
#endif
#include <gnuradio/block.h>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

/** \addtogroup Tracking
 * \{ */
/** \addtogroup Tracking_gnuradio_blocks
 * \{ */


class Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc;

using gps_l1_ca_dll_pll_tracking_gpu_cc_sptr = gnss_shared_ptr<Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc>;

gps_l1_ca_dll_pll_tracking_gpu_cc_sptr
gps_l1_ca_dll_pll_make_tracking_gpu_cc(
    int64_t fs_in,
    uint32_t vector_length,
    bool dump,
    std::string dump_filename,
    float pll_bw_hz,
    float dll_bw_hz,
    float early_late_space_chips);


/*!
 * \brief This class implements a DLL + PLL tracking loop block
 */
class Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc : public gr::block
{
public:
    ~Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc();

    // 锁定判据配置：默认值 = 命令行 flag 默认值；适配器会按 conf 覆盖
    // （对应 CPU 版 Dll_Pll_Conf 里同名参数的口径）
    struct Lock_Config
    {
        double carrier_lock_th{0.7};
        double cn0_min{25.0};
        int32_t cn0_samples{20};
        int32_t max_carrier_lock_fail{5000};
        int32_t max_code_lock_fail{50};
        double cn0_smoother_alpha{0.002};
        double carrier_lock_smoother_alpha{0.002};
        int32_t cn0_smoother_samples{200};
        int32_t carrier_lock_smoother_samples{25};
        double pull_in_time_s{3.0};
        double bit_sync_time_limit_s{20.0};
    };

    void set_channel(uint32_t channel);
    // 由跟踪适配器在 start_tracking() 之前调用，从 conf 读入锁定门限
    void set_lock_config(const Lock_Config &c);
    void set_gnss_synchro(Gnss_Synchro *p_gnss_synchro);
    void start_tracking();

    int general_work(int noutput_items, gr_vector_int &ninput_items,
        gr_vector_const_void_star &input_items, gr_vector_void_star &output_items);

    void forecast(int noutput_items, gr_vector_int &ninput_items_required);

private:
    friend gps_l1_ca_dll_pll_tracking_gpu_cc_sptr
    gps_l1_ca_dll_pll_make_tracking_gpu_cc(
        int64_t fs_in,
        uint32_t vector_length,
        bool dump,
        std::string dump_filename,
        float pll_bw_hz,
        float dll_bw_hz,
        float early_late_space_chips);

    Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc(
        int64_t fs_in,
        uint32_t vector_length,
        bool dump,
        std::string dump_filename,
        float pll_bw_hz,
        float dll_bw_hz,
        float early_late_space_chips);
    void apply_lock_config();
    void update_local_code();
    void update_local_carrier();
    void check_carrier_phase_coherent_initialization();

    // PLL and DLL filter library
    #if 1
    Tracking_2nd_DLL_filter d_code_loop_filter;
    #else
    Tracking_loop_filter d_code_loop_filter;
    #endif
    Tracking_FLL_PLL_filter d_carrier_loop_filter;

    Gnss_Synchro *d_acquisition_gnss_synchro;

    std::vector<gr_complex> d_Prompt_buffer;

    // file dump
    std::string d_dump_filename;
    std::ofstream d_dump_file;

    std::map<std::string, std::string> systemName;
    std::string sys;

    // tracking configuration vars
    int64_t d_if_freq;
    int64_t d_fs_in;
    double d_early_late_spc_chips;
    uint32_t d_vector_length;
    uint32_t d_channel;
    int32_t d_n_correlator_taps;

    // GPU HOST PINNED MEMORY IN/OUT VECTORS
    cuda_multicorrelator *multicorrelator_gpu;
    gr_complex *in_gpu;
    gr_complex *d_correlator_outs;
    gr_complex *d_ca_code;
    float *d_local_code_shift_chips;

    gr_complex *d_Early;
    gr_complex *d_Prompt;
    gr_complex *d_Late;

    // remaining code phase and carrier phase between tracking loops
    double d_rem_code_phase_samples;
    double d_rem_code_phase_chips;
    double d_rem_carrier_phase_rad;

    // acquisition
    double d_acq_code_phase_samples;
    double d_acq_carrier_doppler_hz;

    // tracking vars
    double d_code_freq_chips;
    double d_code_phase_step_chips;
    double d_carrier_doppler_hz;
    double d_carrier_phase_step_rad;
    double d_acc_carrier_phase_cycles;
    double d_code_phase_samples;
    double d_pll_to_dll_assist_secs_Ti;

    // Integration period in samples
    int32_t d_correlation_length_samples;

    // processing samples counters
    uint64_t d_sample_counter;
    uint64_t d_acq_sample_stamp;

    // CN0 estimation and lock detector
    double d_carrier_lock_test;
    double d_CN0_SNV_dB_Hz;
    double d_carrier_lock_threshold;
    int32_t d_carrier_lock_fail_counter;
    int32_t d_cn0_estimation_counter;

    // control vars
    bool d_acc_carrier_phase_initialized;
    bool d_enable_tracking;
    bool d_pull_in;
    bool d_dump;
    bool d_pull_in_transitory;
    bool d_Flag_PLL_180_deg_phase_locked;
    uint64_t d_pull_in_start_sample;
    gr_complex d_P_accu;
    int32_t d_P_accu_count;
    double d_carrier_lock_test_smoothed;
    std::shared_ptr<TOW_to_trk> d_last_tow_received;
    uint64_t d_tow_from_telemetry_ms;

    // 导航比特同步（对齐 20 ms 符号累加窗到导航比特边沿，
    // 与 CPU 版 dll_pll_veml_tracking 的 HistogramBitSynchronizer 行为一致）
    std::unique_ptr<HistogramBitSynchronizer> d_bit_sync;

    Lock_Config d_lock_cfg;
    Exponential_Smoother d_cn0_smoother;
    Exponential_Smoother d_carrier_lock_test_smoother;
    int32_t d_code_lock_fail_counter;
    bool d_bit_sync_locked;
    bool d_wait_for_bit_edge;
    int64_t d_bit_sync_target_epoch;
    int32_t d_symbols_per_bit;
    int32_t d_phase_180_counter;
    void msg_handler_telemetry_to_trk(const pmt::pmt_t &msg);
    float d_pll_bw_hz;
    float d_dll_bw_hz;
    float d_fll_bw_hz;
    int d_pll_filter_order;
};


/** \} */
/** \} */
#endif  // GNSS_SDR_GPS_L1_CA_DLL_PLL_TRACKING_GPU_CC_H
