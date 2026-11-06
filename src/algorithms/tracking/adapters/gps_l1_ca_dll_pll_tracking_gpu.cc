/*!
 * \file gps_l1_ca_dll_pll_tracking_gpu.cc
 * \brief Implementation of an adapter of a DLL+PLL tracking loop block using GPU accelerated functions
 * for GPS L1 C/A to a TrackingInterface
 * \author Javier Arribas, 2015. jarribas(at)cttc.es
 *
 * Code DLL + carrier PLL according to the algorithms described in:
 * K.Borre, D.M.Akos, N.Bertelsen, P.Rinder, and S.H.Jensen,
 * A Software-Defined GPS and Galileo Receiver. A Single-Frequency
 * Approach, Birkhauser, 2007
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


#include "gps_l1_ca_dll_pll_tracking_gpu.h"
#include "GPS_L1_CA.h"
#include "configuration_interface.h"
#include "gnss_sdr_flags.h"

#if USE_GLOG_AND_GFLAGS
#include <glog/logging.h>
#else
#include <absl/log/log.h>
#endif


GpsL1CaDllPllTrackingGPU::GpsL1CaDllPllTrackingGPU(
    const ConfigurationInterface* configuration, std::string role,
    unsigned int in_streams, unsigned int out_streams) : role_(role), in_streams_(in_streams), out_streams_(out_streams)
{
    DLOG(INFO) << "role " << role;
    // ################# CONFIGURATION PARAMETERS ########################
    int fs_in;
    int vector_length;
    bool dump;
    std::string dump_filename;
    std::string item_type;
    const std::string default_item_type("gr_complex");
    float pll_bw_hz;
    float dll_bw_hz;
    float early_late_space_chips;
    item_type = configuration->property(role + ".item_type", default_item_type);
    // vector_length = configuration->property(role + ".vector_length", 2048);
    int fs_in_deprecated = configuration->property("GNSS-SDR.internal_fs_hz", 2048000);
    fs_in = configuration->property("GNSS-SDR.internal_fs_sps", fs_in_deprecated);
    dump = configuration->property(role + ".dump", false);
    pll_bw_hz = configuration->property(role + ".pll_bw_hz", 50.0);
#if USE_GLOG_AND_GFLAGS
    if (FLAGS_pll_bw_hz != 0.0) pll_bw_hz = static_cast<float>(FLAGS_pll_bw_hz);
#else
    if (absl::GetFlag(FLAGS_pll_bw_hz) != 0.0) pll_bw_hz = static_cast<float>(absl::GetFlag(FLAGS_pll_bw_hz));
#endif
    dll_bw_hz = configuration->property(role + ".dll_bw_hz", 2.0);
#if USE_GLOG_AND_GFLAGS
    if (FLAGS_dll_bw_hz != 0.0) dll_bw_hz = static_cast<float>(FLAGS_dll_bw_hz);
#else
    if (absl::GetFlag(FLAGS_dll_bw_hz) != 0.0) dll_bw_hz = static_cast<float>(absl::GetFlag(FLAGS_dll_bw_hz));
#endif

    early_late_space_chips = configuration->property(role + ".early_late_space_chips", 0.5);
    const std::string default_dump_filename("./track_ch");
    dump_filename = configuration->property(role + ".dump_filename", default_dump_filename);
    vector_length = std::round(fs_in / (GPS_L1_CA_CODE_RATE_CPS / GPS_L1_CA_CODE_LENGTH_CHIPS));

    // ============================================================
    // 锁定判据参数：默认值取命令行 flag，可由 conf 里 Tracking_1C.* 覆盖
    // （与 CPU 版 Dll_Pll_Conf 的口径一致；注意 conf 优先于 flag）
    // ============================================================
#if USE_GLOG_AND_GFLAGS
    Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc::Lock_Config lock_cfg;
    lock_cfg.carrier_lock_th = FLAGS_carrier_lock_th;
    lock_cfg.cn0_min = FLAGS_cn0_min;
    lock_cfg.cn0_samples = FLAGS_cn0_samples;
    lock_cfg.max_code_lock_fail = FLAGS_max_lock_fail;
    lock_cfg.max_carrier_lock_fail = FLAGS_max_carrier_lock_fail;
#else
    Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc::Lock_Config lock_cfg;
    lock_cfg.carrier_lock_th = absl::GetFlag(FLAGS_carrier_lock_th);
    lock_cfg.cn0_min = absl::GetFlag(FLAGS_cn0_min);
    lock_cfg.cn0_samples = absl::GetFlag(FLAGS_cn0_samples);
    lock_cfg.max_code_lock_fail = absl::GetFlag(FLAGS_max_lock_fail);
    lock_cfg.max_carrier_lock_fail = absl::GetFlag(FLAGS_max_carrier_lock_fail);
#endif
    lock_cfg.carrier_lock_th = configuration->property(role + ".carrier_lock_th", lock_cfg.carrier_lock_th);
    lock_cfg.cn0_min = configuration->property(role + ".cn0_min", lock_cfg.cn0_min);
    lock_cfg.cn0_samples = configuration->property(role + ".cn0_samples", lock_cfg.cn0_samples);
    lock_cfg.max_code_lock_fail = configuration->property(role + ".max_lock_fail", lock_cfg.max_code_lock_fail);
    lock_cfg.max_carrier_lock_fail = configuration->property(role + ".max_carrier_lock_fail", lock_cfg.max_carrier_lock_fail);
    lock_cfg.cn0_smoother_alpha = configuration->property(role + ".cn0_smoother_alpha", lock_cfg.cn0_smoother_alpha);
    lock_cfg.carrier_lock_smoother_alpha = configuration->property(role + ".carrier_lock_test_smoother_alpha", lock_cfg.carrier_lock_smoother_alpha);
    lock_cfg.cn0_smoother_samples = configuration->property(role + ".cn0_smoother_samples", lock_cfg.cn0_smoother_samples);
    lock_cfg.carrier_lock_smoother_samples = configuration->property(role + ".carrier_lock_test_smoother_samples", lock_cfg.carrier_lock_smoother_samples);
    lock_cfg.pull_in_time_s = configuration->property(role + ".pull_in_time_s", lock_cfg.pull_in_time_s);
    lock_cfg.bit_sync_time_limit_s = configuration->property(role + ".bit_synchronization_time_limit_s", lock_cfg.bit_sync_time_limit_s);

    // ################# MAKE TRACKING GNURadio object ###################
    if (item_type.compare("gr_complex") == 0)
        {
            item_size_ = sizeof(gr_complex);
            tracking_ = gps_l1_ca_dll_pll_make_tracking_gpu_cc(
                fs_in,
                vector_length,
                dump,
                dump_filename,
                pll_bw_hz,
                dll_bw_hz,
                early_late_space_chips);
        }
    else
        {
            item_size_ = 0;
            LOG(WARNING) << item_type << " unknown tracking item type.";
        }
    if (item_size_ != 0)
        {
            tracking_->set_lock_config(lock_cfg);
        }
    channel_ = 0;
    DLOG(INFO) << "tracking(" << tracking_->unique_id() << ")";
    if (in_streams_ > 1)
        {
            LOG(ERROR) << "This implementation only supports one input stream";
        }
    if (out_streams_ > 1)
        {
            LOG(ERROR) << "This implementation only supports one output stream";
        }
}


GpsL1CaDllPllTrackingGPU::~GpsL1CaDllPllTrackingGPU()
{
}


void GpsL1CaDllPllTrackingGPU::start_tracking()
{
    tracking_->start_tracking();
}


void GpsL1CaDllPllTrackingGPU::stop_tracking()
{
}


/*
 * Set tracking channel unique ID
 */
void GpsL1CaDllPllTrackingGPU::set_channel(unsigned int channel)
{
    channel_ = channel;
    tracking_->set_channel(channel);
}


void GpsL1CaDllPllTrackingGPU::set_gnss_synchro(Gnss_Synchro* p_gnss_synchro)
{
    tracking_->set_gnss_synchro(p_gnss_synchro);
}


void GpsL1CaDllPllTrackingGPU::connect(gr::top_block_sptr top_block)
{
    if (top_block)
        { /* top_block is not null */
        };
    // nothing to connect, now the tracking uses gr_sync_decimator
}


void GpsL1CaDllPllTrackingGPU::disconnect(gr::top_block_sptr top_block)
{
    if (top_block)
        { /* top_block is not null */
        };
    // nothing to disconnect, now the tracking uses gr_sync_decimator
}


gr::basic_block_sptr GpsL1CaDllPllTrackingGPU::get_left_block()
{
    return tracking_;
}


gr::basic_block_sptr GpsL1CaDllPllTrackingGPU::get_right_block()
{
    return tracking_;
}
