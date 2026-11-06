/*!
 * \file gps_l1_ca_dll_pll_tracking_gpu_cc.cc
 * \brief Implementation of a code DLL + carrier PLL tracking block GPU ACCELERATED
 * \author Javier Arribas, 2015. jarribas(at)cttc.es
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

#include "gps_l1_ca_dll_pll_tracking_gpu_cc.h"
#include "GPS_L1_CA.h"
#include "gnss_satellite.h"
#include "gnss_sdr_flags.h"
#include "gps_sdr_signal_replica.h"
#include "bit_synchronizer.h"
#include "exponential_smoother.h"
#include "lock_detectors.h"
#include "tracking_discriminators.h"
#include "tow_to_trk.h"
#include <boost/lexical_cast.hpp>
#include <gnuradio/io_signature.h>
#include <algorithm>
#include <cmath>
#include <cuda_profiler_api.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>

#if USE_GLOG_AND_GFLAGS
#include <glog/logging.h>
#else
#include <absl/log/log.h>
#endif

#if PMT_USES_BOOST_ANY
#include <boost/any.hpp>
namespace wht = boost;
#else
#include <any>


namespace wht = std;
#endif

// 调试打印总开关：0 = 关闭（生产默认，编译器会把整段消掉）；1 = 打开
#ifndef GNSS_SDR_VERBOSE_DEBUG
#define GNSS_SDR_VERBOSE_DEBUG 0
#endif

gps_l1_ca_dll_pll_tracking_gpu_cc_sptr
gps_l1_ca_dll_pll_make_tracking_gpu_cc(
    int64_t fs_in,
    uint32_t vector_length,
    bool dump,
    std::string dump_filename,
    float pll_bw_hz,
    float dll_bw_hz,
    float early_late_space_chips)
{
    return gps_l1_ca_dll_pll_tracking_gpu_cc_sptr(new Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc(
        fs_in, vector_length, dump, dump_filename, pll_bw_hz, dll_bw_hz, early_late_space_chips));
}


void Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc::forecast(int noutput_items,
    gr_vector_int &ninput_items_required)
{
    if (noutput_items != 0)
        {
            ninput_items_required[0] = static_cast<int32_t>(d_vector_length) * 2;  // set the required available samples in each call
        }
}


Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc::Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc(
    int64_t fs_in,
    uint32_t vector_length,
    bool dump,
    std::string dump_filename,
    float pll_bw_hz,
    float dll_bw_hz,
    float early_late_space_chips) : gr::block("Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc", gr::io_signature::make(1, 1, sizeof(gr_complex)),
                                        gr::io_signature::make(1, 1, sizeof(Gnss_Synchro)))
{
    // Telemetry bit synchronization message port input
    this->message_port_register_in(pmt::mp("preamble_timestamp_s"));
    this->message_port_register_out(pmt::mp("events"));
    this->message_port_register_in(pmt::mp("telemetry_to_trk"));
    this->set_msg_handler(
        pmt::mp("telemetry_to_trk"),
#if HAS_GENERIC_LAMBDA
        [this](auto &&PH1) { msg_handler_telemetry_to_trk(PH1); });
#else
        boost::bind(&Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc::msg_handler_telemetry_to_trk, this, boost::placeholders::_1));
#endif
    // initialize internal vars
    d_dump = dump;
    d_fs_in = fs_in;
    d_vector_length = vector_length;
    d_dump_filename = dump_filename;
    d_correlation_length_samples = static_cast<int32_t>(d_vector_length);

    // Initialize tracking  ==========================================
    //d_code_loop_filter.set_DLL_BW(dll_bw_hz);
    //d_carrier_loop_filter.set_params(10.0, pll_bw_hz, 2);
    //d_carrier_loop_filter.set_params(0.0, pll_bw_hz, 2);
    d_pll_bw_hz = pll_bw_hz;
    d_dll_bw_hz = dll_bw_hz;
    d_fll_bw_hz = 35.0;        // 和 CPU 版默认值一致
    d_pll_filter_order = 3;    // 和 CPU 版默认值一致
    #if 1
    d_code_loop_filter.set_DLL_BW(dll_bw_hz);
    #else
    d_code_loop_filter = Tracking_loop_filter(0.001f, dll_bw_hz, 2, false);
    #endif
    d_carrier_loop_filter.set_params(d_fll_bw_hz, d_pll_bw_hz, d_pll_filter_order);

    // --- DLL variables -------------------------------------------------------
    d_early_late_spc_chips = early_late_space_chips;  // Define early-late offset (in chips)

    // Set GPU flags
    cudaSetDeviceFlags(cudaDeviceMapHost);
    // allocate host memory
    // pinned memory mode - use special function to get OS-pinned memory
    d_n_correlator_taps = 3;  // Early, Prompt, and Late
    // Get space for a vector with the C/A code replica sampled 1x/chip
    cudaHostAlloc(reinterpret_cast<void **>(&d_ca_code), (static_cast<int32_t>(GPS_L1_CA_CODE_LENGTH_CHIPS) * sizeof(gr_complex)), cudaHostAllocMapped);
    // Get space for the resampled early / prompt / late local replicas
    cudaHostAlloc(reinterpret_cast<void **>(&d_local_code_shift_chips), d_n_correlator_taps * sizeof(float), cudaHostAllocMapped);
    cudaHostAlloc(reinterpret_cast<void **>(&in_gpu), 2 * d_vector_length * sizeof(gr_complex), cudaHostAllocMapped);
    // correlator outputs (scalar)
    cudaHostAlloc(reinterpret_cast<void **>(&d_correlator_outs), sizeof(gr_complex) * d_n_correlator_taps, cudaHostAllocMapped);

    // Set TAPs delay values [chips]
    d_local_code_shift_chips[0] = -d_early_late_spc_chips;
    d_local_code_shift_chips[1] = 0.0;
    d_local_code_shift_chips[2] = d_early_late_spc_chips;

    // --- Perform initializations ------------------------------
    multicorrelator_gpu = new cuda_multicorrelator();
    // local code resampler on GPU
    multicorrelator_gpu->init_cuda_integrated_resampler(2 * d_vector_length, GPS_L1_CA_CODE_LENGTH_CHIPS, d_n_correlator_taps);
    multicorrelator_gpu->set_input_output_vectors(d_correlator_outs, in_gpu);

    // define initial code frequency basis of NCO
    d_code_freq_chips = GPS_L1_CA_CODE_RATE_CPS;
    // define residual code phase (in chips)
    d_rem_code_phase_samples = 0.0;
    // define residual carrier phase
    d_rem_carrier_phase_rad = 0.0;

    // sample synchronization
    d_sample_counter = 0ULL;
    // d_sample_counter_seconds = 0;
    d_acq_sample_stamp = 0;

    d_enable_tracking = false;
    d_pull_in = false;
    d_pull_in_transitory = false;
    d_pull_in_start_sample = 0ULL;
    // 导航比特同步相关（同步器实体在 start_tracking() 里创建）
    d_bit_sync_locked = false;
    d_wait_for_bit_edge = false;
    d_bit_sync_target_epoch = 0;
    d_symbols_per_bit = GPS_CA_TELEMETRY_SYMBOLS_PER_BIT;  // 20 symbols/bit (20 ms)
    d_phase_180_counter = 0;
    d_code_lock_fail_counter = 0;
    d_carrier_lock_test_smoothed = 1.0;
    d_Flag_PLL_180_deg_phase_locked = false;
    d_P_accu = gr_complex(0, 0);
    d_P_accu_count = 0;

    // CN0 estimation and lock detector buffers
    d_cn0_estimation_counter = 0;
#if USE_GLOG_AND_GFLAGS
    d_Prompt_buffer = std::vector<gr_complex>(FLAGS_cn0_samples);
#else
    d_Prompt_buffer = std::vector<gr_complex>(absl::GetFlag(FLAGS_cn0_samples));
#endif


    d_carrier_lock_test = 1;
    apply_lock_config();  // 配置 CN0/载波锁平滑器与门限（适配器可再覆盖）
    d_last_tow_received = std::make_shared<TOW_to_trk>();
    d_tow_from_telemetry_ms = 0ULL;
    d_CN0_SNV_dB_Hz = 0;
    d_carrier_lock_fail_counter = 0;
#if USE_GLOG_AND_GFLAGS
    d_carrier_lock_threshold = FLAGS_carrier_lock_th;
#else
    d_carrier_lock_threshold = absl::GetFlag(FLAGS_carrier_lock_th);
#endif
    systemName["G"] = std::string("GPS");
    systemName["S"] = std::string("SBAS");

#if GNURADIO_GREATER_THAN_38
    this->set_relative_rate(1, static_cast<uint64_t>(d_vector_length * 2));
#else
    this->set_relative_rate(1.0 / static_cast<double>(d_vector_length * 2));
#endif

    d_acquisition_gnss_synchro = 0;
    d_channel = 0;
    d_acq_code_phase_samples = 0.0;
    d_acq_carrier_doppler_hz = 0.0;
    d_carrier_doppler_hz = 0.0;
    d_acc_carrier_phase_cycles = 0.0;
    d_code_phase_samples = 0.0;

    d_pll_to_dll_assist_secs_Ti = 0.0;
    d_rem_code_phase_chips = 0.0;
    d_code_phase_step_chips = 0.0;
    d_carrier_phase_step_rad = 0.0;

    d_acc_carrier_phase_initialized = false;
    // set_min_output_buffer((int64_t)300);
}

void Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc::msg_handler_telemetry_to_trk(const pmt::pmt_t &msg)
{
    if (GNSS_SDR_VERBOSE_DEBUG) std::cout << "[GPU-TOW-MSG] ch=" << d_channel
              << " callback called, type_hash=" << pmt::any_ref(msg).type().hash_code()
              << " expected_hash=" << typeid(const std::shared_ptr<TOW_to_trk>).hash_code()
              << std::endl;
    try
        {
            if (pmt::any_ref(msg).type().hash_code() == typeid(int).hash_code())
                {
                    const int tlm_event = wht::any_cast<int>(pmt::any_ref(msg));
                    if (tlm_event == 1)
                        {
                            if (GNSS_SDR_VERBOSE_DEBUG) std::cout << "[GPU-TLM-FAULT] ch=" << d_channel << " telemetry fault received" << std::endl;
                            gr::thread::scoped_lock lock(d_setlock);
                            d_carrier_lock_fail_counter = 200000;  // force loss-of-lock condition
                        }
                }
            else if (pmt::any_ref(msg).type().hash_code() == typeid(const std::shared_ptr<TOW_to_trk>).hash_code())
                {
                    const auto tow_event = wht::any_cast<const std::shared_ptr<TOW_to_trk>>(pmt::any_ref(msg));
                    if (GNSS_SDR_VERBOSE_DEBUG) std::cout << "[GPU-TOW-MSG] ch=" << d_channel
                              << " tow_channel=" << tow_event->channel
                              << " tow_prn=" << tow_event->prn
                              << " my_prn=" << d_acquisition_gnss_synchro->PRN
                              << " tow=" << tow_event->tow
                              << " sample_stamp=" << tow_event->sample_stamp
                              << std::endl;
                    if (tow_event->channel == static_cast<int32_t>(d_channel) && tow_event->prn == d_acquisition_gnss_synchro->PRN)
                        {
                            d_last_tow_received = tow_event;
                        }
                    else
                        {
                            if (GNSS_SDR_VERBOSE_DEBUG) std::cout << "[GPU-TOW-MSG] ch=" << d_channel
                                      << " condition NOT matched" << std::endl;
                        }
                }
            else
                {
                    if (GNSS_SDR_VERBOSE_DEBUG) std::cout << "[GPU-TOW-MSG] ch=" << d_channel
                              << " type hash NOT matched" << std::endl;
                }
        }
    catch (const wht::bad_any_cast &e)
        {
            LOG(WARNING) << "msg_handler_telemetry_to_trk Bad any_cast: " << e.what();
        }
    catch (std::exception &ex)
        {
            LOG(WARNING) << "msg_handler_telemetry_to_trk Bad any_cast: " << ex.what();
        }
}

void Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc::start_tracking()
{
    /*
     *  correct the code phase according to the delay between acq and trk
     */
    d_acq_code_phase_samples = d_acquisition_gnss_synchro->Acq_delay_samples;
    d_acq_carrier_doppler_hz = d_acquisition_gnss_synchro->Acq_doppler_hz;
    d_acq_sample_stamp = d_acquisition_gnss_synchro->Acq_samplestamp_samples;

    const int64_t acq_trk_diff_samples = static_cast<int64_t>(d_sample_counter) - static_cast<int64_t>(d_acq_sample_stamp);  // -d_vector_length;
    DLOG(INFO) << "Number of samples between Acquisition and Tracking =" << acq_trk_diff_samples;
    const double acq_trk_diff_seconds = static_cast<double>(acq_trk_diff_samples) / static_cast<double>(d_fs_in);
    // doppler effect
    // Fd=(C/(C+Vr))*F
    const double radial_velocity = (GPS_L1_FREQ_HZ + d_acq_carrier_doppler_hz) / GPS_L1_FREQ_HZ;
    // new chip and prn sequence periods based on acq Doppler
    d_code_freq_chips = radial_velocity * GPS_L1_CA_CODE_RATE_CPS;
    d_code_phase_step_chips = static_cast<double>(d_code_freq_chips) / static_cast<double>(d_fs_in);
    const double T_chip_mod_seconds = 1 / d_code_freq_chips;
    const double T_prn_mod_seconds = T_chip_mod_seconds * GPS_L1_CA_CODE_LENGTH_CHIPS;
    const double T_prn_mod_samples = T_prn_mod_seconds * static_cast<double>(d_fs_in);

    d_correlation_length_samples = round(T_prn_mod_samples);

    const double T_prn_true_seconds = GPS_L1_CA_CODE_LENGTH_CHIPS / GPS_L1_CA_CODE_RATE_CPS;
    const double T_prn_true_samples = T_prn_true_seconds * static_cast<double>(d_fs_in);
    const double T_prn_diff_seconds = T_prn_true_seconds - T_prn_mod_seconds;
    const double N_prn_diff = acq_trk_diff_seconds / T_prn_true_seconds;
    double corrected_acq_phase_samples, delay_correction_samples;
    corrected_acq_phase_samples = fmod((d_acq_code_phase_samples + T_prn_diff_seconds * N_prn_diff * static_cast<double>(d_fs_in)), T_prn_true_samples);
    if (corrected_acq_phase_samples < 0)
        {
            corrected_acq_phase_samples = T_prn_mod_samples + corrected_acq_phase_samples;
        }
    delay_correction_samples = d_acq_code_phase_samples - corrected_acq_phase_samples;

    d_acq_code_phase_samples = corrected_acq_phase_samples;

    d_carrier_doppler_hz = d_acq_carrier_doppler_hz;

    d_carrier_phase_step_rad = TWO_PI * d_carrier_doppler_hz / static_cast<double>(d_fs_in);

    // DLL/PLL filter initialization
    //d_carrier_loop_filter.initialize(d_acq_carrier_doppler_hz);  // The carrier loop filter implements the Doppler accumulator
    //d_code_loop_filter.initialize();                             // initialize the code filter
    // DLL/PLL filter initialization
    // DLL/PLL filter initialization
    //d_carrier_loop_filter.set_params(0.0, 30.0, 2);                  // FLL 关闭，PLL 带宽 30 Hz，2 阶
    //d_code_loop_filter.set_DLL_BW(2.0);                              // DLL 带宽 2 Hz
    //d_carrier_loop_filter.initialize(d_acq_carrier_doppler_hz);      // The carrier loop filter implements the Doppler accumulator
    //d_code_loop_filter.initialize();                                 // initialize the code filter
    //d_carrier_loop_filter.set_params(d_fll_bw_hz, d_pll_bw_hz, d_pll_filter_order);
    d_carrier_loop_filter.set_params(d_fll_bw_hz, d_pll_bw_hz, d_pll_filter_order);
    #if 1
    d_code_loop_filter.set_DLL_BW(d_dll_bw_hz);
    #else
    d_code_loop_filter.set_noise_bandwidth(d_dll_bw_hz);
    d_code_loop_filter.set_update_interval(0.001f);
    #endif
    d_carrier_loop_filter.initialize(d_acq_carrier_doppler_hz);  // The carrier loop filter implements the Doppler accumulator
    d_code_loop_filter.initialize();                             // initialize the code filter

#if GNSS_SDR_VERBOSE_DEBUG
    if (GNSS_SDR_VERBOSE_DEBUG) std::cout << "[GPU-FILTER-INIT] ch=" << d_channel
            << " acq_doppler=" << d_acq_carrier_doppler_hz
            << std::endl;
#endif
    // generate local reference ALWAYS starting at chip 1 (1 sample per chip)
    //gps_l1_ca_code_gen_complex(own::span<gr_complex>(d_ca_code, static_cast<int32_t>(GPS_L1_CA_CODE_LENGTH_CHIPS)), d_acquisition_gnss_synchro->PRN, 0);
    std::vector<gr_complex> ca_code_tmp(static_cast<int32_t>(GPS_L1_CA_CODE_LENGTH_CHIPS));
    gps_l1_ca_code_gen_complex(own::span<gr_complex>(ca_code_tmp.data(), static_cast<int32_t>(GPS_L1_CA_CODE_LENGTH_CHIPS)), d_acquisition_gnss_synchro->PRN, 0);
    for (int i = 0; i < static_cast<int32_t>(GPS_L1_CA_CODE_LENGTH_CHIPS); i++)
        {
            d_ca_code[i] = gr_complex(ca_code_tmp[i].imag(), 0.0);  // 把虚部移到实部
        }

    multicorrelator_gpu->set_local_code_and_taps(static_cast<int32_t>(GPS_L1_CA_CODE_LENGTH_CHIPS), d_ca_code, d_local_code_shift_chips, d_n_correlator_taps);

    for (int32_t n = 0; n < d_n_correlator_taps; n++)
        {
            d_correlator_outs[n] = gr_complex(0, 0);
        }

    d_carrier_lock_fail_counter = 0;
    d_rem_code_phase_samples = 0.0;
    d_rem_carrier_phase_rad = 0.0;
    d_rem_code_phase_chips = 0.0;
    d_acc_carrier_phase_cycles = 0.0;
    d_pll_to_dll_assist_secs_Ti = 0.0;
    d_code_phase_samples = d_acq_code_phase_samples;

    sys = std::string(1, d_acquisition_gnss_synchro->System);

    // DEBUG OUTPUT
    std::cout << "Tracking of GPS L1 C/A signal started on channel " << d_channel << " for satellite " << Gnss_Satellite(systemName[sys], d_acquisition_gnss_synchro->PRN) << '\n';
    LOG(INFO) << "Tracking of GPS L1 C/A signal for satellite " << Gnss_Satellite(systemName[sys], d_acquisition_gnss_synchro->PRN) << " on channel " << d_channel;
#if GNSS_SDR_VERBOSE_DEBUG
    // ============================================================
    // 诊断：初始载波相位和相位步进
    // ============================================================
    
    LOG(INFO) << "[GPU-INIT-PHASE] ch=" << d_channel
              << " rem_carrier_phase_rad=" << d_rem_carrier_phase_rad
              << " carr_step=" << d_carrier_phase_step_rad
              << " carrier_doppler_hz=" << d_carrier_doppler_hz
              << " acq_code_phase_samples=" << d_acq_code_phase_samples;
#endif

    // enable tracking
    d_pull_in = true;
    d_enable_tracking = true;
    d_pull_in_transitory = true;
    d_pull_in_start_sample = d_sample_counter;
    d_carrier_lock_test_smoothed = 1.0;
    d_Flag_PLL_180_deg_phase_locked = false;
    d_acc_carrier_phase_initialized = false;
    // 每次跟踪启动都复位锁定检测状态（与 CPU 版一致）
    d_cn0_smoother.reset();
    d_carrier_lock_test_smoother.reset();
    d_cn0_estimation_counter = 0;
    d_carrier_lock_fail_counter = 0;
    d_code_lock_fail_counter = 0;
    d_P_accu = gr_complex(0, 0);
    d_P_accu_count = 0;

    // ============================================================
    // 导航比特同步器：每次跟踪启动都重建。
    // 目的：把 20 ms 的符号累加窗对齐到导航数据比特边沿。
    // 不做这件事时，每个通道的累加窗相位由本通道启动时刻决定且互不相同，
    // 会造成通道间"整数毫秒"级的时间戳/TOW 偏差（1 ms = 299.792 km 伪距），
    // 而该偏差不是公共项，接收机钟差吸收不掉，于是 PVT 被解到几千公里外。
    // 参考实现：dll_pll_veml_tracking.cc 的 state 2（比特同步）-> state 4。
    // ============================================================
    {
        HistogramBitSynchronizer::Config bit_sync_cfg;
        bit_sync_cfg.bit_period_ms = d_symbols_per_bit;  // 20 ms
        bit_sync_cfg.epoch_ms = 1;                       // 每个码周期调用一次 update()
        bit_sync_cfg.use_phase_dot_detector = true;
        bit_sync_cfg.min_prompt_mag = 0.0f;
        d_bit_sync = std::make_unique<HistogramBitSynchronizer>(bit_sync_cfg);
    }
    d_bit_sync_locked = false;
    d_wait_for_bit_edge = false;
    d_bit_sync_target_epoch = 0;
    d_phase_180_counter = 0;


    LOG(INFO) << "PULL-IN Doppler [Hz]=" << d_carrier_doppler_hz
              << " Code Phase correction [samples]=" << delay_correction_samples
              << " PULL-IN Code Phase [samples]=" << d_acq_code_phase_samples;
#if GNSS_SDR_VERBOSE_DEBUG    
    LOG(INFO) << "[GPU-INIT] ch=" << d_channel
            << " acq_doppler=" << d_acq_carrier_doppler_hz
            << " carrier_doppler_hz=" << d_carrier_doppler_hz
            << " rem_carrier_phase_rad=" << d_rem_carrier_phase_rad
            << " rem_code_phase_chips=" << d_rem_code_phase_chips
            << " rem_code_phase_samples=" << d_rem_code_phase_samples
            << " code_phase_step_chips=" << d_code_phase_step_chips
            << " carrier_phase_step_rad=" << d_carrier_phase_step_rad;
    // ============================================================
    // 诊断：捕获多普勒和初始多普勒
    // ============================================================
    LOG(INFO) << "[GPU-INIT-DOPPLER] ch=" << d_channel
              << " acq_doppler=" << d_acq_carrier_doppler_hz
              << " carrier_doppler_hz=" << d_carrier_doppler_hz
              << " carr_step=" << d_carrier_phase_step_rad
              << " code_freq_chips=" << d_code_freq_chips;
#endif
}


Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc::~Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc()
{
    if (d_dump_file.is_open())
        {
            try
                {
                    d_dump_file.close();
                }
            catch (const std::exception &ex)
                {
                    LOG(WARNING) << "Exception in destructor " << ex.what();
                }
        }
    try
        {
            cudaFreeHost(in_gpu);
            cudaFreeHost(d_correlator_outs);
            cudaFreeHost(d_local_code_shift_chips);
            cudaFreeHost(d_ca_code);
            multicorrelator_gpu->free_cuda();
            delete (multicorrelator_gpu);
        }
    catch (const std::exception &ex)
        {
            LOG(WARNING) << "Exception in destructor " << ex.what();
        }
}


void Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc::set_channel(uint32_t channel)
{
    d_channel = channel;
    LOG(INFO) << "Tracking Channel set to " << d_channel;
    // ############# ENABLE DATA FILE LOG #################
    if (d_dump == true)
        {
            if (d_dump_file.is_open() == false)
                {
                    try
                        {
                            d_dump_filename.append(boost::lexical_cast<std::string>(d_channel));
                            d_dump_filename.append(".dat");
                            d_dump_file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
                            d_dump_file.open(d_dump_filename.c_str(), std::ios::out | std::ios::binary);
                            LOG(INFO) << "Tracking dump enabled on channel " << d_channel << " Log file: " << d_dump_filename.c_str();
                        }
                    catch (const std::ofstream::failure *e)
                        {
                            LOG(WARNING) << "channel " << d_channel << " Exception opening trk dump file " << e->what();
                        }
                }
        }
}


void Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc::set_lock_config(const Lock_Config &c)
{
    d_lock_cfg = c;
    apply_lock_config();
}


void Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc::apply_lock_config()
{
    // 与 CPU 版 dll_pll_veml_tracking 的平滑器配置保持一致
    if (d_lock_cfg.cn0_samples < 1)
        {
            d_lock_cfg.cn0_samples = 1;
        }
    if (static_cast<int32_t>(d_Prompt_buffer.size()) < d_lock_cfg.cn0_samples)
        {
            d_Prompt_buffer.resize(d_lock_cfg.cn0_samples);
        }
    d_carrier_lock_threshold = d_lock_cfg.carrier_lock_th;

    d_cn0_smoother = Exponential_Smoother();
    d_cn0_smoother.set_alpha(static_cast<float>(d_lock_cfg.cn0_smoother_alpha));
    d_cn0_smoother.set_samples_for_initialization(d_lock_cfg.cn0_smoother_samples);

    d_carrier_lock_test_smoother = Exponential_Smoother();
    d_carrier_lock_test_smoother.set_alpha(static_cast<float>(d_lock_cfg.carrier_lock_smoother_alpha));
    d_carrier_lock_test_smoother.set_min_value(-1.0);
    d_carrier_lock_test_smoother.set_offset(0.0);
    d_carrier_lock_test_smoother.set_samples_for_initialization(d_lock_cfg.carrier_lock_smoother_samples);

    d_cn0_estimation_counter = 0;
    d_carrier_lock_fail_counter = 0;
    d_code_lock_fail_counter = 0;
}


void Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc::set_gnss_synchro(Gnss_Synchro *p_gnss_synchro)
{
    d_acquisition_gnss_synchro = p_gnss_synchro;
}


void Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc::check_carrier_phase_coherent_initialization()
{
    if (d_acc_carrier_phase_initialized == false)
        {
            d_acc_carrier_phase_cycles = -d_rem_carrier_phase_rad / TWO_PI;
            d_acc_carrier_phase_initialized = true;
        }
}


int Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc::general_work(int noutput_items __attribute__((unused)), gr_vector_int &ninput_items __attribute__((unused)),
    gr_vector_const_void_star &input_items, gr_vector_void_star &output_items)
{
    // Block input data and block output stream pointers
    const gr_complex *in = reinterpret_cast<const gr_complex *>(input_items[0]);
    Gnss_Synchro **out = reinterpret_cast<Gnss_Synchro **>(&output_items[0]);

    // GNSS_SYNCHRO OBJECT to interchange data between tracking->telemetry_decoder
    Gnss_Synchro current_synchro_data = Gnss_Synchro();

    // 用遥测解码器传来的 TOW 消息计算当前符号的 TOW

    if (d_last_tow_received->prn != 0 && d_last_tow_received->prn == d_acquisition_gnss_synchro->PRN)
        {
            const double time_diff_s = (static_cast<double>(this->nitems_read(0)) +
                                           d_correlation_length_samples - static_cast<double>(d_last_tow_received->sample_stamp)) /
                                       static_cast<double>(d_fs_in);
            const auto time_diff_ms = static_cast<uint64_t>((time_diff_s * 1000.0));
            d_tow_from_telemetry_ms = (d_last_tow_received->tow + time_diff_ms) % static_cast<uint64_t>(604800000);
        }
    else
        {
            d_tow_from_telemetry_ms = 0;
        }

    // ============================================================
    // 诊断：TOW 消息内容
    // ============================================================

    // process vars
    double code_error_chips_Ti = 0.0;
    double code_error_filt_chips = 0.0;
    double code_error_filt_secs_Ti = 0.0;
    double CURRENT_INTEGRATION_TIME_S = 0.001;
    double CORRECTED_INTEGRATION_TIME_S = 0.001;
    double dll_code_error_secs_Ti = 0.0;
    double carr_phase_error_secs_Ti = 0.0;
    bool loss_of_lock = false;

    // ============================================================
    // pull-in 过渡期：给 CN0 和载波锁定平滑器足够时间收敛
    // ============================================================
    const double PULL_IN_TIME_S = d_lock_cfg.pull_in_time_s;
    if (d_pull_in_transitory == true)
        {
            if ((d_sample_counter - d_pull_in_start_sample) > PULL_IN_TIME_S * static_cast<double>(d_fs_in))
                {
                    d_pull_in_transitory = false;
                    d_carrier_lock_fail_counter = 0;
                    LOG(INFO) << "[GPU-PULLIN] ch=" << d_channel << " transitory ENDED at sample_counter=" << d_sample_counter;
                }
        }

    // ============================================================
    // 比特同步超时保护（与 CPU 版 dll_pll_veml_tracking 的
    // bit_synchronization_time_limit_s 默认 20 s 保持一致）：
    // 超时仍未完成同步则强制失锁，让通道重新捕获，
    // 避免通道永远卡在"不输出有效符号"的状态。
    // ============================================================
    const double BIT_SYNC_TIME_LIMIT_S = d_lock_cfg.bit_sync_time_limit_s;
    if ((!d_bit_sync_locked) && (d_pull_in_transitory == false) &&
        ((d_sample_counter - d_pull_in_start_sample) > static_cast<uint64_t>(BIT_SYNC_TIME_LIMIT_S * static_cast<double>(d_fs_in))))
        {
            d_carrier_lock_fail_counter = 300000;  // > max_lock_fail，触发失锁
        }

    if (d_enable_tracking == true)
        {
            // Fill the acquisition data
            current_synchro_data = *d_acquisition_gnss_synchro;

            // Receiver signal alignment (pull-in)
            if (d_pull_in == true)
                {
                    const int32_t acq_to_trk_delay_samples = d_sample_counter - d_acq_sample_stamp;
                    const double acq_trk_shif_correction_samples = d_correlation_length_samples - fmod(static_cast<double>(acq_to_trk_delay_samples), static_cast<double>(d_correlation_length_samples));
                    const int32_t samples_offset = round(d_acq_code_phase_samples + acq_trk_shif_correction_samples);
                    //const int32_t samples_offset = round(d_acq_code_phase_samples);
#if GNSS_SDR_VERBOSE_DEBUG
                    LOG(INFO) << "[GPU-START] ch=" << d_channel
                              << " acq_code_phase_samples=" << d_acq_code_phase_samples
                              << " samples_offset=" << samples_offset
                              << " acq_sample_stamp=" << d_acq_sample_stamp
                              << " sample_counter=" << d_sample_counter
                              << " corr_len=" << d_correlation_length_samples;
#endif
                    //current_synchro_data.Tracking_sample_counter = d_sample_counter + static_cast<uint64_t>(samples_offset);
                    current_synchro_data.Tracking_sample_counter = this->nitems_read(0) + static_cast<uint64_t>(samples_offset);
                    current_synchro_data.fs = d_fs_in;
                    current_synchro_data.correlation_length_ms = 1;
                    *out[0] = std::move(current_synchro_data);
                    d_sample_counter += static_cast<uint64_t>(samples_offset);  // count for the processed samples
                    d_pull_in = false;
                    consume_each(samples_offset);  // shift input to perform alignment with local replica
                    return 1;
                }

            // ################# CARRIER WIPEOFF AND CORRELATORS ##############################
            // perform carrier wipe-off and compute Early, Prompt and Late correlation

            std::copy(in, in + d_correlation_length_samples, in_gpu);
            // cudaProfilerStart();  // 需要 profiling 时再打开（生产路径上不要每次 1 ms 调用）
            multicorrelator_gpu->Carrier_wipeoff_multicorrelator_resampler_cuda(static_cast<float>(d_rem_carrier_phase_rad),
                static_cast<float>(d_carrier_phase_step_rad),
                static_cast<float>(d_code_phase_step_chips),
                static_cast<float>(d_rem_code_phase_chips),
                d_correlation_length_samples, d_n_correlator_taps);
            // cudaProfilerStop();
            // std::cout<<"c_out[0]="<<d_correlator_outs[0]<<"c_out[1]="<<d_correlator_outs[1]<<"c_out[2]="<<d_correlator_outs[2]<< '\n';
            // Carrier_wipeoff_multicorrelator_resampler_cuda() 内部已经做了
            // cudaPeekAtLastError() + cudaStreamSynchronize(stream1)，结果对 host 可见，
            // 这里不再做全设备的 cudaDeviceSynchronize()（20 通道 x 1 kHz 开销可观）。
            const cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                LOG(WARNING) << "[GPU-SYNC] ch=" << d_channel
                             << " multicorrelator CUDA error: "
                             << cudaGetErrorString(err);
            }

            // UPDATE INTEGRATION TIME
            CURRENT_INTEGRATION_TIME_S = static_cast<double>(d_correlation_length_samples) / static_cast<double>(d_fs_in);

            // ################## PLL ##########################################################
            // Update PLL discriminator [rads/Ti -> Secs/Ti]
            carr_phase_error_secs_Ti = pll_cloop_two_quadrant_atan(d_correlator_outs[1]) / TWO_PI;  // prompt output
            // Carrier discriminator filter
            // NOTICE: The carrier loop filter includes the Carrier Doppler accumulator, as described in Kaplan
            // d_carrier_doppler_hz = d_acq_carrier_doppler_hz + carr_phase_error_filt_secs_ti/INTEGRATION_TIME;
            // Input [s/Ti] -> output [Hz]
            d_carrier_doppler_hz = d_carrier_loop_filter.get_carrier_error(0.0, carr_phase_error_secs_Ti, 0.001);
            // ============================================================
            // 诊断：GPU 版 Prompt 相关值
            // ============================================================
            // ============================================================
            // 诊断：PLL 环路滤波器输入输出
            // ============================================================
            #if GNSS_SDR_VERBOSE_DEBUG
            static int filter_dbg = 0;
            if (++filter_dbg % 1000 == 0)
            {
                if (GNSS_SDR_VERBOSE_DEBUG) std::cout << "[GPU-FILTER-OUT] ch=" << d_channel
                          << " carr_err_in=" << carr_phase_error_secs_Ti
                          << " doppler_out=" << d_carrier_doppler_hz
                          << " Ti=" << 0.001
                          << std::endl;
            }
            #endif

            // PLL to DLL assistance [Secs/Ti]
            d_pll_to_dll_assist_secs_Ti = (d_carrier_doppler_hz * CURRENT_INTEGRATION_TIME_S) / GPS_L1_FREQ_HZ;

            // ################## DLL ##########################################################
            // DLL discriminator
            code_error_chips_Ti = dll_nc_e_minus_l_normalized(d_correlator_outs[0], d_correlator_outs[2], d_early_late_spc_chips, 1.0);  // [chips/Ti]  // early and late
            #if 1
            // Code discriminator filter
            code_error_filt_chips = d_code_loop_filter.get_code_nco(code_error_chips_Ti);                      // input [chips/Ti] -> output [chips/second]
            #else
            code_error_filt_chips = d_code_loop_filter.apply(code_error_chips_Ti);
            #endif

            // ============================================================
            // 关键修正：d_code_freq_chips 只在这里算一次
            // 和 CPU 版 dll_pll_veml_tracking.cc 的 run_dll_pll() 对齐
            // ============================================================
            d_code_freq_chips = GPS_L1_CA_CODE_RATE_CPS - code_error_filt_chips
                                + (d_carrier_doppler_hz * GPS_L1_CA_CODE_RATE_CPS / GPS_L1_FREQ_HZ);

            code_error_filt_secs_Ti = code_error_filt_chips * CURRENT_INTEGRATION_TIME_S / d_code_freq_chips;  // [s/Ti]
            // DLL code error estimation [s/Ti]
            // TODO: PLL carrier aid to DLL is disabled. Re-enable it and measure performance
            dll_code_error_secs_Ti = -code_error_filt_secs_Ti + d_pll_to_dll_assist_secs_Ti;
            
            // ============================================================
            // 诊断：DLL 环路修正量
            // ============================================================

            // ################## CARRIER AND CODE NCO BUFFER ALIGNMENT #######################
            // keep alignment parameters for the next input buffer
            // Compute the next buffer length based in the new period of the PRN sequence and the code phase error estimation
            const double T_chip_seconds = 1 / d_code_freq_chips;
            const double T_prn_seconds = T_chip_seconds * GPS_L1_CA_CODE_LENGTH_CHIPS;
            const double T_prn_samples = T_prn_seconds * static_cast<double>(d_fs_in);
            const double K_blk_samples = T_prn_samples + d_rem_code_phase_samples - dll_code_error_secs_Ti * static_cast<double>(d_fs_in);
            //const double K_blk_samples = T_prn_samples + d_rem_code_phase_samples;

            d_correlation_length_samples = static_cast<int32_t>(std::round(K_blk_samples));                // round to a discrete samples
            d_rem_code_phase_samples = K_blk_samples - static_cast<double>(d_correlation_length_samples);  // rounding error < 1 sample

            // UPDATE REMNANT CARRIER PHASE
            CORRECTED_INTEGRATION_TIME_S = (static_cast<double>(d_correlation_length_samples) / static_cast<double>(d_fs_in));

            // remnant carrier phase [rad]
            d_carrier_phase_step_rad = TWO_PI * d_carrier_doppler_hz / static_cast<double>(d_fs_in);
            d_rem_carrier_phase_rad = fmod(d_rem_carrier_phase_rad + d_carrier_phase_step_rad * static_cast<double>(d_correlation_length_samples), TWO_PI);
            if (d_rem_carrier_phase_rad < 0.0)
                {
                    d_rem_carrier_phase_rad += TWO_PI;
                }
            // ============================================================
            // 诊断：残余载波相位和相位步进
            // ============================================================
            // UPDATE CARRIER PHASE ACCUULATOR
            // carrier phase accumulator prior to update the PLL estimators (accumulated carrier in this loop depends on the old estimations!)
            d_acc_carrier_phase_cycles -= d_carrier_doppler_hz * CORRECTED_INTEGRATION_TIME_S;

            // ################### DLL COMMANDS #################################################
            // code phase step (Code resampler phase increment per sample) [chips/sample]
            d_code_phase_step_chips = d_code_freq_chips / static_cast<double>(d_fs_in);
            // remnant code phase [chips]
            d_rem_code_phase_chips = d_rem_code_phase_samples * (d_code_freq_chips / static_cast<double>(d_fs_in));

            // ============================================================
            // 诊断：NCO 残余相位
            // ============================================================

            // ####### CN0 ESTIMATION AND LOCK DETECTORS #######################################
            // 与 CPU 版 dll_pll_veml_tracking::cn0_and_tracking_lock_status() 完全同口径：
            //   1) CN0 用 m2m4 估计 + 指数平滑；载波锁定量用"当前单个 prompt"的 cos(2*phi) + 指数平滑
            //      （原来用 cn0_samples 个 prompt 相干累加后再取比值，比特翻转会把 I/Q 和平均掉，
            //        指标退化成噪声，导致随机失锁）
            //   2) 载波锁失败与 CN0 过低各自独立计数，门限分别 max_carrier_lock_fail / max_code_lock_fail
            //   3) 缓冲满之后每 1 ms 评估一次（滑动窗），与 CPU 版的计数节奏一致
            if (d_cn0_estimation_counter < d_lock_cfg.cn0_samples)
                {
                    // fill buffer with prompt correlator output values
                    d_Prompt_buffer[d_cn0_estimation_counter] = d_correlator_outs[1];  // prompt
                    d_cn0_estimation_counter++;
                }
            else
                {
                    d_Prompt_buffer[d_cn0_estimation_counter % d_lock_cfg.cn0_samples] = d_correlator_outs[1];
                    d_cn0_estimation_counter++;

                    const float cn0_raw = cn0_m2m4_estimator(d_Prompt_buffer.data(), d_lock_cfg.cn0_samples,
                        static_cast<float>(GPS_L1_CA_CODE_PERIOD_S));
                    d_CN0_SNV_dB_Hz = d_cn0_smoother.smooth(cn0_raw);
                    d_carrier_lock_test = d_carrier_lock_test_smoother.smooth(carrier_lock_detector(d_Prompt_buffer.data(), 1));

                    if (d_pull_in_transitory == false)
                        {
                            if (d_carrier_lock_test < d_carrier_lock_threshold)
                                {
                                    d_carrier_lock_fail_counter++;
                                }
                            else if (d_carrier_lock_fail_counter > 0)
                                {
                                    d_carrier_lock_fail_counter--;
                                }

                            if (d_CN0_SNV_dB_Hz < d_lock_cfg.cn0_min)
                                {
                                    d_code_lock_fail_counter++;
                                }
                            else if (d_code_lock_fail_counter > 0)
                                {
                                    d_code_lock_fail_counter--;
                                }
                        }
                    else
                        {
                            d_carrier_lock_fail_counter = 0;
                            d_code_lock_fail_counter = 0;
                        }

                    if ((d_carrier_lock_fail_counter > d_lock_cfg.max_carrier_lock_fail) ||
                        (d_code_lock_fail_counter > d_lock_cfg.max_code_lock_fail))
                        {
                            std::cout << "Loss of lock in channel " << d_channel
                                      << " (carrier_lock_fail: " << d_carrier_lock_fail_counter
                                      << ", code_lock_fail: " << d_code_lock_fail_counter
                                      << ", lock_test: " << d_carrier_lock_test
                                      << ", CN0: " << d_CN0_SNV_dB_Hz << " dB-Hz)" << '\n';
#if GNSS_SDR_VERBOSE_DEBUG  
							LOG(INFO) << "[GPU-LOST] ch=" << d_channel
                                      << " carrier_lock_fail=" << d_carrier_lock_fail_counter
                                      << " code_lock_fail=" << d_code_lock_fail_counter
                                      << " lock_test=" << d_carrier_lock_test
                                      << " CN0=" << d_CN0_SNV_dB_Hz;
#endif
                            d_carrier_lock_fail_counter = 0;
                            d_code_lock_fail_counter = 0;
                            d_enable_tracking = false;  // TODO: check if disabling tracking is consistent with the channel state machine
                            d_pull_in = false;
                            d_pull_in_transitory = false;
                            loss_of_lock = true;
                            this->message_port_pub(pmt::mp("events"), pmt::from_long(3));
                        }
                    check_carrier_phase_coherent_initialization();

                    // 检测 Costas 环 180 度相位模糊（计数按通道保存）。
                    // 评估现在每 1 ms 一次（原来每 20 ms 一次），门限相应放大到 +-400，
                    // 保持约 0.4 s 的时间常数。
                    if (d_CN0_SNV_dB_Hz > 35.0)
                        {
                            if (d_correlator_outs[1].real() < 0.0)
                                {
                                    d_phase_180_counter++;
                                }
                            else
                                {
                                    d_phase_180_counter--;
                                }
                            if (d_phase_180_counter > 400)
                                {
                                    d_Flag_PLL_180_deg_phase_locked = true;
                                }
                            else if (d_phase_180_counter < -400)
                                {
                                    d_Flag_PLL_180_deg_phase_locked = false;
                                }
                        }
                }

            // ########### 输出跟踪结果给遥测解码器 ###########
            // 与 CPU 版 dll_pll_veml_tracking.cc 的 state 4 一致：
            //   - 先做导航比特同步，把累加窗对齐到比特边沿；
            //   - 之后每 d_symbols_per_bit 个 1 ms 符号累加出 1 个导航比特；
            //   - 只有整比特就绪时才置 Flag_valid_symbol_output = true。
            if (!d_bit_sync_locked)
                {
                    if ((d_pull_in_transitory == false) && (d_bit_sync != nullptr))
                        {
                            const bool lock_event = d_bit_sync->update(d_correlator_outs[1], !loss_of_lock);
                            if (lock_event)
                                {
                                    d_wait_for_bit_edge = true;
                                    const int64_t k_now = d_bit_sync->get_epoch_count() - 1;
                                    int wait = d_bit_sync->epochs_until_next_edge() - 1;
                                    if (wait < 0)
                                        {
                                            wait += d_bit_sync->bins();
                                        }
                                    d_bit_sync_target_epoch = k_now + wait;
                                }
                            if (d_wait_for_bit_edge && ((d_bit_sync->get_epoch_count() - 1) == d_bit_sync_target_epoch))
                                {
                                    // 到达预测的比特边沿：从这里重新开始累加
                                    d_wait_for_bit_edge = false;
                                    d_bit_sync_locked = true;
                                    d_P_accu = gr_complex(0, 0);
                                    d_P_accu_count = 0;
                                    LOG(INFO) << "GPU tracking: histogram bit synchronization locked in channel " << d_channel
                                              << " for satellite " << Gnss_Satellite(systemName[sys], d_acquisition_gnss_synchro->PRN);
                                    // 同时打到 stdout，方便像 CPU 版一样直接 grep run log
                                    std::cout << "GPU tracking bit synchronization locked in channel " << d_channel
                                              << " for satellite " << Gnss_Satellite(systemName[sys], d_acquisition_gnss_synchro->PRN) << '\n';
                                }
                        }
                    // 同步未完成：不发有效符号
                    current_synchro_data.Prompt_I = 0.0;
                    current_synchro_data.Prompt_Q = 0.0;
                    current_synchro_data.Flag_valid_symbol_output = false;
                }
            else
                {
                    // 从比特边沿起累加 d_symbols_per_bit 个 1 ms 符号 = 1 个导航比特
                    d_P_accu += d_correlator_outs[1];
                    d_P_accu_count++;
                    if (d_P_accu_count >= d_symbols_per_bit)
                        {
                            current_synchro_data.Prompt_I = static_cast<double>(d_P_accu.real());
                            current_synchro_data.Prompt_Q = static_cast<double>(d_P_accu.imag());
                            d_P_accu = gr_complex(0, 0);
                            d_P_accu_count = 0;
                            current_synchro_data.Flag_valid_symbol_output = !loss_of_lock;
                        }
                    else
                        {
                            current_synchro_data.Prompt_I = 0.0;
                            current_synchro_data.Prompt_Q = 0.0;
                            current_synchro_data.Flag_valid_symbol_output = false;
                        }
                }
            current_synchro_data.TOW_at_current_symbol_ms = d_tow_from_telemetry_ms;
            // CPU 版是在 consume_each() 之后取 nitems_read(0)（= 本块结束 / 下一块开始），
            // 这里在 consume 之前，显式加上块长，保持两版打点约定一致（消掉 +1 ms 公共项）。
            current_synchro_data.Tracking_sample_counter = this->nitems_read(0) + static_cast<uint64_t>(d_correlation_length_samples);
            current_synchro_data.Code_phase_samples = d_rem_code_phase_samples;
            current_synchro_data.Carrier_phase_rads = TWO_PI * d_acc_carrier_phase_cycles;
            current_synchro_data.Carrier_Doppler_hz = d_carrier_doppler_hz;
            current_synchro_data.CN0_dB_hz = d_CN0_SNV_dB_Hz;
            //current_synchro_data.Flag_valid_symbol_output = !loss_of_lock;
            current_synchro_data.Flag_PLL_180_deg_phase_locked = d_Flag_PLL_180_deg_phase_locked;
            current_synchro_data.correlation_length_ms = 1;
        }
    else
        {
            for (int32_t n = 0; n < d_n_correlator_taps; n++)
                {
                    d_correlator_outs[n] = gr_complex(0, 0);
                }

            current_synchro_data.System = {'G'};
            current_synchro_data.correlation_length_ms = 1;
            current_synchro_data.TOW_at_current_symbol_ms = d_tow_from_telemetry_ms;
            //current_synchro_data.Tracking_sample_counter = d_sample_counter + static_cast<uint64_t>(d_correlation_length_samples);
            current_synchro_data.Tracking_sample_counter = this->nitems_read(0) + static_cast<uint64_t>(d_correlation_length_samples);
        }

    // 只在整比特就绪（或失锁）时输出一个符号给遥测解码器，
    // 其余调用返回 0 且不写输出缓冲（与 CPU 版 dll_pll_veml_tracking.cc 一致）。
    const bool output_valid_symbol = (current_synchro_data.Flag_valid_symbol_output || loss_of_lock);
    current_synchro_data.fs = d_fs_in;
    if (output_valid_symbol)
        {
            *out[0] = std::move(current_synchro_data);
        }

    if (d_dump)
        {
            // MULTIPLEXED FILE RECORDING - Record results to file
            float prompt_I;
            float prompt_Q;
            float tmp_E, tmp_P, tmp_L;
            float tmp_VE = 0.0;
            float tmp_VL = 0.0;
            float tmp_float;
            prompt_I = d_correlator_outs[1].real();
            prompt_Q = d_correlator_outs[1].imag();
            tmp_E = std::abs<float>(d_correlator_outs[0]);
            tmp_P = std::abs<float>(d_correlator_outs[1]);
            tmp_L = std::abs<float>(d_correlator_outs[2]);
            try
                {
                    // Dump correlators output
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_VE), sizeof(float));
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_E), sizeof(float));
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_P), sizeof(float));
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_L), sizeof(float));
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_VL), sizeof(float));
                    // PROMPT I and Q (to analyze navigation symbols)
                    d_dump_file.write(reinterpret_cast<char *>(&prompt_I), sizeof(float));
                    d_dump_file.write(reinterpret_cast<char *>(&prompt_Q), sizeof(float));
                    // PRN start sample stamp
                    d_dump_file.write(reinterpret_cast<char *>(&d_sample_counter), sizeof(uint64_t));
                    // accumulated carrier phase
                    tmp_float = static_cast<float>(d_acc_carrier_phase_cycles * TWO_PI);
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_float), sizeof(float));
                    // carrier and code frequency
                    tmp_float = static_cast<float>(d_carrier_doppler_hz);
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_float), sizeof(float));
                    tmp_float = static_cast<float>(d_code_freq_chips);
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_float), sizeof(float));
                    // PLL commands
                    tmp_float = 1.0 / (carr_phase_error_secs_Ti * CURRENT_INTEGRATION_TIME_S);
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_float), sizeof(float));
                    tmp_float = 1.0 / (code_error_filt_secs_Ti * CURRENT_INTEGRATION_TIME_S);
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_float), sizeof(float));
                    // DLL commands
                    tmp_float = code_error_chips_Ti * CURRENT_INTEGRATION_TIME_S;
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_float), sizeof(float));
                    tmp_float = code_error_filt_secs_Ti;
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_float), sizeof(float));
                    // CN0 and carrier lock test
                    tmp_float = static_cast<float>(d_CN0_SNV_dB_Hz);
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_float), sizeof(float));
                    tmp_float = static_cast<float>(d_carrier_lock_test);
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_float), sizeof(float));
                    // AUX vars (for debug purposes)
                    tmp_float = code_error_chips_Ti * CURRENT_INTEGRATION_TIME_S;
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_float), sizeof(float));
                    double tmp_double = static_cast<double>(d_sample_counter + d_correlation_length_samples);
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_double), sizeof(double));
                    // PRN
                    uint32_t prn_ = d_acquisition_gnss_synchro->PRN;
                    d_dump_file.write(reinterpret_cast<char *>(&prn_), sizeof(uint32_t));
                }
            catch (const std::ofstream::failure *e)
                {
                    LOG(WARNING) << "Exception writing trk dump file " << e->what();
                }
        }

    consume_each(d_correlation_length_samples);        // this is necessary in gr::block derivates
    d_sample_counter += d_correlation_length_samples;  // count for the processed samples


    return output_valid_symbol ? 1 : 0;
}
