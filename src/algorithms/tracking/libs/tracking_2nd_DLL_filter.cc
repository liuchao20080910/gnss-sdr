/*!
 * \file tracking_2nd_DLL_filter.cc
 * \brief Implementation of a 2nd order DLL filter for code tracking loop.
 * \author Javier Arribas, 2011. jarribas(at)cttc.es
 *
 * Class that implements 2 order PLL filter for code tracking loop.
 * The algorithm is described in :
 * K.Borre, D.M.Akos, N.Bertelsen, P.Rinder, and S.~H.~Jensen, A Software-Defined
 * GPS and Galileo Receiver. A Single-Frequency Approach,
 * Birkhauser, 2007, Applied and Numerical Harmonic Analysis.
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


#include "tracking_2nd_DLL_filter.h"
#include <iostream>

// 调试打印总开关：0 = 关闭（生产默认，编译器会把整段消掉）；1 = 打开
#ifndef GNSS_SDR_VERBOSE_DEBUG
#define GNSS_SDR_VERBOSE_DEBUG 0
#endif


void Tracking_2nd_DLL_filter::calculate_lopp_coef(float* tau1, float* tau2, float lbw, float zeta, float k)
{
    // Solve natural frequency
    const float Wn = lbw * 8.0F * zeta / (4.0F * zeta * zeta + 1.0F);
    // solve for t1 & t2
    *tau1 = k / (Wn * Wn);
    //*tau1 = k / (Wn * Wn) / 740.0;  // 让积分项系数增大 740 倍
    *tau2 = 2.0F * zeta / Wn;
}


void Tracking_2nd_DLL_filter::set_DLL_BW(float dll_bw_hz)
{
    // Calculate filter coefficient values
    d_dllnoisebandwidth = dll_bw_hz;
    calculate_lopp_coef(&d_tau1_code, &d_tau2_code, d_dllnoisebandwidth, d_dlldampingratio, 1.0);  // Calculate filter coefficient values
}


void Tracking_2nd_DLL_filter::initialize()
{
    // code tracking loop parameters
    d_old_code_nco = 0.0;
    d_old_code_error = 0.0;
}


float Tracking_2nd_DLL_filter::get_code_nco(float DLL_discriminator)
{
    float prop_term = (d_tau2_code / d_tau1_code) * (DLL_discriminator - d_old_code_error);
    float integ_term = (DLL_discriminator + d_old_code_error) * (d_pdi_code / (2.0F * d_tau1_code));
    float code_nco = d_old_code_nco + prop_term + integ_term;
#if GNSS_SDR_VERBOSE_DEBUG
    static int dll_dbg = 0;
    if (++dll_dbg % 1000 == 0)
    {
        if (GNSS_SDR_VERBOSE_DEBUG) std::cout << "[DLL-NCO] DLL_discriminator=" << DLL_discriminator
                  << " prop_term=" << prop_term
                  << " integ_term=" << integ_term
                  << " code_nco=" << code_nco
                  << std::endl;
    }
#endif
    d_old_code_nco = code_nco;
    d_old_code_error = DLL_discriminator;  // [chips]
    return code_nco;
}


Tracking_2nd_DLL_filter::Tracking_2nd_DLL_filter(float pdi_code) : d_pdi_code(pdi_code),
                                                                   d_dlldampingratio(0.7)
{
}


Tracking_2nd_DLL_filter::Tracking_2nd_DLL_filter() : d_pdi_code(0.001),
                                                     d_dlldampingratio(0.7)
{
}


void Tracking_2nd_DLL_filter::set_pdi(float pdi_code)
{
    d_pdi_code = pdi_code;  // Summation interval for code
}
