/*
 * kpm_capture.c — capture REAL FlexRIC-encoded E2SM-KPM v3.00 indication PDUs.
 *
 * This harness reproduces the exact data path a live FlexRIC deployment runs
 * when the KPM monitor xApp (examples/xApp/c/monitor/xapp_kpm_moni.c) holds a
 * REPORT Style 4 subscription against the emulated gNB E2 agent
 * (examples/emulator/agent/emu_agent_gnb):
 *
 *   1. the xApp's Action Definition (Format 4, S-NSSAI == 1 matching
 *      condition, 3GPP TS 28.552 measurement names, 1000 ms granularity) is
 *      built with the same logic as xapp_kpm_moni.c fill_report_style_4();
 *   2. the emulator agent's OWN indication callback read_kpm_sm()
 *      (examples/emulator/agent/sm_kpm.c, linked unmodified) produces the
 *      indication header + message exactly as it does when the agent's
 *      subscription timer fires;
 *   3. FlexRIC's production wire encoder kpm_enc_ind_hdr_asn() /
 *      kpm_enc_ind_msg_asn() (asn1c-generated, ATS_ALIGNED_BASIC_PER)
 *      serialises them to the same bytes that would be carried in the E2AP
 *      RIC Indication's IndicationHeader / IndicationMessage OCTET STRINGs.
 *
 * The ONLY thing missing vs. a live run is the SCTP transport (unavailable
 * in this sandbox kernel); no PDU content is hand-crafted here.
 *
 * Output: indication_hdr.aper.bin and indication_msg.aper.bin in the CWD.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../flexric/examples/emulator/agent/sm_kpm.h"
#include "../flexric/src/sm/agent_if/read/sm_ag_if_rd.h"
#include "../flexric/src/sm/kpm_sm/kpm_sm_v03.00/enc/kpm_enc_asn.h"
#include "../flexric/src/util/byte_array.h"

static uint64_t const period_ms = 1000;

/* Mirror of xapp_kpm_moni.c filter_predicate(): S-NSSAI == 1 condition. */
static test_info_lst_t filter_predicate(test_cond_type_e type, test_cond_e cond, int value)
{
  test_info_lst_t dst = {0};

  dst.test_cond_type = type;
  dst.S_NSSAI = TRUE_TEST_COND_TYPE;

  dst.test_cond = calloc(1, sizeof(test_cond_e));
  assert(dst.test_cond != NULL && "Memory exhausted");
  *dst.test_cond = cond;

  dst.test_cond_value = calloc(1, sizeof(test_cond_value_t));
  assert(dst.test_cond_value != NULL && "Memory exhausted");
  dst.test_cond_value->type = OCTET_STRING_TEST_COND_VALUE;

  dst.test_cond_value->octet_string_value = calloc(1, sizeof(byte_array_t));
  assert(dst.test_cond_value->octet_string_value != NULL && "Memory exhausted");
  const size_t len_nssai = 1;
  dst.test_cond_value->octet_string_value->len = len_nssai;
  dst.test_cond_value->octet_string_value->buf = calloc(len_nssai, sizeof(uint8_t));
  assert(dst.test_cond_value->octet_string_value->buf != NULL && "Memory exhausted");
  dst.test_cond_value->octet_string_value->buf[0] = value;

  return dst;
}

/* Mirror of xapp_kpm_moni.c fill_kpm_label(): noLabel = true. */
static label_info_lst_t fill_kpm_label(void)
{
  label_info_lst_t label_item = {0};

  label_item.noLabel = calloc(1, sizeof(enum_value_e));
  assert(label_item.noLabel != NULL && "Memory exhausted");
  *label_item.noLabel = TRUE_ENUM_VALUE;

  return label_item;
}

/* The gNB-mono measurement set the emulator's RAN function advertises
 * (examples/emulator/agent/sm_kpm.c fill_kpm_report_style_4, NGRAN_GNB). */
static const char* kpm_meas[] = {
  "DRB.PdcpSduVolumeDL",
  "DRB.PdcpSduVolumeUL",
  "DRB.RlcSduDelayDl",
  "DRB.UEThpDl",
  "DRB.UEThpUl",
  "RRU.PrbTotDl",
  "RRU.PrbTotUl",
};

/* Mirror of xapp_kpm_moni.c fill_act_def_frm_1() + fill_report_style_4(). */
static kpm_act_def_t fill_report_style_4_act_def(void)
{
  kpm_act_def_t act_def = {.type = FORMAT_4_ACTION_DEFINITION};

  act_def.frm_4.matching_cond_lst_len = 1;
  act_def.frm_4.matching_cond_lst =
      calloc(act_def.frm_4.matching_cond_lst_len, sizeof(matching_condition_format_4_lst_t));
  assert(act_def.frm_4.matching_cond_lst != NULL && "Memory exhausted");
  act_def.frm_4.matching_cond_lst[0].test_info_lst =
      filter_predicate(S_NSSAI_TEST_COND_TYPE, EQUAL_TEST_COND, 1);

  kpm_act_def_format_1_t* ad = &act_def.frm_4.action_def_format_1;
  size_t const sz = sizeof(kpm_meas) / sizeof(kpm_meas[0]);
  ad->meas_info_lst_len = sz;
  ad->meas_info_lst = calloc(sz, sizeof(meas_info_format_1_lst_t));
  assert(ad->meas_info_lst != NULL && "Memory exhausted");
  for (size_t i = 0; i < sz; i++) {
    meas_info_format_1_lst_t* item = &ad->meas_info_lst[i];
    item->meas_type.type = NAME_MEAS_TYPE;
    item->meas_type.name = cp_str_to_ba(kpm_meas[i]);
    item->label_info_lst_len = 1;
    item->label_info_lst = calloc(1, sizeof(label_info_lst_t));
    assert(item->label_info_lst != NULL && "Memory exhausted");
    item->label_info_lst[0] = fill_kpm_label();
  }
  ad->gran_period_ms = period_ms;
  ad->cell_global_id = NULL;
  ad->meas_bin_range_info_lst_len = 0;
  ad->meas_bin_info_lst = NULL;

  return act_def;
}

static void dump(const char* path, byte_array_t ba)
{
  FILE* f = fopen(path, "wb");
  assert(f != NULL && "cannot open output file");
  size_t n = fwrite(ba.buf, 1, ba.len, f);
  assert(n == ba.len);
  fclose(f);
  printf("wrote %zu bytes -> %s\n", ba.len, path);
}

int main(int argc, char** argv)
{
  unsigned seed = (argc > 1) ? (unsigned)atoi(argv[1]) : (unsigned)time(NULL);
  srand(seed);
  printf("kpm_capture: srand seed = %u\n", seed);

  /* Emulator agent init: registers the TS 28.552 measurement fill table. */
  init_kpm_sm();

  kpm_act_def_t act_def = fill_report_style_4_act_def();

  /* The agent's real indication callback, exactly as fired on subscription. */
  kpm_rd_ind_data_t rd = {0};
  rd.act_def = &act_def;
  bool ok = read_kpm_sm(&rd);
  assert(ok && "emulator read_kpm_sm produced no indication");

  /* FlexRIC's production ASN.1 APER wire encoder. */
  byte_array_t ba_hdr = kpm_enc_ind_hdr_asn(&rd.ind.hdr);
  byte_array_t ba_msg = kpm_enc_ind_msg_asn(&rd.ind.msg);

  dump("indication_hdr.aper.bin", ba_hdr);
  dump("indication_msg.aper.bin", ba_msg);

  printf("indication format: %d (3 == Format 3, UE-level Style 4)\n", rd.ind.msg.type);
  return 0;
}
