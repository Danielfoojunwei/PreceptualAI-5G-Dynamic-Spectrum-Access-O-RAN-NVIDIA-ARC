/*
 * Decode Horizon's E2SM-RC control payloads with FlexRIC's asn1c codec.
 *
 * G8 proves Horizon's Style 2 Action 6 message carries the RAN Parameter IDs
 * OCUDU declares, nested in the order OCUDU's parser walks, and that it
 * round-trips through *our* encoder (python asn1tools over the vendored O-RAN
 * ASN.1).
 *
 * Round-tripping through the encoder that produced the bytes proves the
 * encoder is self-consistent. It does not prove another implementation can
 * read them. This does: it hands the same bytes to FlexRIC's **asn1c-generated
 * C codec** — a different tool, a different language, generated from the same
 * ASN.1 source text — and prints the parameter tree it recovers.
 *
 * That is the interop claim that matters for G4. Horizon is a Non-RT RIC
 * rApp; in production the near-RT RIC owns the E2 codec, so the bytes we
 * construct have to be readable by somebody else's decoder or they are not
 * control messages, they are a private format.
 *
 * Reads the PER from stdin (raw octets). Prints the recovered tree as JSON on
 * stdout. Exit 0 on a successful decode, 1 otherwise.
 *
 * Build (see ocudu/e2/README.md for the exact link line):
 *   gcc -o flexric_decode_check flexric_decode_check.c \
 *       -I<flexric>/src -I<flexric>/build \
 *       <flexric>/build/src/sm/rc_sm/librc_sm.a ... -lm
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sm/rc_sm/ie/rc_data_ie.h"
#include "sm/rc_sm/dec/rc_dec_asn.h"
/* rc_data_ie.h only forward-declares the recursive value types (the header
 * comment calls the pointer indirection "bad design" — it is what avoids the
 * include cycle). Walking the tree needs the concrete definitions. */
#include "sm/rc_sm/ie/ir/ran_param_struct.h"
#include "sm/rc_sm/ie/ir/ran_param_list.h"
#include "sm/rc_sm/ie/ir/ran_parameter_value.h"

static void indent(int n)
{
  for (int i = 0; i < n; ++i) fputs("  ", stdout);
}

/* Walk the recovered value tree, printing the ids in visit order so the
 * output can be compared directly against horizon_ocudu.rc_slice_quota's
 * visit_order(). */
static void print_val(const ran_param_val_type_t* v, int depth);

static void print_seq(const seq_ran_param_t* p, int depth)
{
  indent(depth);
  printf("{\"id\": %u, \"value\": ", p->ran_param_id);
  print_val(&p->ran_param_val, depth);
  printf("}");
}

static void print_val(const ran_param_val_type_t* v, int depth)
{
  switch (v->type) {
    case ELEMENT_KEY_FLAG_TRUE_RAN_PARAMETER_VAL_TYPE:
    case ELEMENT_KEY_FLAG_FALSE_RAN_PARAMETER_VAL_TYPE: {
      const ran_parameter_value_t* e =
          v->type == ELEMENT_KEY_FLAG_TRUE_RAN_PARAMETER_VAL_TYPE ? v->flag_true
                                                                  : v->flag_false;
      switch (e->type) {
        case INTEGER_RAN_PARAMETER_VALUE:
          printf("{\"int\": %ld}", (long)e->int_ran);
          break;
        case OCTET_STRING_RAN_PARAMETER_VALUE: {
          printf("{\"octets\": \"");
          for (size_t i = 0; i < e->octet_str_ran.len; ++i)
            printf("%02x", e->octet_str_ran.buf[i]);
          printf("\"}");
          break;
        }
        case REAL_RAN_PARAMETER_VALUE:
          printf("{\"real\": %f}", e->real_ran);
          break;
        default:
          printf("{\"other_element_type\": %d}", (int)e->type);
      }
      break;
    }
    case STRUCTURE_RAN_PARAMETER_VAL_TYPE: {
      printf("{\"struct\": [\n");
      for (size_t i = 0; i < v->strct->sz_ran_param_struct; ++i) {
        print_seq(&v->strct->ran_param_struct[i], depth + 1);
        if (i + 1 < v->strct->sz_ran_param_struct) printf(",");
        printf("\n");
      }
      indent(depth);
      printf("]}");
      break;
    }
    case LIST_RAN_PARAMETER_VAL_TYPE: {
      printf("{\"list\": [\n");
      for (size_t i = 0; i < v->lst->sz_lst_ran_param; ++i) {
        /* lst_ran_param_t nests a ran_param_struct_t BY VALUE — the standard
         * gives a list item a RAN Parameter ID but the ASN.1 does not, and
         * FlexRIC follows the ASN.1. So there is no id to print here, and the
         * struct is reached through .ran_param_struct, not directly. */
        const ran_param_struct_t* s = &v->lst->lst_ran_param[i].ran_param_struct;
        indent(depth + 1);
        printf("{\"struct\": [\n");
        for (size_t j = 0; j < s->sz_ran_param_struct; ++j) {
          print_seq(&s->ran_param_struct[j], depth + 2);
          if (j + 1 < s->sz_ran_param_struct) printf(",");
          printf("\n");
        }
        indent(depth + 1);
        printf("]}");
        if (i + 1 < v->lst->sz_lst_ran_param) printf(",");
        printf("\n");
      }
      indent(depth);
      printf("]}");
      break;
    }
    default:
      printf("{\"unhandled_val_type\": %d}", (int)v->type);
  }
}

int main(void)
{
  uint8_t  buf[65536];
  size_t   len = fread(buf, 1, sizeof(buf), stdin);
  if (len == 0) {
    fprintf(stderr, "error: no input bytes on stdin\n");
    return 1;
  }
  fprintf(stderr, "decoding %zu octet(s) with FlexRIC's asn1c codec\n", len);

  e2sm_rc_ctrl_msg_t msg = rc_dec_ctrl_msg_asn(len, buf);

  if (msg.format != FORMAT_1_E2SM_RC_CTRL_MSG) {
    fprintf(stderr, "error: decoded format %d, expected Format 1\n", (int)msg.format);
    return 1;
  }

  /* A decode that recovers nothing is a FAILED decode, not an empty message.
   * FlexRIC's asn1c wrapper does not raise on a malformed buffer — flipping a
   * byte in the middle of a valid message yields format 1 with
   * sz_ran_param == 0 and no error. Without this guard the negative control
   * passed: a corrupted payload "decoded successfully" and exited 0. */
  if (msg.frmt_1.sz_ran_param == 0) {
    fprintf(stderr,
            "error: decoder recovered 0 RAN parameters from %zu octets — "
            "treating an empty recovery as a failed decode, not an empty "
            "message\n",
            len);
    return 1;
  }

  printf("{\n  \"format\": 1,\n  \"ran_parameters\": [\n");
  for (size_t i = 0; i < msg.frmt_1.sz_ran_param; ++i) {
    print_seq(&msg.frmt_1.ran_param[i], 2);
    if (i + 1 < msg.frmt_1.sz_ran_param) printf(",");
    printf("\n");
  }
  printf("  ]\n}\n");

  fprintf(stderr, "decoded %zu top-level RAN parameter(s)\n", msg.frmt_1.sz_ran_param);
  return 0;
}
