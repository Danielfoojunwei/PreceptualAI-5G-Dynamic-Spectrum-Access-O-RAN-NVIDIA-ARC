/*
 * sctp_udp_shim.c — LD_PRELOAD transport substitute for FlexRIC in kernels
 * without SCTP support.
 *
 * This sandbox kernel has no SCTP (socket(AF_INET, SOCK_SEQPACKET,
 * IPPROTO_SCTP) fails with EPROTONOSUPPORT and no modules are available).
 * FlexRIC's entire message IO funnels through one-to-many SCTP semantics:
 * every send/receive passes the peer's sockaddr_in explicitly
 * (src/lib/ep/e2ap_ep.c: sctp_sendmsg / sctp_recvmsg), the server never
 * calls accept(), and the client never calls connect(). Those are exactly
 * UDP datagram semantics, so this shim maps:
 *
 *   socket(AF_INET, SOCK_SEQPACKET, IPPROTO_SCTP) -> socket(AF_INET, SOCK_DGRAM, 0)
 *   setsockopt(fd, IPPROTO_SCTP, ...)             -> success no-op
 *   listen(fd, n)                                 -> success no-op
 *   sctp_sendmsg(fd, buf, len, to, ...)           -> sendto(fd, buf, len, 0, to, ...)
 *   sctp_recvmsg(fd, buf, len, from, ...)         -> recvfrom(fd, buf, len, 0, from, ...)
 *                                                    (sinfo zeroed, *msg_flags = MSG_EOR)
 *
 * EVERY E2AP/E42AP/E2SM byte FlexRIC produces and parses is untouched; only
 * the L4 transport differs (loopback UDP datagrams instead of SCTP, both
 * preserving message boundaries). This is a disclosed transport
 * substitution for sandbox use — NOT standards-conformant E2 transport.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/sctp.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define MAX_FDS 4096
static unsigned char shimmed[MAX_FDS];

static int (*real_socket)(int, int, int);
static int (*real_setsockopt)(int, int, int, const void*, socklen_t);
static int (*real_listen)(int, int);
static int (*real_close)(int);

static void init_reals(void)
{
  if (!real_socket) {
    real_socket = dlsym(RTLD_NEXT, "socket");
    real_setsockopt = dlsym(RTLD_NEXT, "setsockopt");
    real_listen = dlsym(RTLD_NEXT, "listen");
    real_close = dlsym(RTLD_NEXT, "close");
  }
}

int socket(int domain, int type, int protocol)
{
  init_reals();
  if (domain == AF_INET && type == SOCK_SEQPACKET && protocol == IPPROTO_SCTP) {
    int fd = real_socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (fd >= 0 && fd < MAX_FDS) {
      shimmed[fd] = 1;
      int sz = 4 * 1024 * 1024;
      real_setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &sz, sizeof(sz));
      fprintf(stderr, "[sctp-udp-shim] SCTP socket request -> UDP fd %d\n", fd);
    }
    return fd;
  }
  return real_socket(domain, type, protocol);
}

int setsockopt(int fd, int level, int optname, const void* optval, socklen_t optlen)
{
  init_reals();
  if (level == IPPROTO_SCTP && fd >= 0 && fd < MAX_FDS && shimmed[fd])
    return 0; /* SCTP_EVENTS / SCTP_AUTOCLOSE / SCTP_NODELAY: no-op on UDP */
  return real_setsockopt(fd, level, optname, optval, optlen);
}

int listen(int fd, int backlog)
{
  init_reals();
  if (fd >= 0 && fd < MAX_FDS && shimmed[fd])
    return 0; /* connectionless: nothing to listen for */
  return real_listen(fd, backlog);
}

int close(int fd)
{
  init_reals();
  if (fd >= 0 && fd < MAX_FDS)
    shimmed[fd] = 0;
  return real_close(fd);
}

int sctp_sendmsg(int sd, const void* msg, size_t len, struct sockaddr* to,
                 socklen_t tolen, uint32_t ppid, uint32_t flags,
                 uint16_t stream_no, uint32_t timetolive, uint32_t context)
{
  (void)ppid; (void)flags; (void)stream_no; (void)timetolive; (void)context;
  return (int)sendto(sd, msg, len, 0, to, tolen);
}

int sctp_recvmsg(int sd, void* msg, size_t len, struct sockaddr* from,
                 socklen_t* fromlen, struct sctp_sndrcvinfo* sinfo, int* msg_flags)
{
  ssize_t rc = recvfrom(sd, msg, len, 0, from, fromlen);
  if (rc >= 0) {
    if (sinfo)
      memset(sinfo, 0, sizeof(*sinfo));
    if (msg_flags)
      *msg_flags = MSG_EOR; /* one datagram == one complete message */
  }
  return (int)rc;
}
