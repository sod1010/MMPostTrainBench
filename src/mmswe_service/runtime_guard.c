#define _GNU_SOURCE
/* Fixed trusted bootstrap. Static ELF: no loader hooks or instance Python.
 * No candidate instruction executes until restrictions AND host ACK succeed. */
#include <errno.h>
#include <grp.h>
#include <inttypes.h>
#include <linux/capability.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#define INSTANCE_CAPS UINT64_C(0xa00405fb)
static void fail(const char *reason) {
    dprintf(2, "MMPTB_GUARD_FAILED %s errno=%d\n", reason, errno);
    _exit(78);
}
static void pc(int op, unsigned long a, unsigned long b) {
    if (prctl(op, a, b, 0UL, 0UL) != 0) fail("prctl");
}
static uint64_t join(__u32 low, __u32 high) {
    return (uint64_t)low | ((uint64_t)high << 32);
}
int main(int argc, char **argv) {
    if (argc != 4) fail("arguments");
    int workload = strcmp(argv[1], "workload") == 0;
    int instance = strcmp(argv[1], "instance") == 0;
    int probe = strcmp(argv[2], "probe") == 0;
    if ((!workload && !instance) ||
        (!probe && !(workload && (!strcmp(argv[2], "train") || !strcmp(argv[2], "generate"))) &&
         !(instance && !strcmp(argv[2], "grade")))) fail("purpose");
    if (strlen(argv[3]) != 64 || strspn(argv[3], "0123456789abcdef") != 64) fail("nonce");
    if (getuid() || geteuid() || getgid() || getegid()) fail("initial-identity");
    pc(PR_SET_NO_NEW_PRIVS, 1, 0);
    pc(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL, 0);
    if (setgroups(0, NULL)) fail("clear-groups");
    uint64_t allowed = workload ? 0 : INSTANCE_CAPS, bounding = 0;
    int cap;
    for (cap = 0; cap < 64; cap++) {
        int present = prctl(PR_CAPBSET_READ, cap, 0UL, 0UL, 0UL);
        if (present < 0) {
            if (errno != EINVAL || cap <= CAP_MKNOD) fail("capability-range");
            break;
        }
        if (present && !(allowed & (UINT64_C(1) << cap))) pc(PR_CAPBSET_DROP, cap, 0);
    }
    if (cap == 64) fail("unsupported-capability-range");
    if (workload && (setresgid(65534, 65534, 65534) || setresuid(65534, 65534, 65534))) fail("drop-identity");
    struct __user_cap_header_struct header = {_LINUX_CAPABILITY_VERSION_3, 0};
    struct __user_cap_data_struct data[2] = {{0}};
    if (syscall(SYS_capget, &header, data)) fail("capget");
    for (int i = 0; i < 2; i++) {
        __u32 mask = (__u32)(allowed >> (32 * i));
        data[i].effective &= mask;
        data[i].permitted &= mask;
        data[i].inheritable = 0;
    }
    if (syscall(SYS_capset, &header, data) || syscall(SYS_capget, &header, data)) fail("capset");
    uint64_t eff = join(data[0].effective, data[1].effective);
    uint64_t prm = join(data[0].permitted, data[1].permitted);
    uint64_t inh = join(data[0].inheritable, data[1].inheritable), ambient = 0;
    for (int i = 0; i < cap; i++) {
        int b = prctl(PR_CAPBSET_READ, i, 0UL, 0UL, 0UL);
        int a = prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_IS_SET, i, 0UL, 0UL);
        if (b < 0 || a < 0) fail("verify-caps");
        if (b) bounding |= UINT64_C(1) << i;
        if (a) ambient |= UINT64_C(1) << i;
    }
    int nnp = prctl(PR_GET_NO_NEW_PRIVS, 0UL, 0UL, 0UL, 0UL);
    uid_t ruid, euid, suid; gid_t rgid, egid, sgid;
    if (getresuid(&ruid, &euid, &suid) || getresgid(&rgid, &egid, &sgid)) fail("verify-identity");
    uid_t want = workload ? 65534 : 0;
    if (nnp != 1 || ruid != want || euid != want || suid != want ||
        rgid != want || egid != want || sgid != want || getgroups(0, NULL) != 0 ||
        ((eff | prm | bounding) & ~allowed) || inh || ambient) fail("restriction-mismatch");
    umask(077);
    /* This is the only output before host approval. It cannot be forged by a
     * workload, because that workload has not executed yet. */
    dprintf(1, "{\"protocol\":1,\"kind\":\"%s\",\"purpose\":\"%s\",\"nonce\":\"%s\","
        "\"uid\":%u,\"gid\":%u,\"groups\":0,\"nnp\":1,\"caps\":{"
        "\"eff\":\"%016" PRIx64 "\",\"prm\":\"%016" PRIx64 "\",\"inh\":\"%016" PRIx64
        "\",\"bnd\":\"%016" PRIx64 "\",\"amb\":\"%016" PRIx64 "\"}}\n",
        argv[1], argv[2], argv[3], (unsigned)euid, (unsigned)egid, eff, prm, inh, bounding, ambient);
    char ack[69] = {0}, expected[69];
    snprintf(expected, sizeof(expected), "GO %s\n", argv[3]);
    size_t used = 0;
    while (used < sizeof(ack) - 1) {
        ssize_t n = read(0, ack + used, 1);
        if (n < 0 && errno == EINTR) continue;
        if (n != 1) fail("approval-eof");
        if (ack[used++] == '\n') break;
    }
    if (strcmp(ack, expected)) fail("approval-mismatch");
    unsetenv("BASH_ENV"); unsetenv("ENV"); unsetenv("LD_PRELOAD");
    if (probe) {
        const char *command = workload ?
          "set -eu; id; sed -n '/^Cap/p;/^NoNewPrivs:/p' /proc/self/status; echo MMPTB_GUARD_PROBE_FINISHED" :
          "set -eu; su nobody -s /bin/sh -c \"id; sed -n '/^Cap/p;/^NoNewPrivs:/p' /proc/self/status\"; echo MMPTB_GUARD_PROBE_FINISHED";
        execl("/bin/sh", "sh", "-c", command, (char *)NULL);
    } else if (workload) {
        execl("/bin/bash", "bash", "/opt/mmptb-service/entry.sh", argv[2], (char *)NULL);
    } else {
        execl("/bin/bash", "bash", "/run_in_chroot.sh", (char *)NULL);
    }
    fail("exec");
}
