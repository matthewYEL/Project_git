/*
 * Card recognition on the DE10-Nano, under Linux, across both Cortex-A9 cores.
 *
 * This is the Linux counterpart of the bare-metal atlas_main.c. Same FPGA
 * bitstream, same accelerator, same arithmetic (card_cnn_hw.h) -- what changes
 * is that the HPS now runs Linux and the work is split across the two cores of
 * the Cortex-A9 MPCore as two pthreads with hard CPU affinity:
 *
 *   CPU0  vision thread   /dev/mem -> LW bridge: trigger the camera, read the
 *                         96x96 snapshot, scale to Q6.10, upload it, start the
 *                         CNN, poll for the result, decode rank/suit/joker.
 *                         I/O bound -- ~64,500 uncached bridge accesses a frame.
 *
 *   CPU1  game thread     Blackjack state machine over the recognised cards,
 *                         plus a Monte-Carlo odds engine that re-estimates the
 *                         bust/win probabilities after every committed card.
 *                         Compute bound, and deliberately so: it gives CPU1
 *                         measurable work rather than a token thread.
 *
 * The two are decoupled by a small lock-protected ring buffer, so neither
 * stalls the other: the camera keeps running at full rate while the odds
 * engine is busy, which is the whole point of using both cores.
 *
 * Evidence that both cores really are working is printed, not asserted: each
 * thread reports the CPU it is running on (sched_getcpu()) and its own consumed
 * CPU time (CLOCK_THREAD_CPUTIME_ID), and --stats adds a per-core utilisation
 * read straight from /proc/stat.
 *
 * Build (natively, on the board):  make
 * Run (needs /dev/mem, so root):   sudo ./card_cnn --stats
 */
#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "card_cnn_hw.h"

/* ---- tunables ---------------------------------------------------------- */
#define VISION_CPU        0
#define GAME_CPU          1
#define RING_SLOTS        16
#define STABLE_FRAMES     3        /* identical reads needed before a card counts */
#define MC_TRIALS         200000   /* Monte-Carlo trials per odds update */
#define DEALER_STANDS_ON  17
#define PLAYER_AUTO_STAND 17

/* ---- shared state ------------------------------------------------------ */

struct card {
    int      rank;      /* 0..12 -> 2..A, or -1 for a joker */
    int      suit;      /* 0..3 */
    int      joker;
    unsigned frame;
    int      score;     /* winning rank logit, Q6.10 */
};

struct ring {
    struct card      slot[RING_SLOTS];
    unsigned         head, tail;
    pthread_mutex_t  lock;
    pthread_cond_t   notempty;
};

static struct ring g_ring;
static volatile sig_atomic_t g_stop;

static pthread_mutex_t g_print_lock = PTHREAD_MUTEX_INITIALIZER;

/* per-thread accounting, published for the stats line */
static struct {
    volatile unsigned frames;       /* vision: inferences completed */
    volatile unsigned timeouts;
    volatile unsigned cards;        /* game: cards committed */
    volatile unsigned mc_runs;      /* game: odds evaluations */
    volatile int      vision_cpu;   /* sched_getcpu() as actually observed */
    volatile int      game_cpu;
    volatile double   vision_cpu_s;
    volatile double   game_cpu_s;
} g_stat;

static volatile uint8_t *g_lw;      /* mmap'd lightweight bridge */
static int g_opt_stats, g_opt_rt, g_opt_quiet, g_opt_fake;

/* ---- small helpers ----------------------------------------------------- */

static void say(const char *fmt, ...)
{
    va_list ap;

    pthread_mutex_lock(&g_print_lock);
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    fflush(stdout);
    pthread_mutex_unlock(&g_print_lock);
}

static inline void reg_write(unsigned off, uint32_t v)
{
    *(volatile uint32_t *)(g_lw + off) = v;
}

static inline uint32_t reg_read(unsigned off)
{
    return *(volatile uint32_t *)(g_lw + off);
}

/* The snapshot RAM read is registered, so the first access after moving the
 * address returns the previous cell. Same dummy read as atlas_main.c. */
static unsigned snap_read(unsigned addr)
{
    reg_write(REG_SNAPSHOT_ADDR, addr);
    (void)reg_read(REG_SNAPSHOT_DATA);
    return reg_read(REG_SNAPSHOT_DATA) & 0xFFFFu;
}

static double now_s(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double thread_cpu_s(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void on_signal(int sig)
{
    (void)sig;
    g_stop = 1;
}

/* ---- CPU affinity ------------------------------------------------------ */

/* Pin the calling thread to one core and prove it took: the demo claim is
 * "these two things run on different cores", so a silent failure here would
 * quietly invalidate it. */
static int pin_to_cpu(int cpu, const char *who)
{
    cpu_set_t set;
    int rc, got;

    CPU_ZERO(&set);
    CPU_SET(cpu, &set);

    rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (rc != 0) {
        say("!! %s: pthread_setaffinity_np(cpu%d) failed: %s\n",
            who, cpu, strerror(rc));
        return -1;
    }

    sched_yield();                 /* let the scheduler act on it before asking */
    got = sched_getcpu();
    if (got != cpu) {
        say("!! %s: asked for cpu%d but running on cpu%d\n", who, cpu, got);
        return -1;
    }
    return 0;
}

static void maybe_go_realtime(const char *who, int prio)
{
    struct sched_param sp;

    if (!g_opt_rt)
        return;

    memset(&sp, 0, sizeof(sp));
    sp.sched_priority = prio;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0)
        say("   %s: SCHED_FIFO unavailable (%s), staying with SCHED_OTHER\n",
            who, strerror(errno));
    else
        say("   %s: SCHED_FIFO priority %d\n", who, prio);
}

/* ---- ring buffer ------------------------------------------------------- */

static void ring_init(struct ring *r)
{
    memset(r, 0, sizeof(*r));
    pthread_mutex_init(&r->lock, NULL);
    pthread_cond_init(&r->notempty, NULL);
}

static void ring_push(struct ring *r, const struct card *c)
{
    pthread_mutex_lock(&r->lock);
    if (r->head - r->tail < RING_SLOTS) {
        r->slot[r->head % RING_SLOTS] = *c;
        r->head++;
        pthread_cond_signal(&r->notempty);
    }
    /* Full means the game thread is behind; dropping is correct here -- the
     * camera should never stall waiting for the odds engine. */
    pthread_mutex_unlock(&r->lock);
}

/* Blocks with a timeout so the game thread still wakes to notice g_stop. */
static int ring_pop(struct ring *r, struct card *out, int timeout_ms)
{
    struct timespec ts;
    int got = 0;

    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_nsec += (long)timeout_ms * 1000000L;
    ts.tv_sec  += ts.tv_nsec / 1000000000L;
    ts.tv_nsec %= 1000000000L;

    pthread_mutex_lock(&r->lock);
    while (r->head == r->tail && !g_stop) {
        if (pthread_cond_timedwait(&r->notempty, &r->lock, &ts) == ETIMEDOUT)
            break;
    }
    if (r->head != r->tail) {
        *out = r->slot[r->tail % RING_SLOTS];
        r->tail++;
        got = 1;
    }
    pthread_mutex_unlock(&r->lock);
    return got;
}

/* ---- FPGA pipeline (CPU0) ---------------------------------------------- */

static int capture_once(void)
{
    int i;

    reg_write(REG_CAMERA_TRIGGER, 1);
    for (i = 0; i < 200; i++)
        (void)reg_read(REG_CNN_RESULT);
    reg_write(REG_CAMERA_TRIGGER, 0);

    /* snapshot_done is left high by the previous capture: wait for it to drop
     * (trigger accepted), then rise again (a new frame landed). */
    for (i = 0; i < 100000 && RES_SNAP_DONE(reg_read(REG_CNN_RESULT)); i++)
        ;
    for (i = 0; i < 4000000; i++)
        if (RES_SNAP_DONE(reg_read(REG_CNN_RESULT)))
            return 1;
    return 0;
}

static void read_snapshot(uint16_t *raw, unsigned *minv, unsigned *maxv)
{
    unsigned lo = 0xFFFFu, hi = 0;
    int i;

    for (i = 0; i < IMG_PIXELS; i++) {
        raw[i] = (uint16_t)snap_read((unsigned)i);
        if (raw[i] < lo) lo = raw[i];
        if (raw[i] > hi) hi = raw[i];
    }
    *minv = lo;
    *maxv = hi;
}

static unsigned upload_and_infer(const uint16_t *q, unsigned red_count)
{
    unsigned ctrl = 0x4u | ((red_count > RED_THRESH_SW) ? 0x8u : 0x0u);
    int i;

    for (i = 0; i < IMG_PIXELS; i++) {
        reg_write(REG_IMG_WR_ADDR, (unsigned)i);
        reg_write(REG_IMG_WR_DATA, q[i]);
        reg_write(REG_IMG_WR_CTRL, ctrl | 0x1u);   /* wr_en high for >= 1 clock */
        reg_write(REG_IMG_WR_CTRL, ctrl);
    }

    reg_write(REG_CNN_START, 1);       /* edge-detected; clears sticky done */
    for (i = 0; i < 50; i++)
        (void)reg_read(REG_CNN_RESULT);
    reg_write(REG_CNN_START, 0);

    for (i = 0; i < 20000000; i++) {
        unsigned r = reg_read(REG_CNN_RESULT);
        if (RES_DONE(r))
            return r;
    }
    return 0;
}

static void card_name(const struct card *c, char *buf, size_t n)
{
    if (c->joker)
        snprintf(buf, n, "JOKER");
    else if (c->rank >= 0 && c->rank < RANK_COUNT)
        snprintf(buf, n, "%s of %s", RANK_NAMES[c->rank], SUIT_NAMES[c->suit]);
    else
        snprintf(buf, n, "bad rank %d", c->rank);
}

static void check_camera(void)
{
    unsigned st   = snap_read(SNAP_STATUS);
    unsigned vs0  = snap_read(SNAP_VS_COUNT);
    unsigned pix0 = snap_read(SNAP_PIXCLK);
    unsigned vs1, pix1;

    say("camera: mipi_cfg=%u cam_cfg=%u vs=%u pixclk=%u retries=%u "
        "(audpll=%u hdmi=%u)\n",
        ST_MIPI_REL(st), ST_CAM_REL(st), vs0, pix0,
        snap_read(SNAP_RETRIES), ST_AUDPLL_OK(st), ST_HDMI_RDY(st));

    usleep(50000);
    vs1  = snap_read(SNAP_VS_COUNT);
    pix1 = snap_read(SNAP_PIXCLK);

    if (vs1 == vs0) {
        say("\n*** WARNING: no MIPI frames (vs_count stuck at %u) ***\n", vs1);
        if (pix1 == pix0)
            say("    no pixel clock either -- reseat the D8M ribbon.\n");
        else
            say("    pixel clock runs but no framing -- wrong lane count/format.\n");
        say("    Every prediction below is the network's answer to a blank "
            "image.\n\n");
    }
}

/*
 * --fake: synthesise the result word the accelerator would have returned, so
 * the two-core split, the debounce, the ring buffer and the game can be
 * exercised (and the demo rehearsed) with no board attached. It deals a fixed
 * sequence, holding each card long enough to clear STABLE_FRAMES and dropping
 * a blank frame between cards to re-arm the debounce.
 */
static unsigned fake_result(unsigned frame, unsigned *red_count, unsigned *maxv)
{
    /* rank index 0..12 = 2..10,J,Q,K,A; suit 0..3 = S,C,H,D */
    static const struct { int rank, suit, joker; } deal[] = {
        {  6, 0, 0 },   /* 8 of Spades   */
        {  3, 2, 0 },   /* 5 of Hearts   */
        { 12, 1, 0 },   /* A of Clubs    */
        {  9, 3, 0 },   /* J of Diamonds */
        {  0, 0, 1 },   /* joker: reset  */
        { 11, 2, 0 },   /* K of Hearts   */
        {  8, 1, 0 },   /* 10 of Clubs   */
    };
    const unsigned hold  = STABLE_FRAMES + 2;
    const unsigned cycle = hold + 1;                 /* + one blank frame */
    unsigned idx  = (frame / cycle) % (sizeof(deal) / sizeof(deal[0]));
    unsigned step = frame % cycle;
    unsigned r;

    usleep(300000);                                  /* ~3 frames a second, as 96x96 inference runs */

    if (step == hold) {                              /* blank frame */
        *maxv = 0;
        *red_count = 0;
        return (1u << 7);
    }

    /* Plausible at the 96x96 scale: cell sums top out at SUM_MAX 510, and the
     * red count is over an 18,432-pixel crop against RED_THRESH_SW 1600. */
    *maxv = 480;
    *red_count = (deal[idx].suit >= 2) ? 3000u : 250u;

    r  = (unsigned)deal[idx].rank & 0xFu;
    r |= ((unsigned)deal[idx].suit & 0x3u) << 4;
    r |= (unsigned)(deal[idx].joker != 0) << 6;
    r |= 1u << 7;                                    /* done */
    r |= (deal[idx].suit >= 2 ? 1u : 0u) << 8;       /* colour used */
    r |= 420u << 16;                                 /* plausible rank logit */
    return r;
}

static void *vision_thread(void *arg)
{
    static uint16_t raw[IMG_PIXELS];
    static uint16_t q[IMG_PIXELS];

    struct card last = { -2, -2, 0, 0, 0 };
    int      stable = 0, emitted = 0;
    unsigned frame = 0;

    (void)arg;
    pthread_setname_np(pthread_self(), "vision");

    pin_to_cpu(VISION_CPU, "vision");
    maybe_go_realtime("vision", 40);
    g_stat.vision_cpu = sched_getcpu();

    say("[vision] running on CPU%d -- camera + CNN accelerator\n",
        g_stat.vision_cpu);

    while (!g_stop) {
        unsigned minv = 0, maxv, red_count, r;
        struct card c;
        char name[48];

        if (g_opt_fake) {
            r = fake_result(frame, &red_count, &maxv);
        } else {
            if (!capture_once()) {
                g_stat.timeouts++;
                if (!g_opt_quiet)
                    say("[vision] snapshot timeout -- camera not streaming\n");
                usleep(100000);
                continue;
            }

            read_snapshot(raw, &minv, &maxv);
            red_count = snap_read(SNAP_RED_COUNT);

            ccp_preprocess(raw, q);
            r = upload_and_infer(q, red_count);
        }

        g_stat.vision_cpu   = sched_getcpu();
        g_stat.vision_cpu_s = thread_cpu_s();

        if (!r) {
            g_stat.timeouts++;
            if (!g_opt_quiet)
                say("[vision] inference timeout\n");
            continue;
        }

        g_stat.frames = ++frame;

        c.joker = (int)RES_JOKER(r);
        c.rank  = c.joker ? -1 : (int)RES_RANK(r);
        c.suit  = (int)RES_SUIT(r);
        c.score = RES_SCORE(r);
        c.frame = frame;

        if (maxv == 0) {
            /* Blank frame: no card in view. Also re-arms the debounce, so the
             * same card can be presented twice in a row. */
            stable  = 0;
            emitted = 0;
            last.rank = -2;
            continue;
        }

        if (c.rank == last.rank && c.suit == last.suit && c.joker == last.joker) {
            stable++;
        } else {
            last    = c;
            stable  = 1;
            emitted = 0;
        }

        card_name(&c, name, sizeof(name));
        if (!g_opt_quiet)
            say("[vision cpu%d] %-16s logit %5d  red %5u->%-5s  "
                "min %4u max %4u  stable %d\n",
                g_stat.vision_cpu, name, c.score, red_count,
                RES_COLOUR(r) ? "red" : "black", minv, maxv, stable);

        if (stable >= STABLE_FRAMES && !emitted) {
            emitted = 1;
            ring_push(&g_ring, &c);
        }
    }

    say("[vision] stopping after %u frames (%u timeouts)\n",
        g_stat.frames, g_stat.timeouts);
    return NULL;
}

/* ---- blackjack + odds engine (CPU1) ------------------------------------ */

/* rank index 0..12 = 2..10,J,Q,K,A -> blackjack value, aces counted high */
static int card_value(int rank)
{
    if (rank < 0)   return 0;         /* joker */
    if (rank <= 8)  return rank + 2;  /* 2..10 */
    if (rank <= 11) return 10;        /* J,Q,K */
    return 11;                        /* A */
}

static int hand_total(const int *ranks, int n, int *soft_out)
{
    int total = 0, aces = 0, i;

    for (i = 0; i < n; i++) {
        total += card_value(ranks[i]);
        if (ranks[i] == 12)
            aces++;
    }
    while (total > 21 && aces > 0) {
        total -= 10;
        aces--;
    }
    if (soft_out)
        *soft_out = aces;
    return total;
}

/* xorshift32: the odds engine needs a lot of cheap randomness and must not use
 * rand(), whose state would be shared with the other thread. */
static uint32_t rng_state = 0x1234567u;

static inline uint32_t rng_next(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

/*
 * Monte-Carlo estimate of what happens next, given the cards already on the
 * table. This is the deliberate CPU1 workload: MC_TRIALS deals per update.
 *
 * Cards already seen are removed from the shoe, so the estimate reflects the
 * remaining deck rather than an idealised one.
 */
static void estimate_odds(const int *player, int np, const int *seen_count,
                          double *p_bust, double *p_win_stand)
{
    int bust = 0, win = 0, t;
    int total = hand_total(player, np, NULL);

    for (t = 0; t < MC_TRIALS; t++) {
        int shoe[13], i, drawn, dealer, guard;

        for (i = 0; i < 13; i++)
            shoe[i] = 4 - seen_count[i];

        /* one more card for the player */
        drawn = -1;
        for (guard = 0; guard < 64 && drawn < 0; guard++) {
            int pick = (int)(rng_next() % 13);
            if (shoe[pick] > 0) {
                shoe[pick]--;
                drawn = pick;
            }
        }
        if (drawn < 0)
            continue;

        {
            int hand[16], n = 0;
            for (i = 0; i < np && n < 15; i++)
                hand[n++] = player[i];
            hand[n++] = drawn;
            if (hand_total(hand, n, NULL) > 21)
                bust++;
        }

        /* dealer plays out against the player's current standing total */
        {
            int dhand[16], dn = 0;
            for (guard = 0; guard < 32; guard++) {
                int pick = (int)(rng_next() % 13);
                if (shoe[pick] <= 0)
                    continue;
                shoe[pick]--;
                if (dn < 15)
                    dhand[dn++] = pick;
                if (dn >= 2 && hand_total(dhand, dn, NULL) >= DEALER_STANDS_ON)
                    break;
            }
            dealer = hand_total(dhand, dn, NULL);
            if (total <= 21 && (dealer > 21 || total > dealer))
                win++;
        }
    }

    *p_bust      = (double)bust / MC_TRIALS;
    *p_win_stand = (double)win  / MC_TRIALS;
}

static void new_round(int *np, int *seen_count, int *round)
{
    *np = 0;
    memset(seen_count, 0, 13 * sizeof(int));
    (*round)++;
    say("\n--- Round %d: show the dealer a card ---\n", *round);
}

static void *game_thread(void *arg)
{
    int player[16], np = 0;
    int seen_count[13];
    int round = 1;

    (void)arg;
    memset(seen_count, 0, sizeof(seen_count));
    pthread_setname_np(pthread_self(), "game");

    pin_to_cpu(GAME_CPU, "game");
    maybe_go_realtime("game", 30);
    g_stat.game_cpu = sched_getcpu();

    say("[game]   running on CPU%d -- blackjack + Monte-Carlo odds\n",
        g_stat.game_cpu);
    say("\n--- Round %d: show the dealer a card ---\n", round);

    while (!g_stop) {
        struct card c;
        char   name[48];
        double p_bust, p_win;
        int    total, soft;

        if (!ring_pop(&g_ring, &c, 250))
            continue;

        g_stat.game_cpu = sched_getcpu();
        card_name(&c, name, sizeof(name));

        if (c.joker) {
            say("\n[game cpu%d] JOKER -- round reset\n", g_stat.game_cpu);
            new_round(&np, seen_count, &round);
            continue;
        }

        if (np >= 15) {
            say("[game]   hand full, ignoring %s\n", name);
            continue;
        }

        player[np++] = c.rank;
        if (c.rank >= 0 && c.rank < 13)
            seen_count[c.rank]++;
        g_stat.cards++;

        total = hand_total(player, np, &soft);

        say("\n[game cpu%d] card %d: %-16s -> hand total %d%s\n",
            g_stat.game_cpu, np, name, total, soft ? " (soft)" : "");

        if (total > 21) {
            say("[game]   BUST at %d -- dealer wins round %d\n", total, round);
            new_round(&np, seen_count, &round);
            continue;
        }

        if (total == 21) {
            say("[game]   TWENTY-ONE -- player wins round %d\n", round);
            new_round(&np, seen_count, &round);
            continue;
        }

        /* The expensive part, and the reason CPU1 exists in this design. */
        {
            double t0 = now_s();

            estimate_odds(player, np, seen_count, &p_bust, &p_win);
            g_stat.mc_runs++;
            g_stat.game_cpu_s = thread_cpu_s();

            say("[game]   %d trials in %.0f ms on CPU%d: "
                "P(bust if hit) %.1f%%  P(win if stand) %.1f%%  -> %s\n",
                MC_TRIALS, (now_s() - t0) * 1000.0, sched_getcpu(),
                p_bust * 100.0, p_win * 100.0,
                (total >= PLAYER_AUTO_STAND || p_bust > 0.5) ? "STAND" : "HIT");
        }
    }

    say("[game]   stopping after %u cards, %u odds evaluations\n",
        g_stat.cards, g_stat.mc_runs);
    return NULL;
}

/* ---- per-core utilisation, straight from the kernel -------------------- */

struct cpu_sample { unsigned long long busy, total; };

static int read_cpu_sample(int cpu, struct cpu_sample *s)
{
    char  key[16], line[512];
    FILE *f = fopen("/proc/stat", "r");
    int   found = 0;
    size_t klen;

    if (!f)
        return -1;
    snprintf(key, sizeof(key), "cpu%d ", cpu);
    klen = strlen(key);

    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, key, klen) == 0) {
            unsigned long long v[10] = { 0 }, total = 0, idle;
            int n, i;

            n = sscanf(line + klen,
                       "%llu %llu %llu %llu %llu %llu %llu %llu %llu %llu",
                       &v[0], &v[1], &v[2], &v[3], &v[4],
                       &v[5], &v[6], &v[7], &v[8], &v[9]);
            if (n < 5) {
                break;
            }
            idle = v[3] + v[4];                 /* idle + iowait */
            for (i = 0; i < n; i++)
                total += v[i];
            s->total = total;
            s->busy  = total - idle;
            found = 1;
            break;
        }
    }
    fclose(f);
    return found ? 0 : -1;
}

static void *stats_thread(void *arg)
{
    struct cpu_sample prev[2], cur[2];
    int i, ok = 1;

    (void)arg;
    pthread_setname_np(pthread_self(), "stats");

    for (i = 0; i < 2; i++)
        if (read_cpu_sample(i, &prev[i]) != 0)
            ok = 0;

    while (!g_stop) {
        sleep(5);
        if (g_stop)
            break;

        if (ok) {
            char   util[96];
            size_t used = 0;

            util[0] = '\0';
            for (i = 0; i < 2; i++) {
                double pct = 0.0;

                if (read_cpu_sample(i, &cur[i]) == 0 &&
                    cur[i].total > prev[i].total) {
                    pct = 100.0 * (double)(cur[i].busy - prev[i].busy) /
                          (double)(cur[i].total - prev[i].total);
                }
                used += (size_t)snprintf(util + used, sizeof(util) - used,
                                         "%scpu%d %5.1f%%", i ? "  " : "", i, pct);
                prev[i] = cur[i];
            }
            say("\n[stats]  %s | vision %u frames (%.1fs cpu) | "
                "game %u cards, %u odds runs (%.1fs cpu)\n\n",
                util, g_stat.frames, g_stat.vision_cpu_s,
                g_stat.cards, g_stat.mc_runs, g_stat.game_cpu_s);
        } else {
            say("\n[stats]  vision %u frames (%.1fs cpu) | "
                "game %u cards, %u odds runs (%.1fs cpu)\n\n",
                g_stat.frames, g_stat.vision_cpu_s,
                g_stat.cards, g_stat.mc_runs, g_stat.game_cpu_s);
        }
    }
    return NULL;
}

/* ---- main -------------------------------------------------------------- */

static void usage(const char *me)
{
    fprintf(stderr,
        "usage: %s [--stats] [--rt] [--quiet]\n"
        "  --stats  print per-core utilisation from /proc/stat every 5 s\n"
        "  --rt     run both worker threads SCHED_FIFO (needs privilege)\n"
        "  --quiet  only print committed cards, not every frame\n"
        "  --fake   deal a fixed sequence instead of reading the FPGA, so the\n"
        "           two-core split and the game can be exercised with no board\n"
        "Needs /dev/mem (so, root) unless --fake is given.\n", me);
}

int main(int argc, char **argv)
{
    pthread_t t_vision, t_game, t_stats;
    long      ncpu;
    int       fd, i;
    void     *map;

    for (i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--stats")) g_opt_stats = 1;
        else if (!strcmp(argv[i], "--rt"))    g_opt_rt    = 1;
        else if (!strcmp(argv[i], "--quiet")) g_opt_quiet = 1;
        else if (!strcmp(argv[i], "--fake"))  g_opt_fake  = 1;
        else { usage(argv[0]); return 2; }
    }

    ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    printf("DE10-Nano card CNN -- Linux, %ld CPU%s online\n",
           ncpu, ncpu == 1 ? "" : "s");
    if (ncpu < 2)
        fprintf(stderr,
            "!! only %ld CPU online -- the second Cortex-A9 is not up, so the\n"
            "   two-core split cannot be demonstrated. Check that the kernel\n"
            "   booted SMP: dmesg | grep -i smp\n", ncpu);

    fd  = -1;
    map = MAP_FAILED;

    if (g_opt_fake) {
        printf("*** --fake: dealing a fixed sequence, FPGA not touched ***\n");
    } else {
        fd = open("/dev/mem", O_RDWR | O_SYNC);
        if (fd < 0) {
            perror("open /dev/mem");
            fprintf(stderr, "   (run as root, or pass --fake)\n");
            return 1;
        }

        map = mmap(NULL, LWBRIDGE_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED,
                   fd, (off_t)LWBRIDGE_BASE);
        if (map == MAP_FAILED) {
            perror("mmap lightweight bridge");
            fprintf(stderr,
                "   The FPGA must be configured and the HPS-to-FPGA bridges\n"
                "   enabled. See README.md.\n");
            close(fd);
            return 1;
        }
        g_lw = (volatile uint8_t *)map;
        printf("mapped LW bridge 0x%08lX (%lu bytes)\n",
               LWBRIDGE_BASE, LWBRIDGE_SPAN);
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    if (!g_opt_fake)
        check_camera();
    ring_init(&g_ring);

    printf("\ntask allocation:\n"
           "  CPU%d  vision  camera trigger, 96x96 snapshot read, Q6.10 scale,\n"
           "                CNN upload/start/poll, rank+suit decode\n"
           "  CPU%d  game    blackjack state machine, Monte-Carlo odds\n"
           "                (%d trials per committed card)\n\n",
           VISION_CPU, GAME_CPU, MC_TRIALS);

    if (pthread_create(&t_vision, NULL, vision_thread, NULL) != 0) {
        perror("pthread_create vision");
        return 1;
    }
    if (pthread_create(&t_game, NULL, game_thread, NULL) != 0) {
        perror("pthread_create game");
        g_stop = 1;
        pthread_join(t_vision, NULL);
        return 1;
    }
    if (g_opt_stats && pthread_create(&t_stats, NULL, stats_thread, NULL) != 0) {
        perror("pthread_create stats");
        g_opt_stats = 0;
    }

    pthread_join(t_vision, NULL);
    pthread_join(t_game, NULL);
    if (g_opt_stats) {
        pthread_cancel(t_stats);
        pthread_join(t_stats, NULL);
    }

    printf("\nfinal: vision %u frames on CPU%d (%.2fs CPU), "
           "game %u cards / %u odds runs on CPU%d (%.2fs CPU)\n",
           g_stat.frames, g_stat.vision_cpu, g_stat.vision_cpu_s,
           g_stat.cards, g_stat.mc_runs, g_stat.game_cpu, g_stat.game_cpu_s);

    if (map != MAP_FAILED)
        munmap(map, LWBRIDGE_SPAN);
    if (fd >= 0)
        close(fd);
    return 0;
}
