/* Card finder -- see locate.h. Every rule here is locate.py's, in integers. */
#include <string.h>

#include "locate.h"

#define STEP        2                           /* sample every 2nd frame pixel */
#define SW_MAX      ( ( FRAME_W + STEP - 1 ) / STEP )
#define SH_MAX      ( ( FRAME_H + STEP - 1 ) / STEP )
#define MIN_BOX     100                         /* sample px; smaller is a speck */
#define MAX_CAND    64
#define VIRT        384                         /* conv1's virtual input */
#define RED_PER     16                          /* red when red samples x 16 > samples */
#define INDEX_NU    12                          /* samples per index corner, short side */
#define INDEX_NV    20                          /* ... and long side */
#define UP_DROP     30                          /* face_up: paper = within 30 of the border's white */
#define UP_PCT      19                          /* ... face up when >= 19 % of the middle is paper */

/* Where the index sits: (u0, v0, u1, v1) percent of the card's short and long
 * side from a corner, along the card's own edges (locate.py INDEX_BOX_PCT) */
static const int IDX[ 4 ] = { 4, 4, 18, 28 };

struct pt  { int x, y; };
struct box { int x0, y0, x1, y1; struct pt c[ 4 ]; };  /* c: TL, TR, BL, BR corner, sample px */

/* Static, not on a stack: CPU1's is 32 KB. Only one core locates at a time. */
static uint8_t  s_bin[ SH_MAX ][ SW_MAX ];      /* sample brighter than the threshold */
static uint8_t  s_seen[ SH_MAX ][ SW_MAX ];
static uint32_t s_stack[ SH_MAX * SW_MAX ];      /* y << 16 | x */
static uint32_t s_hist[ 256 ];

#define MIN( a, b ) ( ( a ) < ( b ) ? ( a ) : ( b ) )
#define MAX( a, b ) ( ( a ) > ( b ) ? ( a ) : ( b ) )

static long box_area( const struct box * b )
{
    return ( long ) ( b->x1 - b->x0 + 1 ) * ( b->y1 - b->y0 + 1 );
}

/* Threshold maximising the between-class variance; ties keep the lowest. */
static unsigned otsu( const uint32_t * hist )
{
    uint32_t total = 0, w_b = 0;
    uint64_t s     = 0;
    double   sum_all, sum_b = 0.0, best = -1.0;
    unsigned t, thr = 0;

    for( t = 0; t < 256; t++ ) {
        total += hist[ t ];
        s     += ( uint64_t ) t * hist[ t ];
    }
    sum_all = ( double ) s;
    for( t = 0; t < 256; t++ ) {
        uint32_t w_f;
        double   m_b, m_f, d, between;

        w_b += hist[ t ];
        if( w_b == 0 ) continue;
        w_f = total - w_b;
        if( w_f == 0 ) break;
        sum_b  += ( double ) ( ( uint64_t ) t * hist[ t ] );
        m_b     = sum_b / w_b;
        m_f     = ( sum_all - sum_b ) / w_f;
        d       = m_b - m_f;
        between = ( double ) w_b * ( double ) w_f * ( d * d );
        if( between > best ) {
            best = between;
            thr  = t;
        }
    }
    return thr;
}

static int shaped( int w, int h )
{
    return ( 2 * w >= h && 10 * w <= 9 * h ) || ( 2 * h >= w && 10 * h <= 9 * w );
}

/* A card's corners are its blob's extreme points: TL minimises x + y, BR
 * maximises it, TR maximises x - y, BL minimises it. Ties go to the upper point
 * for the top corners and the lower one for the bottom ones, so the pick doesn't
 * depend on the fill order -- locate.py _ksum / _kdiff. Is p beyond q as k? */
static int beyond( int k, struct pt p, struct pt q )
{
    int ps, qs;

    switch( k ) {
    case 0:  ps = p.x + p.y; qs = q.x + q.y; return ps < qs || ( ps == qs && p.y < q.y );
    case 1:  ps = p.x - p.y; qs = q.x - q.y; return ps > qs || ( ps == qs && p.y < q.y );
    case 2:  ps = p.x - p.y; qs = q.x - q.y; return ps < qs || ( ps == qs && p.y > q.y );
    default: ps = p.x + p.y; qs = q.x + q.y; return ps > qs || ( ps == qs && p.y > q.y );
    }
}

static void box_corners( struct box * b )          /* a split half: its box is all we know */
{
    b->c[ 0 ].x = b->x0; b->c[ 0 ].y = b->y0;
    b->c[ 1 ].x = b->x1; b->c[ 1 ].y = b->y0;
    b->c[ 2 ].x = b->x0; b->c[ 2 ].y = b->y1;
    b->c[ 3 ].x = b->x1; b->c[ 3 ].y = b->y1;
}

/* Union boxes overlapping by more than a quarter of the smaller, until none do
 * -- restarting after every merge, exactly as locate.py does. */
static int merge_overlapping( struct box * b, int n )
{
    int merged = 1;

    while( merged ) {
        int i, j, k;

        merged = 0;
        for( i = 0; i < n && !merged; i++ ) {
            for( j = i + 1; j < n; j++ ) {
                int  iw = MIN( b[ i ].x1, b[ j ].x1 ) - MAX( b[ i ].x0, b[ j ].x0 ) + 1;
                int  ih = MIN( b[ i ].y1, b[ j ].y1 ) - MAX( b[ i ].y0, b[ j ].y0 ) + 1;
                long small;

                if( iw <= 0 || ih <= 0 ) continue;
                small = MIN( box_area( &b[ i ] ), box_area( &b[ j ] ) );
                if( 4L * iw * ih > small ) {
                    b[ i ].x0 = MIN( b[ i ].x0, b[ j ].x0 );
                    b[ i ].y0 = MIN( b[ i ].y0, b[ j ].y0 );
                    b[ i ].x1 = MAX( b[ i ].x1, b[ j ].x1 );
                    b[ i ].y1 = MAX( b[ i ].y1, b[ j ].y1 );
                    for( k = 0; k < 4; k++ )
                        if( beyond( k, b[ j ].c[ k ], b[ i ].c[ k ] ) ) b[ i ].c[ k ] = b[ j ].c[ k ];
                    memmove( &b[ j ], &b[ j + 1 ], ( size_t ) ( n - j - 1 ) * sizeof *b );
                    n--;
                    merged = 1;
                    break;
                }
            }
        }
    }
    return n;
}

static void sort_ints( long * v, int n )            /* insertion sort, ascending */
{
    int i, j;

    for( i = 1; i < n; i++ ) {
        long x = v[ i ];
        for( j = i; j > 0 && v[ j - 1 ] > x; j-- ) v[ j ] = v[ j - 1 ];
        v[ j ] = x;
    }
}

/* floor( sqrt( v ) ) for v >= 0, as Python's math.isqrt */
static int64_t isqrt64( int64_t v )
{
    int64_t r = 0, bit = ( int64_t ) 1 << 62;

    while( bit > v ) bit >>= 2;
    while( bit ) {
        if( v >= r + bit ) {
            v -= r + bit;
            r  = ( r >> 1 ) + bit;
        } else {
            r >>= 1;
        }
        bit >>= 2;
    }
    return r;
}

/* Group index of each value, split where sorted values jump by more than gap2
 * (values and gap both doubled, so the half-pixel centres stay integers).
 * Stable: equal values keep their input order, as Python's sorted() does. */
static int cluster( const int64_t * v, int n, int64_t gap2, uint8_t * idx )
{
    int order[ SCAN_MAX ], i, j, g = 0;

    for( i = 0; i < n; i++ ) order[ i ] = i;
    for( i = 1; i < n; i++ ) {
        int x = order[ i ];
        for( j = i; j > 0 && v[ order[ j - 1 ] ] > v[ x ]; j-- ) order[ j ] = order[ j - 1 ];
        order[ j ] = x;
    }
    for( i = 0; i < n; i++ ) {
        if( i && v[ order[ i ] ] - v[ order[ i - 1 ] ] > gap2 ) g++;
        idx[ order[ i ] ] = ( uint8_t ) g;
    }
    return g + 1;
}

/* Rows and columns from the gaps between the centres, first turned by the
 * grid's own direction (dx, dy) = the sum of the cards' top edges, TR - TL, so
 * a grid held at a slant still falls into clean rows and columns -- locate.py
 * assign_grid. Unnormalised, so the gaps are scaled by isqrt(|d|^2) to match. */
static void assign_grid( struct scan_result * out, unsigned orient, int64_t dx, int64_t dy )
{
    int64_t  ra[ SCAN_MAX ], ca[ SCAN_MAX ], norm;
    long     er[ SCAN_MAX ], ec[ SCAN_MAX ];
    uint8_t  ri[ SCAN_MAX ], ci[ SCAN_MAX ];
    int      i, n = ( int ) out->n, nr, nc, swap = ( orient & ORIENT_SWAP ) != 0;

    if( !n ) {
        out->nrows = out->ncols = 0;
        return;
    }
    if( !dx && !dy ) dx = 1;
    norm = isqrt64( dx * dx + dy * dy );
    for( i = 0; i < n; i++ ) {
        const struct scan_card * c = &out->card[ i ];
        int64_t sx = c->x0 + c->x1, sy = c->y0 + c->y1;      /* doubled centres */
        int64_t tx = sx * dx + sy * dy, ty = sy * dx - sx * dy;
        long    w  = c->x1 - c->x0 + 1, h = c->y1 - c->y0 + 1;

        ra[ i ] = swap ? tx : ty;
        ca[ i ] = swap ? ty : tx;
        er[ i ] = swap ? w : h;
        ec[ i ] = swap ? h : w;
    }
    sort_ints( er, n );
    sort_ints( ec, n );
    nr = cluster( ra, n, er[ n / 2 ] * norm, ri );           /* gap: half the median extent */
    nc = cluster( ca, n, ec[ n / 2 ] * norm, ci );
    for( i = 0; i < n; i++ ) {
        out->card[ i ].row = ( uint8_t ) ( ( orient & ORIENT_FLIP_R ) ? nr - ri[ i ] : ri[ i ] + 1 );
        out->card[ i ].col = ( uint8_t ) ( ( orient & ORIENT_FLIP_C ) ? nc - ci[ i ] : ci[ i ] + 1 );
    }
    out->nrows = ( uint32_t ) nr;
    out->ncols = ( uint32_t ) nc;
}

/* floor( n / d ) for d > 0, as Python's // */
static long fdiv( long n, long d )
{
    return n >= 0 ? n / d : -( ( -n + d - 1 ) / d );
}

/* The box's sample-px corners in frame px (a sample stands for STEP x STEP) */
static void frame_corners( const struct box * b, int fw, int fh, struct pt f[ 4 ] )
{
    f[ 0 ].x = b->c[ 0 ].x * STEP;                            f[ 0 ].y = b->c[ 0 ].y * STEP;
    f[ 1 ].x = MIN( fw - 1, b->c[ 1 ].x * STEP + STEP - 1 );  f[ 1 ].y = b->c[ 1 ].y * STEP;
    f[ 2 ].x = b->c[ 2 ].x * STEP;                            f[ 2 ].y = MIN( fh - 1, b->c[ 2 ].y * STEP + STEP - 1 );
    f[ 3 ].x = MIN( fw - 1, b->c[ 3 ].x * STEP + STEP - 1 );  f[ 3 ].y = MIN( fh - 1, b->c[ 3 ].y * STEP + STEP - 1 );
}

/* Colour from both index corners: INDEX_NU x INDEX_NV samples each, laid along
 * the card's own edges from the corner (u on the short side, v on the long), so
 * tilt and perspective can't move them off the index -- locate.py
 * index_samples. Top-left and bottom-right of a portrait card, top-right and
 * bottom-left of one lying sideways. Only the indices: court artwork is red
 * and gold whatever the suit. */
/* Sample (i, j) of an nu x nv patch box = { u0, v0, u1, v1 } percent, laid from
 * corner c along the card's own edges to a (u, the short side) and b (v, the
 * long side), nearest pixel -- locate.py _patch. 0 if it falls off the frame. */
static int patch_point( struct pt c, struct pt a, struct pt b, const int box[ 4 ], int nu, int nv,
                        int i, int j, int fw, int fh, struct pt * p )
{
    const long d  = 100L * ( nu - 1 ) * ( nv - 1 );
    long       fu = ( long ) ( box[ 0 ] * ( nu - 1 ) + ( box[ 2 ] - box[ 0 ] ) * i ) * ( nv - 1 );
    long       fv = ( long ) ( box[ 1 ] * ( nv - 1 ) + ( box[ 3 ] - box[ 1 ] ) * j ) * ( nu - 1 );
    long       nx = fu * ( a.x - c.x ) + fv * ( b.x - c.x );
    long       ny = fu * ( a.y - c.y ) + fv * ( b.y - c.y );

    p->x = c.x + ( int ) fdiv( 2 * nx + d, 2 * d );
    p->y = c.y + ( int ) fdiv( 2 * ny + d, 2 * d );
    return p->x >= 0 && p->x < fw && p->y >= 0 && p->y < fh;
}

static int index_red( const struct pt f[ 4 ], int landscape, int fw, int fh )
{
    /* corner, its neighbour along the short side, along the long side */
    static const uint8_t PORTRAIT[ 2 ][ 3 ] = { { 0, 1, 2 }, { 3, 2, 1 } };
    static const uint8_t SIDEWAYS[ 2 ][ 3 ] = { { 1, 3, 0 }, { 2, 0, 3 } };
    const uint8_t ( *patch )[ 3 ] = landscape ? SIDEWAYS : PORTRAIT;
    long n = 0, m = 0;
    int  k, i, j;

    for( k = 0; k < 2; k++ ) {
        struct pt c = f[ patch[ k ][ 0 ] ], a = f[ patch[ k ][ 1 ] ], b = f[ patch[ k ][ 2 ] ], p;

        for( i = 0; i < INDEX_NU; i++ )
            for( j = 0; j < INDEX_NV; j++ ) {
                if( !patch_point( c, a, b, IDX, INDEX_NU, INDEX_NV, i, j, fw, fh, &p ) ) continue;
                m++;
                n += PIX_RED( snap_read( FRAME_ADDR( p.y, p.x ) ) );
            }
    }
    return n * RED_PER > m;
}

/* Face up? A face is mostly paper-white in the middle; a back (a black pattern
 * inside a white border, 30 Sep) blurs to grey at D8M sharpness. White is the
 * card's own border strip, so dim light doesn't matter: face up when UP_PCT %
 * of the middle is within UP_DROP of the strip's median -- locate.py face_up. */
static int face_up( const struct pt f[ 4 ], int landscape, int fw, int fh )
{
    static const int BORDER[ 4 ] = { 1, 10, 4, 90 }, CENTRE[ 4 ] = { 15, 12, 85, 88 };
    struct pt c = f[ landscape ? 1 : 0 ], a = f[ landscape ? 3 : 1 ], b = f[ landscape ? 0 : 2 ], p;
    int       v[ 4 * 16 ], nb = 0, n = 0, paper = 0, white, i, j, k;

    for( i = 0; i < 4; i++ )
        for( j = 0; j < 16; j++ )
            if( patch_point( c, a, b, BORDER, 4, 16, i, j, fw, fh, &p ) )
                v[ nb++ ] = ( int ) PIX_GRAY( snap_read( FRAME_ADDR( p.y, p.x ) ) );
    if( !nb ) return 1;
    for( i = 1; i < nb; i++ ) {                              /* insertion sort, ascending */
        int x = v[ i ];
        for( k = i; k > 0 && v[ k - 1 ] > x; k-- ) v[ k ] = v[ k - 1 ];
        v[ k ] = x;
    }
    white = ( v[ ( nb - 1 ) / 2 ] + v[ nb / 2 ] ) / 2;
    for( i = 0; i < 12; i++ )
        for( j = 0; j < 16; j++ )
            if( patch_point( c, a, b, CENTRE, 12, 16, i, j, fw, fh, &p ) ) {
                n++;
                paper += ( int ) PIX_GRAY( snap_read( FRAME_ADDR( p.y, p.x ) ) ) >= white - UP_DROP;
            }
    return !n || paper * 100 >= UP_PCT * n;
}

unsigned locate_cards( struct scan_result * out, unsigned orient, int fw, int fh )
{
    static struct box cand[ MAX_CAND ], cards[ MAX_CAND ];
    long     areas[ MAX_CAND ], ws[ MAX_CAND ], hs[ MAX_CAND ];
    int      sw = ( fw + STEP - 1 ) / STEP, sh = ( fh + STEP - 1 ) / STEP;
    int      x, y, i, j, n_cand = 0, n_shaped = 0, n_cards = 0;
    int64_t  dx = 0, dy = 0;                    /* the grid's direction: sum of top edges */
    unsigned thr;

    memset( out, 0, sizeof *out );
    if( sw > SW_MAX || sh > SH_MAX ) return 0;

    /* 1-2: sample, histogram, Otsu */
    memset( s_hist, 0, sizeof s_hist );
    for( y = 0; y < sh; y++ )
        for( x = 0; x < sw; x++ ) {
            unsigned g = PIX_GRAY( snap_read( FRAME_ADDR( y * STEP, x * STEP ) ) );
            s_bin[ y ][ x ] = ( uint8_t ) g;                 /* the grey for now */
            s_hist[ g & 0xFFu ]++;
        }
    thr = otsu( s_hist );
    for( y = 0; y < sh; y++ )
        for( x = 0; x < sw; x++ ) s_bin[ y ][ x ] = s_bin[ y ][ x ] > thr;
    out->threshold = thr;

    /* 3: 4-connected components, keeping boxes off the border and not specks */
    memset( s_seen, 0, sizeof s_seen );
    for( y = 0; y < sh; y++ ) {
        for( x = 0; x < sw; x++ ) {
            int       top = 0, x0 = x, x1 = x, y0 = y, y1 = y, q;
            struct pt cor[ 4 ];                              /* TL, TR, BL, BR so far */

            if( !s_bin[ y ][ x ] || s_seen[ y ][ x ] ) continue;
            for( q = 0; q < 4; q++ ) {
                cor[ q ].x = x;
                cor[ q ].y = y;
            }
            s_seen[ y ][ x ] = 1;
            s_stack[ top++ ] = ( ( uint32_t ) y << 16 ) | ( uint32_t ) x;
            while( top ) {
                uint32_t  p  = s_stack[ --top ];
                int       cy = ( int ) ( p >> 16 ), cx = ( int ) ( p & 0xFFFFu );
                static const int DY[ 4 ] = { -1, 1, 0, 0 }, DX[ 4 ] = { 0, 0, -1, 1 };
                struct pt at;
                int       k;

                at.x = cx;
                at.y = cy;
                x0 = MIN( x0, cx ); x1 = MAX( x1, cx );
                y0 = MIN( y0, cy ); y1 = MAX( y1, cy );
                for( k = 0; k < 4; k++ )
                    if( beyond( k, at, cor[ k ] ) ) cor[ k ] = at;
                for( k = 0; k < 4; k++ ) {
                    int ny = cy + DY[ k ], nx = cx + DX[ k ];
                    if( ny >= 0 && ny < sh && nx >= 0 && nx < sw && s_bin[ ny ][ nx ] && !s_seen[ ny ][ nx ] ) {
                        s_seen[ ny ][ nx ] = 1;
                        s_stack[ top++ ] = ( ( uint32_t ) ny << 16 ) | ( uint32_t ) nx;
                    }
                }
            }
            if( x0 == 0 || y0 == 0 || x1 == sw - 1 || y1 == sh - 1 ) continue;
            if( ( long ) ( x1 - x0 + 1 ) * ( y1 - y0 + 1 ) < MIN_BOX ) continue;
            if( n_cand < MAX_CAND ) {
                cand[ n_cand ].x0 = x0; cand[ n_cand ].y0 = y0;
                cand[ n_cand ].x1 = x1; cand[ n_cand ].y1 = y1;
                memcpy( cand[ n_cand ].c, cor, sizeof cor );
                n_cand++;
            }
        }
    }

    /* 4: merge pieces of one card, then keep the card-shaped, card-sized ones */
    n_cand = merge_overlapping( cand, n_cand );
    /* the size reference counts only card-shaped boxes at least a quarter of
     * the largest: with 1-2 cards, slivers of other cards dragged the median
     * down and got the real card rejected as too big (locate.py, 29 Sep) */
    {
        long top = 0;

        for( i = 0; i < n_cand; i++ ) {
            int w = cand[ i ].x1 - cand[ i ].x0 + 1, h = cand[ i ].y1 - cand[ i ].y0 + 1;
            if( shaped( w, h ) ) top = MAX( top, ( long ) w * h );
        }
        for( i = 0; i < n_cand; i++ ) {
            int w = cand[ i ].x1 - cand[ i ].x0 + 1, h = cand[ i ].y1 - cand[ i ].y0 + 1;
            if( !shaped( w, h ) || 4L * w * h < top ) continue;
            areas[ n_shaped ] = ( long ) w * h;
            ws[ n_shaped ]    = w;
            hs[ n_shaped ]    = h;
            n_shaped++;
        }
    }
    if( n_shaped ) {
        long med, mw, mh;

        sort_ints( areas, n_shaped ); sort_ints( ws, n_shaped ); sort_ints( hs, n_shaped );
        med = areas[ n_shaped / 2 ]; mw = ws[ n_shaped / 2 ]; mh = hs[ n_shaped / 2 ];
        for( i = 0; i < n_cand && n_cards < MAX_CAND - 1; i++ ) {
            const struct box * c = &cand[ i ];
            int  w = c->x1 - c->x0 + 1, h = c->y1 - c->y0 + 1;
            long a = ( long ) w * h;

            if( 2 * a >= med && 10 * a <= 16 * med && shaped( w, h ) ) {
                cards[ n_cards++ ] = *c;
            } else if( 17 * med <= 10 * a && 10 * a <= 24 * med ) {
                struct box hv[ 2 ];                          /* two touching cards */
                int k;

                hv[ 0 ] = hv[ 1 ] = *c;
                if( ( long ) w * mh >= ( long ) h * mw ) {
                    int xm = c->x0 + w / 2;
                    hv[ 0 ].x1 = xm - 1; hv[ 1 ].x0 = xm;
                } else {
                    int ym = c->y0 + h / 2;
                    hv[ 0 ].y1 = ym - 1; hv[ 1 ].y0 = ym;
                }
                for( k = 0; k < 2; k++ ) {
                    box_corners( &hv[ k ] );
                    if( shaped( hv[ k ].x1 - hv[ k ].x0 + 1, hv[ k ].y1 - hv[ k ].y0 + 1 ) )
                        cards[ n_cards++ ] = hv[ k ];
                }
            }
        }
    }

    /* drop boxes inside another card's box (court-card artwork fragments),
     * then frame pixels and the per-card colour */
    for( i = 0; i < n_cards; i++ ) {
        const struct box * c = &cards[ i ];
        int inside = 0;

        for( j = 0; j < n_cards && !inside; j++ ) {
            const struct box * o = &cards[ j ];
            inside = j != i && o->x0 <= c->x0 && o->y0 <= c->y0 && c->x1 <= o->x1 && c->y1 <= o->y1
                     && ( long ) ( o->x1 - o->x0 ) * ( o->y1 - o->y0 ) > ( long ) ( c->x1 - c->x0 ) * ( c->y1 - c->y0 );
        }
        if( inside || out->n >= SCAN_MAX ) continue;
        {
            struct scan_card * s = &out->card[ out->n++ ];
            struct pt          f[ 4 ];

            s->x0 = ( uint16_t ) ( c->x0 * STEP );
            s->y0 = ( uint16_t ) ( c->y0 * STEP );
            s->x1 = ( uint16_t ) MIN( fw - 1, c->x1 * STEP + STEP - 1 );
            s->y1 = ( uint16_t ) MIN( fh - 1, c->y1 * STEP + STEP - 1 );
            frame_corners( c, fw, fh, f );
            s->red = ( uint8_t ) index_red( f, s->x1 - s->x0 > s->y1 - s->y0, fw, fh );
            s->up  = ( uint8_t ) face_up( f, s->x1 - s->x0 > s->y1 - s->y0, fw, fh );
            dx += f[ 1 ].x - f[ 0 ].x;
            dy += f[ 1 ].y - f[ 0 ].y;
        }
    }

    /* 5: rows and columns, then (row, col) order */
    assign_grid( out, orient, dx, dy );
    for( i = 1; i < ( int ) out->n; i++ ) {
        struct scan_card c = out->card[ i ];
        for( j = i; j > 0 && ( out->card[ j - 1 ].row > c.row ||
                               ( out->card[ j - 1 ].row == c.row && out->card[ j - 1 ].col > c.col ) ); j-- )
            out->card[ j ] = out->card[ j - 1 ];
        out->card[ j ] = c;
    }
    return out->n;
}

void window_for_card( const struct scan_card * c, unsigned orient, int fw, int fh,
                      struct win_regs * w )
{
    int cw = c->x1 - c->x0 + 1, ch = c->y1 - c->y0 + 1;
    int lng = MAX( cw, ch ), step, span, m;

    step = ( 23 * lng + 15 ) / 30;                           /* 1.15 x long x 256/384 */
    step = MAX( 1, MIN( 511, step ) );
    span = ( ( VIRT - 1 ) * step ) >> 8;                     /* frame px it covers */
    m    = MAX( 2, ( 4 * lng + 50 ) / 100 );
    /* centre - span/2, floor division: the sum can be negative */
    w->x0      = ( int16_t ) ( ( c->x0 + c->x1 - span + 1 ) >> 1 );
    w->y0      = ( int16_t ) ( ( c->y0 + c->y1 - span + 1 ) >> 1 );
    w->step    = ( uint16_t ) step;
    w->flags   = ( uint16_t ) ( cw > ch ? ( WIN_T | ( ( orient & ORIENT_FLIP_R ) ? WIN_FY : WIN_FX ) ) : 0 );
    w->clip[ 0 ] = ( uint16_t ) MAX( 0, c->x0 - m );
    w->clip[ 1 ] = ( uint16_t ) MAX( 0, c->y0 - m );
    w->clip[ 2 ] = ( uint16_t ) MIN( fw - 1, c->x1 + m );
    w->clip[ 3 ] = ( uint16_t ) MIN( fh - 1, c->y1 + m );
}
