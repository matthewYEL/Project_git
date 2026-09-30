// ---------------------------------------------------------------------------
// conv_layer.v  --  parameterised convolution + ReLU + rescale
//
// Follows LAB 5's conv_full.v in arithmetic and structure (Q4.12 fixed point,
// FSM sequencing, $readmemh weights) but differs in three ways that the lab's
// version cannot scale to:
//
//   1. PIPELINED SINGLE-PORT READS, NOT PARALLEL ARRAY READS.
//      The lab does `for (n=0;n<9;n=n+1) px[n] <= image_mem[px_addr[n]];`
//      -- nine reads in one cycle. Quartus cannot map that onto M10K (which
//      has two ports), so image_mem becomes LUT registers. At the lab's 784
//      words that costs 12,544 bits of logic; this model's conv1 input is
//      2304 words = 36,864 bits, against ~41K LEs total on the 5CSEBA6. Not
//      viable. Here one tap is issued per cycle to a real dual-port RAM and
//      the multiply happens one cycle later, so the memory stays in M10K and
//      only one multiplier is used.
//
//   2. MULTI-CHANNEL INPUT. conv2 needs 8 input channels; the lab's layer is
//      single-channel only. Channels are accumulated in the same tap loop.
//
//   3. PADDING. The lab uses 'valid' convolution (28x28 -> 26x26). This model
//      was trained with padding=2, so output dimensions match input. Taps
//      falling outside the image contribute zero, which is what PyTorch's
//      zero padding does -- get this wrong and every border pixel is skewed.
//
// THROUGHPUT: one tap per cycle in steady state, plus 4 cycles of per-output
// overhead (S_INIT 1 + S_DRAIN 2 + S_WRITE 1), so the real cost of one output
// pixel is IN_CH*K*K + 4 cycles. At 384x384 with the default 8 conv1 filters:
//   conv1:  8 * 384*384 * ( 25 + 4) = 34,209,792 cycles  ~684 ms @ 50 MHz
//   conv2: 16 *  96*96  * (200 + 4) = 30,081,024 cycles  ~602 ms
//   conv3: 16 *  48*48  * (400 + 4) = 14,893,056 cycles  ~298 ms
//   conv4: 16 *  24*24  * (400 + 4) =  3,723,264 cycles   ~74 ms
//                                     ----------          ------
//                                     82,907,136          ~1.66 s
// (4 conv1 filters halves conv1 and conv2: ~51 M cycles, ~1.0 s. fc_shared
// adds a few tens of ms fetching its weights from DDR3 -- see fc_layer.v.)
// That is why inference is snapshot-triggered rather than free-running. If it
// needs to be faster, the natural next step is to bank the activation RAM by
// kernel row and process 5 taps per cycle (5 DSPs instead of 1), cutting every
// figure by 5x; 102 of the device's 112 DSPs are idle. Not done here --
// correctness first, and ~1.7 s is inside the 5-second card exposure.
// ---------------------------------------------------------------------------

module conv_layer #(
    parameter IN_CH   = 1,          // input channels
    parameter OUT_CH  = 8,          // output channels (filters)
    parameter DIM     = 48,         // input is DIM x DIM; output is too (padded)
    parameter K       = 5,          // kernel size
    parameter PAD     = 2,          // zero padding
    parameter IN_AW   = 12,         // input activation address width
    parameter OUT_AW  = 15,         // output activation address width
    // Fold the following POOL x POOL max pool into this layer, so the output
    // written is DIM/POOL x DIM/POOL per channel instead of DIM x DIM. See the
    // note above the pool logic below. POOL must be 1, 2, 4 or 8; 1 disables
    // pooling (what FUSE_POOL=0 used to mean).
    //
    // conv1 uses POOL=4, not 2, because of a hard device limit: at 384x384 its
    // pooled output would be 8 x 192 x 192 = 294,912 words = 576 M10K blocks,
    // and the 5CSEBA6U23I7DK has 553 in total. POOL=4 makes it 8 x 96 x 96 =
    // 144 blocks, which is the most this chip can carry.
    parameter POOL    = 2,
    parameter WFILE   = "conv1_w.hex",
    parameter BFILE   = "conv1_b.hex",
    // Fixed-point format: FRAC_BITS fractional bits in a 16-bit signed word.
    // Q6.10 (10) was chosen by measurement, not convention -- see README.
    // Q4.12 saturated fc_shared badly enough to cost 6.6% accuracy.
    parameter FRAC_BITS = 10,
    // Width of the weight ROM's read port. The .hex FILE FORMAT DOES NOT
    // CHANGE -- it is always four hex digits of a 16-bit two's-complement
    // Q6.10 value -- and $readmemh into a narrower reg keeps the low W_BITS,
    // which is the correct value provided it fits. export_weights_rtl.py's
    // NARROW_ROM_BITS gate is what guarantees it fits; without that gate a
    // too-large weight is silently truncated rather than failing the build.
    //
    // Why narrow at all: an M10K holds 512 words at x16 but 1024 at x10, so
    // dropping 16 -> 10 halves a ROM's block count. On fcs_w (147,520 words)
    // that is 289 blocks down to 145 -- which is what pays for the 384x384
    // image RAM.
    parameter W_BITS  = 16,
    // conv1 of the M2 whole-grid scan: read a square window of the stored
    // FRAME_W x FRAME_H grey frame (one card), scaled onto this layer's
    // DIM x DIM input, instead of a DIM x DIM image. The frame is addressed
    // {row, col}, so FRAME_W must be a power of two. See "window mapper" below.
    // Only meaningful with IN_CH == 1; other layers leave it 0 and tie the
    // win_* / clip_* ports to 0.
    parameter WIN_MAP = 0,
    parameter FRAME_W = 512,
    parameter FRAME_H = 384

)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,
    output reg                  done,

    // read port -> input activation RAM (registered read, 1 cycle latency)
    output reg  [IN_AW-1:0]     in_rd_addr,
    input  wire signed [15:0]   in_rd_data,

    // write port -> output activation RAM
    output reg  [OUT_AW-1:0]    out_wr_addr,
    output reg  signed [15:0]   out_wr_data,
    output reg                  out_wr_en,

    // window registers (WIN_MAP only), held for the whole layer by the caller
    input  wire signed [10:0]   win_x0,      // frame column of virtual (0, 0); may be < 0
    input  wire signed [10:0]   win_y0,      // frame row of virtual (0, 0)
    input  wire        [8:0]    win_step,    // Q8.8 frame px per virtual px (256 = 1:1)
    input  wire        [2:0]    win_flags,   // [0] transpose [1] flip x [2] flip y
    input  wire        [9:0]    clip_x0,     // inclusive frame rectangle; taps outside read 0
    input  wire        [9:0]    clip_y0,
    input  wire        [9:0]    clip_x1,
    input  wire        [9:0]    clip_y1
);

    localparam N_W  = OUT_CH * IN_CH * K * K;
    localparam PLANE = DIM * DIM;
    // Counter width. At DIM 384 an 8-bit ox/oy wraps at 255 and the layer
    // silently convolves the top-left 256x256 corner forever, so this must
    // track DIM rather than being the fixed [7:0] the 96x96 build could use.
    localparam CW = $clog2(DIM);
    // log2(POOL), for slicing the pool phase out of the counters. Floored at 1
    // even for POOL==1: a part-select ox[PW-1:0] with PW==0 is ox[-1:0], which
    // is an elaboration error, not a harmless unused expression. POOL==1 short
    // -circuits those tests with a constant anyway.
    localparam PW = (POOL >= 8) ? 3 : (POOL >= 4) ? 2 : 1;

    reg signed [W_BITS-1:0] w_mem [0:N_W-1];
    reg signed [15:0]       b_mem [0:OUT_CH-1];

    initial begin
        $readmemh(WFILE, w_mem);
        $readmemh(BFILE, b_mem);
    end

    // ---- loop counters ----------------------------------------------------
    reg [7:0]     f;           // output channel  (<= 16, 8 bits is plenty)
    reg [CW-1:0]  oy, ox;      // output pixel
    reg [7:0]     c;           // input channel   (<= 16)
    reg [3:0]     ky, kx;      // kernel position

    // 40 bits, not 32: conv2 sums 200 products of up to 32767 x 566, which can
    // exceed 2^31 on saturated activations. Costs nothing at 50 MHz.
    reg signed [39:0] acc;

    // ---- pipeline registers -----------------------------------------------
    // TWO stages, not one. There are two registers between issuing an address
    // and the data being usable: in_rd_addr is itself a register (so the RAM
    // sees the address one cycle after we set it), and the RAM's output is
    // registered (so the data appears one cycle after that). A single-stage
    // delay multiplies each weight against the PREVIOUS tap's pixel, which
    // produces a plausible-looking but systematically wrong result.
    reg               issue_v_d1, issue_v_d2;
    reg               bounds_d1,  bounds_d2;
    reg signed [15:0] w_d1,       w_d2;
    reg [1:0]         drain_cnt;

    wire last_tap = (c == IN_CH-1) && (ky == K-1) && (kx == K-1);

    // current tap's source coordinates, signed so the padding test works.
    // Widened explicitly to 12 bits rather than assuming an 8-bit counter:
    // DIM 384 plus K and minus PAD spans -2 .. 388.
    wire signed [11:0] oy_s = $signed({{(12-CW){1'b0}}, oy});
    wire signed [11:0] ox_s = $signed({{(12-CW){1'b0}}, ox});
    wire signed [11:0] ky_s = $signed({8'b0, ky});
    wire signed [11:0] kx_s = $signed({8'b0, kx});
    wire signed [11:0] iy = oy_s + ky_s - PAD;
    wire signed [11:0] ix = ox_s + kx_s - PAD;
    wire in_bounds = (iy >= 0) && (iy < DIM) && (ix >= 0) && (ix < DIM);

    wire [IN_AW-1:0] tap_addr = c*PLANE + iy*DIM + ix;
    wire [$clog2(N_W)-1:0] w_addr = ((f*IN_CH + c)*K + ky)*K + kx;

    // ---- window mapper (WIN_MAP) ------------------------------------------
    // Virtual tap (iy, ix), valid when in_bounds: u = flip x ? DIM-1-ix : ix,
    // v likewise from iy with flip y; su = (u*STEP)>>8, sv = (v*STEP)>>8; the
    // frame pixel is (col, row) = transpose ? (X0+sv, Y0+su) : (X0+su, Y0+sv).
    // Taps outside the clip rectangle or the frame read zero, exactly like the
    // padding taps. Transpose plus one flip turns a sideways card upright.
    // sim_card_cnn.window_map() is the bit-exact reference.
    //
    // Combinational into in_rd_addr: two 9x9 multiplies, two adds and the
    // compares. If the fitter shows this path short of 50 MHz, register su/sv
    // one stage earlier (and extend the issue/bounds/weight pipeline by one).
    localparam COLB = $clog2(FRAME_W);
    wire [8:0]  wu   = win_flags[1] ? (DIM - 1 - ix[8:0]) : ix[8:0];
    wire [8:0]  wv   = win_flags[2] ? (DIM - 1 - iy[8:0]) : iy[8:0];
    wire [17:0] pu   = wu * win_step;
    wire [17:0] pv   = wv * win_step;
    wire [9:0]  su   = pu[17:8];
    wire [9:0]  sv   = pv[17:8];
    wire signed [11:0] fcol = win_x0 + $signed({2'b00, win_flags[0] ? sv : su});
    wire signed [11:0] frow = win_y0 + $signed({2'b00, win_flags[0] ? su : sv});
    wire in_window = (fcol >= $signed({2'b00, clip_x0})) && (fcol <= $signed({2'b00, clip_x1}))
                  && (frow >= $signed({2'b00, clip_y0})) && (frow <= $signed({2'b00, clip_y1}))
                  && (fcol >= 0) && (fcol < FRAME_W) && (frow >= 0) && (frow < FRAME_H);

    wire             tap_ok  = WIN_MAP ? (in_bounds && in_window) : in_bounds;
    wire [IN_AW-1:0] rd_next = WIN_MAP ? {frow[IN_AW-COLB-1:0], fcol[COLB-1:0]} : tap_addr;

    localparam S_IDLE = 3'd0,
               S_INIT = 3'd1,
               S_RUN  = 3'd2,
               S_DRAIN= 3'd3,
               S_WRITE= 3'd4;
    reg [2:0] state;

    // ReLU, then rescale from Q8.24 back to Q4.12 with saturation.
    //
    // ROUND-TO-NEAREST, NOT TRUNCATE. A bare >>> 12 floors, losing up to one
    // LSB in the same direction on every value. That bias accumulates through
    // conv1 -> pool -> conv2 -> pool -> fc, and measurably cost rank accuracy
    // in fixed-point emulation (93.4% vs the float model's 100%). Adding half
    // an LSB before the shift removes the bias for the price of one adder.
    localparam signed [39:0] HALF_LSB = 40'sd1 <<< (FRAC_BITS - 1);
    wire signed [39:0] relu_val = (acc[39]) ? 40'sd0 : acc;
    wire signed [39:0] scaled   = (relu_val + HALF_LSB) >>> FRAC_BITS;
    wire signed [15:0] sat_val  = (scaled > 40'sd32767) ? 16'sd32767 : scaled[15:0];
    // explicit 16x16 product so the DSP block gets a 32-bit multiply, not a 40-bit one
    wire signed [31:0] prod     = $signed(in_rd_data) * $signed(w_d2);

    // ---- fused POOL x POOL max pool ----------------------------------------
    // With POOL > 1 the layer writes pooled output directly rather than
    // materialising the full DIM x DIM map for a separate maxpool_layer to
    // read back. That buffer is the largest thing in the design: at 384x384
    // conv1's unpooled output is 8*384*384 = 1,179,648 words = 2,304 M10K
    // blocks, against 553 on the whole device. Pooled by 4 it is 73,728 words
    // = 144 blocks.
    //
    // S_WRITE emits pixels in raster order within a channel (ox, then oy, then
    // f), so one DIM/POOL-word row of running maxima is all the state needed:
    //   - `pair` accumulates the running max across POOL columns;
    //   - on the first row of a group that horizontal max is parked in rowmax[];
    //   - on later rows it is maxed into rowmax[];
    //   - on the last row of the group rowmax[] and it are combined and written.
    // ReLU has already been applied to every term, and max(relu(a), relu(b)) ==
    // relu(max(a, b)), so this is bit-identical to a separate maxpool_layer.
    //
    // DIM must be divisible by POOL. Otherwise both pooled dimensions are
    // floor(DIM/POOL) and the trailing rows/columns are silently dropped.
    localparam DIM_OUT   = DIM / POOL;
    localparam PLANE_OUT = DIM_OUT * DIM_OUT;

    // pool phase within the current group, and the group index it feeds
    wire              col_first = (POOL == 1) || (ox[PW-1:0] == 0);
    wire              col_last  = (POOL == 1) || (ox[PW-1:0] == POOL-1);
    wire              row_first = (POOL == 1) || (oy[PW-1:0] == 0);
    wire              row_last  = (POOL == 1) || (oy[PW-1:0] == POOL-1);
    wire [CW-PW-1:0]  gx        = ox[CW-1:PW];
    wire [CW-PW-1:0]  gy        = oy[CW-1:PW];

    reg  signed [15:0] rowmax [0:DIM_OUT-1];
    reg  signed [15:0] pair;
    // horizontal running max: restart it on the first column of a group
    wire signed [15:0] hmax = (col_first || sat_val > pair) ? sat_val : pair;
    // vertical: combine with the row of running maxima
    wire signed [15:0] vmax = (hmax > rowmax[gx]) ? hmax : rowmax[gx];

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_IDLE; done <= 1'b0; out_wr_en <= 1'b0;
            f <= 0; oy <= 0; ox <= 0; c <= 0; ky <= 0; kx <= 0;
            issue_v_d1 <= 1'b0; issue_v_d2 <= 1'b0;
        end else begin
            out_wr_en  <= 1'b0;
            issue_v_d1 <= 1'b0;

            // advance the second pipeline stage every cycle
            issue_v_d2 <= issue_v_d1;
            bounds_d2  <= bounds_d1;
            w_d2       <= w_d1;

            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        f <= 0; oy <= 0; ox <= 0;
                        state <= S_INIT;
                    end
                end

                // Seed the accumulator with the bias. The bias is stored
                // scaled by 2^FRAC_BITS, and products carry 2^(2*FRAC_BITS),
                // so it is shifted up by FRAC_BITS to line up. (LAB 5 writes
                // this as `bs*16'sd4096` with its fixed Q4.12 format.)
                S_INIT: begin
                    acc <= $signed(b_mem[f]) <<< FRAC_BITS;
                    c <= 0; ky <= 0; kx <= 0;
                    state <= S_RUN;
                end

                // One tap issued per cycle; the multiply for a tap happens on
                // the following cycle, when its RAM data has arrived.
                S_RUN: begin
                    in_rd_addr <= tap_ok ? rd_next : {IN_AW{1'b0}};
                    w_d1       <= w_mem[w_addr];
                    bounds_d1  <= tap_ok;
                    issue_v_d1 <= 1'b1;

                    if (last_tap) begin
                        drain_cnt <= 2'd0;
                        state <= S_DRAIN;
                    end else if (kx == K-1) begin
                        kx <= 0;
                        if (ky == K-1) begin ky <= 0; c <= c + 1; end
                        else ky <= ky + 1;
                    end else begin
                        kx <= kx + 1;
                    end
                end

                // Two cycles for the last issued taps to work through the
                // address register and the RAM's output register.
                S_DRAIN: begin
                    if (drain_cnt == 2'd1) state <= S_WRITE;
                    else drain_cnt <= drain_cnt + 1'b1;
                end

                S_WRITE: begin
                    if (POOL > 1) begin
                        // Carry the horizontal running max forward until the
                        // last column of the group; there, fold it into the
                        // row of maxima, and on the group's last row emit.
                        pair <= hmax;
                        if (col_last) begin
                            if (row_last) begin
                                out_wr_addr <= f*PLANE_OUT + gy*DIM_OUT + gx;
                                out_wr_data <= vmax;
                                out_wr_en   <= 1'b1;
                            end else if (row_first)
                                rowmax[gx] <= hmax;
                            else
                                rowmax[gx] <= vmax;
                        end
                    end else begin
                        out_wr_addr <= f*PLANE + oy*DIM + ox;
                        out_wr_data <= sat_val;
                        out_wr_en   <= 1'b1;
                    end

                    if (ox == DIM-1) begin
                        ox <= 0;
                        if (oy == DIM-1) begin
                            oy <= 0;
                            if (f == OUT_CH-1) begin
                                done  <= 1'b1;
                                state <= S_IDLE;
                            end else begin
                                f <= f + 1;
                                state <= S_INIT;
                            end
                        end else begin
                            oy <= oy + 1;
                            state <= S_INIT;
                        end
                    end else begin
                        ox <= ox + 1;
                        state <= S_INIT;
                    end
                end

                default: state <= S_IDLE;
            endcase

            // accumulate the tap issued two cycles ago, whose data is valid now
            if (issue_v_d2 && bounds_d2)
                acc <= acc + prod;
        end
    end

endmodule
