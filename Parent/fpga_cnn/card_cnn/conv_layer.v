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
// THROUGHPUT: one tap per cycle in steady state.
//   conv1: 8*96*96*1*25  = 1,843,200 cycles  ~36.9 ms @ 50 MHz
//   conv2: 16*48*48*8*25 = 7,372,800 cycles ~147.5 ms @ 50 MHz
//   conv3: 16*24*24*16*25 = 3,686,400 cycles ~73.7 ms @ 50 MHz
// If that is too slow, the natural next step is to bank the activation RAM
// by kernel row and process 5 taps per cycle (5 DSPs instead of 1), cutting
// both figures by 5x. Not done here -- correctness first.
// ---------------------------------------------------------------------------

module conv_layer #(
    parameter IN_CH   = 1,          // input channels
    parameter OUT_CH  = 8,          // output channels (filters)
    parameter DIM     = 48,         // input is DIM x DIM; output is too (padded)
    parameter K       = 5,          // kernel size
    parameter PAD     = 2,          // zero padding
    parameter IN_AW   = 12,         // input activation address width
    parameter OUT_AW  = 15,         // output activation address width
    // Fold the following 2x2 max pool into this layer, so the output written
    // is DIM/2 x DIM/2 per channel instead of DIM x DIM. See the note above
    // the pool logic below -- at 96x96 the unpooled conv1 map does not fit.
    parameter FUSE_POOL = 1,
    parameter WFILE   = "conv1_w.hex",
    parameter BFILE   = "conv1_b.hex",
    // Fixed-point format: FRAC_BITS fractional bits in a 16-bit signed word.
    // Q6.10 (10) was chosen by measurement, not convention -- see README.
    // Q4.12 saturated fc_shared badly enough to cost 6.6% accuracy.
    parameter FRAC_BITS = 10

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
    output reg                  out_wr_en
);

    localparam N_W  = OUT_CH * IN_CH * K * K;
    localparam PLANE = DIM * DIM;

    reg signed [15:0] w_mem [0:N_W-1];
    reg signed [15:0] b_mem [0:OUT_CH-1];

    initial begin
        $readmemh(WFILE, w_mem);
        $readmemh(BFILE, b_mem);
    end

    // ---- loop counters ----------------------------------------------------
    reg [7:0]  f;              // output channel
    reg [7:0]  oy, ox;         // output pixel
    reg [7:0]  c;              // input channel
    reg [3:0]  ky, kx;         // kernel position

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

    // current tap's source coordinates, signed so the padding test works
    wire signed [10:0] iy = $signed({3'b0, oy}) + $signed({7'b0, ky}) - PAD;
    wire signed [10:0] ix = $signed({3'b0, ox}) + $signed({7'b0, kx}) - PAD;
    wire in_bounds = (iy >= 0) && (iy < DIM) && (ix >= 0) && (ix < DIM);

    wire [IN_AW-1:0] tap_addr = c*PLANE + iy*DIM + ix;
    wire [$clog2(N_W)-1:0] w_addr = ((f*IN_CH + c)*K + ky)*K + kx;

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

    // ---- fused 2x2 max pool ------------------------------------------------
    // With FUSE_POOL the layer writes pooled output directly rather than
    // materialising the full DIM x DIM map for a separate maxpool_layer to
    // read back. That buffer is the largest thing in the design at 96x96:
    // conv1's unpooled output is 8*96*96 = 73,728 words = 144 M10K blocks,
    // against 553 on the whole device. Pooled, it is 18,432 words = 36 blocks.
    //
    // S_WRITE emits pixels in raster order within a channel (ox, then oy, then
    // f), so one DIM/2-word row of running maxima is all the state needed: on
    // an even row each horizontal pair's max is parked in rowmax[]; on an odd
    // row it is maxed against rowmax[] and written out. ReLU has already been
    // applied to both, and max(relu(a), relu(b)) == relu(max(a, b)), so this is
    // bit-identical to what the separate maxpool_layer produced.
    //
    // DIM must be even. Both pooled dimensions are floor(DIM/2) otherwise, and
    // the trailing row/column would be silently dropped.
    localparam DIM_OUT   = DIM / 2;
    localparam PLANE_OUT = DIM_OUT * DIM_OUT;

    reg  signed [15:0] rowmax [0:DIM_OUT-1];
    reg  signed [15:0] pair;
    wire signed [15:0] pmax = (sat_val > pair)            ? sat_val : pair;
    wire signed [15:0] vmax = (pmax > rowmax[ox[7:1]])    ? pmax    : rowmax[ox[7:1]];

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
                    in_rd_addr <= in_bounds ? tap_addr : {IN_AW{1'b0}};
                    w_d1       <= w_mem[w_addr];
                    bounds_d1  <= in_bounds;
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
                    if (FUSE_POOL) begin
                        // even column: park the pixel. odd column: reduce the
                        // horizontal pair, then either stash it (even row) or
                        // combine it with the stashed row and emit (odd row).
                        if (!ox[0])
                            pair <= sat_val;
                        else if (!oy[0])
                            rowmax[ox[7:1]] <= pmax;
                        else begin
                            out_wr_addr <= f*PLANE_OUT + oy[7:1]*DIM_OUT + ox[7:1];
                            out_wr_data <= vmax;
                            out_wr_en   <= 1'b1;
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
