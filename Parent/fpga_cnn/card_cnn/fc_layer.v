// ---------------------------------------------------------------------------
// fc_layer.v  --  parameterised fully-connected layer
//
// Serves all four dense layers in this model:
//   fc_shared : 2305 -> 64   with ReLU   (2304 pooled features + colour flag)
//   fc_rank   :   64 -> 13   no ReLU (logits)
//   fc_suit   :   64 ->  4   no ReLU
//   fc_joker  :   64 ->  2   no ReLU
//
// Differences from LAB 5's fc_layer.v:
//
//   1. PIPELINED MAC. The lab uses six states per multiply-accumulate
//      (SET_ADDR / WAIT_READ / READ_MEM / MULT / ADD ...). At its size that
//      costs 10*676*6 = 40K cycles, which is nothing. This model's fc_shared
//      is 64*2305 = 147,520 MACs; at six cycles each that would be 885K
//      cycles (17.7 ms) for one layer. Issuing one MAC per cycle brings it
//      to 147K cycles (2.9 ms).
//
//   2. OUTPUTS TO MEMORY, NOT PORTS. The lab declares fc_out0..fc_out9 as
//      ten individual 32-bit ports and switch-cases across them. That does
//      not scale to a 64-wide hidden layer, and it makes the module
//      non-parameterisable. Results are written to a small RAM instead.
//
//   3. THE APPENDED COLOUR FLAG. fc_shared takes 2305 inputs: 2304 pooled
//      activations plus one scalar colour bit. Input index 2304 is sourced
//      from the colour_val port rather than the activation RAM. Getting this
//      wrong shifts every weight by one position and destroys the layer.
// ---------------------------------------------------------------------------

module fc_layer #(
    parameter N_IN     = 2305,
    parameter N_OUT    = 64,
    parameter USE_RELU = 1,
    parameter APPEND_COLOUR = 1,     // treat input index N_IN-1 as colour_val
    parameter IN_AW    = 12,
    parameter OUT_AW   = 6,
    parameter WFILE    = "fcs_w.hex",
    parameter BFILE    = "fcs_b.hex",
    parameter FRAC_BITS = 10,         // Q6.10 -- see conv_layer.v
    // Fractional bits of the WEIGHTS. Activations, biases and the output stay
    // Q6.10; products carry 2^(10 + W_FRAC), so the bias is aligned up by
    // W_FRAC and the result shifted back down by W_FRAC. fc_shared uses 14
    // (Q1.14, from DDR3): its weights are ~0.1 at most, where Q6.10 left ~16%
    // of them at 0 or +/-1 LSB. export_weights_rtl.py FCS_W_FRAC must match.
    parameter W_FRAC   = FRAC_BITS,
    // Weight ROM read width. See the long note in conv_layer.v: the .hex file
    // format is unchanged, only the port narrows, and export_weights_rtl.py's
    // NARROW_ROM_BITS gate guarantees every value still fits.
    parameter W_BITS   = 16,
    // 1 = weights come from HPS DDR3 through the w_req/w_resp handshake
    // (card_cnn_core.v's sdram_master) instead of an on-chip ROM.
    //
    // Used for fc_shared only. Its 147,520 weights were 289 M10K blocks at x16
    // (145 even narrowed to x10) -- the largest memory in the design -- and
    // they are the one big array that suits a one-transaction-at-a-time DDR3
    // master: read ONCE per inference, strictly in order. The conv
    // activations stay on-chip because each is re-read 25*C times per output.
    // Cost: one DDR3 round trip per weight, ~147,520 x (10-25 cycles), a few
    // tens of ms against ~1.7 s of convolution.
    parameter W_FROM_DDR = 0

)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,
    output reg                  done,

    input  wire signed [15:0]   colour_val,   // Q6.10: 1024 = red, 0 = black

    output reg  [IN_AW-1:0]     in_rd_addr,
    input  wire signed [15:0]   in_rd_data,

    output reg  [OUT_AW-1:0]    out_wr_addr,
    output reg  signed [15:0]   out_wr_data,
    output reg                  out_wr_en,

    // DDR3 weight fetch, W_FROM_DDR only (tie w_resp_* to 0 otherwise).
    // w_req_rd pulses for one cycle with the weight's index; the index is a
    // WORD offset from the master's BASE_ADDR (one weight per 32-bit word).
    output reg                  w_req_rd,
    output reg  [29:0]          w_req_addr,
    input  wire                 w_resp_valid,
    input  wire signed [15:0]   w_resp_data,
    // Set when a weight read goes unanswered for DDR_TIMEOUT cycles -- the
    // FPGA-to-SDRAM port is not responding (see atlas_main.c). The rest of the
    // layer then runs with zero weights so `done` still fires and the host can
    // say so, rather than the core hanging forever. Cleared by start.
    output reg                  ddr_err
);
    localparam N_W = N_OUT * N_IN;
    // A DDR-fed layer keeps a 1-word dummy so no 147K-word ROM is inferred.
    localparam N_WMEM = W_FROM_DDR ? 1 : N_W;
    localparam [15:0] DDR_TIMEOUT = 16'hFFFF;   // ~1.3 ms at 50 MHz

    reg signed [W_BITS-1:0] w_mem [0:N_WMEM-1];
    reg signed [15:0]       b_mem [0:N_OUT-1];

    generate if (!W_FROM_DDR) begin : g_rom
        initial $readmemh(WFILE, w_mem);
    end endgenerate
    initial $readmemh(BFILE, b_mem);

    reg [$clog2(N_OUT+1)-1:0] o;
    reg [$clog2(N_IN+1)-1:0]  k;
    reg signed [47:0] acc;

    // Two-stage delay -- see the note in conv_layer.v: the address register
    // and the RAM output register are both a cycle each.
    reg               v_d1, v_d2;
    reg               is_colour_d1, is_colour_d2;
    reg signed [15:0] w_d1, w_d2;
    reg [1:0]         drain_cnt;

    wire is_colour = (APPEND_COLOUR != 0) && (k == N_IN-1);
    wire [$clog2(N_W)-1:0] w_addr = o*N_IN + k;
    // the ROM path's index, pinned to 0 in a DDR-fed layer whose w_mem is 1 deep
    wire [$clog2(N_W)-1:0] w_addr_rom = W_FROM_DDR ? {$clog2(N_W){1'b0}} : w_addr;
    reg  [15:0]            wait_cnt;

    localparam S_IDLE=0, S_INIT=1, S_RUN=2, S_DRAIN=3, S_WRITE=4, S_WWAIT=5;
    reg [2:0] state;

    // Round-to-nearest before the shift -- see the note in conv_layer.v.
    localparam signed [47:0] HALF_LSB = 48'sd1 <<< (W_FRAC - 1);
    wire signed [47:0] relu_val = (USE_RELU != 0 && acc[47]) ? 48'sd0 : acc;
    wire signed [47:0] scaled   = (relu_val + HALF_LSB) >>> W_FRAC;
    wire signed [15:0] sat_val  = (scaled >  48'sd32767) ?  16'sd32767 :
                                  (scaled < -48'sd32768) ? -16'sd32768 : scaled[15:0];

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_IDLE; done <= 0; out_wr_en <= 0;
            o <= 0; k <= 0; v_d1 <= 0; v_d2 <= 0;
            w_req_rd <= 0; ddr_err <= 0;
        end else begin
            out_wr_en <= 0;
            v_d1      <= 0;
            w_req_rd  <= 0;

            v_d2         <= v_d1;
            is_colour_d2 <= is_colour_d1;
            w_d2         <= w_d1;

            case (state)
                S_IDLE: begin
                    done <= 0;
                    if (start) begin o <= 0; ddr_err <= 1'b0; state <= S_INIT; end
                end

                S_INIT: begin
                    acc <= $signed(b_mem[o]) <<< W_FRAC;  // align bias to 2^(FRAC_BITS + W_FRAC)
                    k <= 0;
                    state <= S_RUN;
                end

                S_RUN: begin
                    in_rd_addr   <= k[IN_AW-1:0];
                    if (W_FROM_DDR) begin
                        // One weight in flight at a time. The activation read
                        // above lands in 2 cycles; the DDR3 round trip takes at
                        // least 4, so both are ready when S_WWAIT sees the
                        // response. Once ddr_err is set, stop asking.
                        if (!ddr_err) begin
                            w_req_rd   <= 1'b1;
                            w_req_addr <= w_addr;
                        end
                        wait_cnt <= 16'd0;
                        state    <= S_WWAIT;
                    end else begin
                        w_d1         <= w_mem[w_addr_rom];
                        is_colour_d1 <= is_colour;
                        v_d1         <= 1'b1;
                        if (k == N_IN-1) begin drain_cnt <= 2'd0; state <= S_DRAIN; end
                        else k <= k + 1'b1;
                    end
                end

                // DDR-fed layers only: wait for this tap's weight, MAC, advance.
                // The MAC happens here rather than through the v_d1/v_d2 pipe,
                // so there is no drain state on the way out.
                S_WWAIT: begin
                    if (w_resp_valid || ddr_err) begin
                        if (!ddr_err)
                            acc <= acc + ($signed(is_colour ? colour_val : in_rd_data)
                                          * $signed(w_resp_data));
                        if (k == N_IN-1) state <= S_WRITE;
                        else begin k <= k + 1'b1; state <= S_RUN; end
                    end else if (wait_cnt == DDR_TIMEOUT)
                        ddr_err <= 1'b1;        // this and every later weight read as 0
                    else
                        wait_cnt <= wait_cnt + 1'b1;
                end

                S_DRAIN: begin
                    if (drain_cnt == 2'd1) state <= S_WRITE;
                    else drain_cnt <= drain_cnt + 1'b1;
                end

                S_WRITE: begin
                    // Plain assignment, NOT o[OUT_AW-1:0]. The counter o is
                    // sized by $clog2(N_OUT+1) -- three bits for the 4-output
                    // suit head -- so a 5-bit part-select of it reads two bits
                    // that do not exist and yields x. The write then lands at
                    // an undefined address and silently disappears. Direct
                    // assignment zero-extends correctly.
                    out_wr_addr <= o;
                    out_wr_data <= sat_val;
                    out_wr_en   <= 1'b1;
                    if (o == N_OUT-1) begin done <= 1'b1; state <= S_IDLE; end
                    else begin o <= o + 1'b1; state <= S_INIT; end
                end

                default: state <= S_IDLE;
            endcase

            if (v_d2)
                acc <= acc + ($signed(is_colour_d2 ? colour_val : in_rd_data)
                              * $signed(w_d2));
        end
    end
endmodule
