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
    parameter FRAC_BITS = 10          // Q6.10 -- see conv_layer.v

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
    output reg                  out_wr_en
);
    localparam N_W = N_OUT * N_IN;

    reg signed [15:0] w_mem [0:N_W-1];
    reg signed [15:0] b_mem [0:N_OUT-1];

    initial begin
        $readmemh(WFILE, w_mem);
        $readmemh(BFILE, b_mem);
    end

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

    localparam S_IDLE=0, S_INIT=1, S_RUN=2, S_DRAIN=3, S_WRITE=4;
    reg [2:0] state;

    // Round-to-nearest before the shift -- see the note in conv_layer.v.
    localparam signed [47:0] HALF_LSB = 48'sd1 <<< (FRAC_BITS - 1);
    wire signed [47:0] relu_val = (USE_RELU != 0 && acc[47]) ? 48'sd0 : acc;
    wire signed [47:0] scaled   = (relu_val + HALF_LSB) >>> FRAC_BITS;
    wire signed [15:0] sat_val  = (scaled >  48'sd32767) ?  16'sd32767 :
                                  (scaled < -48'sd32768) ? -16'sd32768 : scaled[15:0];

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_IDLE; done <= 0; out_wr_en <= 0;
            o <= 0; k <= 0; v_d1 <= 0; v_d2 <= 0;
        end else begin
            out_wr_en <= 0;
            v_d1      <= 0;

            v_d2         <= v_d1;
            is_colour_d2 <= is_colour_d1;
            w_d2         <= w_d1;

            case (state)
                S_IDLE: begin
                    done <= 0;
                    if (start) begin o <= 0; state <= S_INIT; end
                end

                S_INIT: begin
                    acc <= $signed(b_mem[o]) <<< FRAC_BITS;  // align bias to 2^(2*FRAC_BITS)
                    k <= 0;
                    state <= S_RUN;
                end

                S_RUN: begin
                    in_rd_addr   <= k[IN_AW-1:0];
                    w_d1         <= w_mem[w_addr];
                    is_colour_d1 <= is_colour;
                    v_d1         <= 1'b1;
                    if (k == N_IN-1) begin drain_cnt <= 2'd0; state <= S_DRAIN; end
                    else k <= k + 1'b1;
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
