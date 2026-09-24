// ---------------------------------------------------------------------------
// maxpool_layer.v  --  parameterised 2x2 / stride-2 max pool
//
// Same role as LAB 5's maxpool.v, but parameterised over channels and
// dimension so one module serves both pool stages (8ch 48->24, 16ch 24->12),
// and restructured to 4 states per window instead of the lab's 16.
//
// The lab spends three states per input value (A/W/L: set address, wait,
// latch) because it re-reads a single port serially. That is 16 states per
// 2x2 window. Here the address is issued one cycle ahead and the value
// consumed on the next, so a window costs 4 cycles rather than 16.
//
//   pool1: 8*24*24*4  =  18,432 cycles  (0.37 ms @ 50 MHz)
//   pool2: 16*12*12*4 =   9,216 cycles  (0.18 ms @ 50 MHz)
// Negligible either way next to the conv layers -- this is tidiness, not a
// bottleneck fix.
// ---------------------------------------------------------------------------

module maxpool_layer #(
    parameter CH      = 8,
    parameter DIM_IN  = 48,        // must be even; output is DIM_IN/2
    parameter IN_AW   = 15,
    parameter OUT_AW  = 13
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,
    output reg                  done,

    output reg  [IN_AW-1:0]     in_rd_addr,
    input  wire signed [15:0]   in_rd_data,

    output reg  [OUT_AW-1:0]    out_wr_addr,
    output reg  signed [15:0]   out_wr_data,
    output reg                  out_wr_en
);
    localparam DIM_OUT   = DIM_IN / 2;
    localparam PLANE_IN  = DIM_IN * DIM_IN;
    localparam PLANE_OUT = DIM_OUT * DIM_OUT;

    reg [7:0] ch, oy, ox;
    reg [1:0] tap;                     // which of the 4 window pixels
    reg signed [15:0] best;
    // Two-stage delay -- see conv_layer.v.
    reg tap_v_d1, tap_v_d2;
    reg [1:0] drain_cnt;

    // window source coordinates for the current tap
    wire [7:0] sy = oy*2 + tap[1];
    wire [7:0] sx = ox*2 + tap[0];
    wire [IN_AW-1:0] src_addr = ch*PLANE_IN + sy*DIM_IN + sx;

    localparam S_IDLE=0, S_INIT=1, S_RUN=2, S_DRAIN=3, S_WRITE=4;
    reg [2:0] state;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_IDLE; done <= 0; out_wr_en <= 0;
            ch <= 0; oy <= 0; ox <= 0; tap <= 0; tap_v_d1 <= 0; tap_v_d2 <= 0;
        end else begin
            out_wr_en <= 0;
            tap_v_d1  <= 0;
            tap_v_d2  <= tap_v_d1;

            case (state)
                S_IDLE: begin
                    done <= 0;
                    if (start) begin ch <= 0; oy <= 0; ox <= 0; state <= S_INIT; end
                end

                S_INIT: begin
                    // ReLU has already been applied upstream, so every value
                    // is >= 0 and seeding with 0 is safe.
                    best <= 16'sd0;
                    tap  <= 0;
                    state <= S_RUN;
                end

                S_RUN: begin
                    in_rd_addr <= src_addr;
                    tap_v_d1   <= 1'b1;
                    if (tap == 2'd3) begin drain_cnt <= 2'd0; state <= S_DRAIN; end
                    else tap <= tap + 1'b1;
                end

                S_DRAIN: begin
                    if (drain_cnt == 2'd1) state <= S_WRITE;
                    else drain_cnt <= drain_cnt + 1'b1;
                end

                S_WRITE: begin
                    out_wr_addr <= ch*PLANE_OUT + oy*DIM_OUT + ox;
                    out_wr_data <= best;
                    out_wr_en   <= 1'b1;

                    if (ox == DIM_OUT-1) begin
                        ox <= 0;
                        if (oy == DIM_OUT-1) begin
                            oy <= 0;
                            if (ch == CH-1) begin done <= 1'b1; state <= S_IDLE; end
                            else begin ch <= ch + 1'b1; state <= S_INIT; end
                        end else begin oy <= oy + 1'b1; state <= S_INIT; end
                    end else begin ox <= ox + 1'b1; state <= S_INIT; end
                end

                default: state <= S_IDLE;
            endcase

            if (tap_v_d2 && ($signed(in_rd_data) > best))
                best <= $signed(in_rd_data);
        end
    end
endmodule


// ---------------------------------------------------------------------------
// ram_dp.v  --  simple dual-port RAM, registered read.
// Written this way (single write port, single read port, registered output)
// specifically so Quartus infers M10K rather than logic.
// ---------------------------------------------------------------------------
module ram_dp #(
    parameter AW    = 12,
    parameter DW    = 16,
    parameter DEPTH = (1<<AW)      // real depth; a full 2^AW would waste M10K blocks
)(
    input  wire             clk,
    input  wire             wr_en,
    input  wire [AW-1:0]    wr_addr,
    input  wire [DW-1:0]    wr_data,
    input  wire [AW-1:0]    rd_addr,
    output reg  [DW-1:0]    rd_data
);
    reg [DW-1:0] mem [0:DEPTH-1];
    always @(posedge clk) begin
        if (wr_en) mem[wr_addr] <= wr_data;
        rd_data <= mem[rd_addr];
    end
endmodule
