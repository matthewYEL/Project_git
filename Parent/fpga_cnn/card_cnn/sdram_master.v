// Vendored from the professor's DDR3 reference (lab 5/new ddr3 diles/). The
// header below describes HIS project's callers (cnn_core.v buffer_A/B); here
// the only caller is card_cnn_core.v, fetching fc_shared's weights for u_fcs.
//
// Single non-pipelined Avalon-MM master driving the HPS's F2H SDRAM bridge
// (hps_0.f2h_sdram0_data, enabled in soc_system.qsys - see cnn_core.v header).
// One outstanding transaction at a time: simplest correct thing to build
// without hardware/simulation access to verify a pipelined version against.
//
// 16-bit CNN values are stored one per 32-bit DDR3 word (byteenable picks the
// low halfword) - wastes 2 of every 4 bytes, but avoids read-modify-write for
// packing two values per word, which isn't worth the risk here.
//
// Callers (cnn_core.v's buffer_A/buffer_B logic) drive req_addr/req_wdata and
// pulse req_wr or req_rd; only one caller may be active at a time (enforced
// by cnn_core.v's existing priority muxing, since only one pipeline stage is
// ever running). req_ack pulses once the request is accepted (write: data
// latched by the bridge; read: the read command was issued - resp_valid
// pulses separately, later, when resp_rdata is actually ready).
module sdram_master #(
    // WORD address of word 0 in DDR3 (byte address >> 2). The port is
    // word-addressed -- see the note in S_IDLE -- so this goes out unshifted.
    // (The original comment here said "byte address", contradicting the code.)
    parameter [29:0] BASE_ADDR = 30'h0000_0000
)(
    input  wire        clk,
    input  wire        rst,

    input  wire [29:0] req_addr,       // WORD address (this module adds BASE_ADDR and <<2)
    input  wire signed [15:0] req_wdata,
    input  wire        req_wr,
    input  wire        req_rd,
    output reg          req_ack,
    output reg  signed [15:0] resp_rdata,
    output reg          resp_valid,

    output reg  [29:0] avm_address,
    output wire [7:0]  avm_burstcount,
    output reg          avm_read,
    output reg          avm_write,
    output reg  [31:0] avm_writedata,
    output reg  [3:0]  avm_byteenable,
    input  wire         avm_waitrequest,
    input  wire [31:0] avm_readdata,
    input  wire         avm_readdatavalid
);
    assign avm_burstcount = 8'd1;   // always single-beat - no bursting
    localparam S_IDLE=0, S_WR=1, S_RD_CMD=2, S_RD_WAIT=3;
    reg [1:0] state;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_IDLE;
            avm_read <= 0; avm_write <= 0; req_ack <= 0; resp_valid <= 0;
        end else begin
            req_ack <= 0;
            resp_valid <= 0;
            case (state)
                S_IDLE: begin
                    if (req_wr) begin
                        // NOT "<< 2": hps_0.f2h_sdram0_data's Avalon address
                        // port is WORD-addressed (addressUnits=WORDS in
                        // soc_system.sopcinfo, 32-bit data width - confirmed
                        // by reading the generated .sopcinfo directly, not
                        // assumed), so req_addr (already a word index by
                        // convention - see cnn_core.v's/rank_core.v's
                        // *_BASE_WORDS naming) needs no further conversion.
                        // A previous <<2 here silently multiplied every real
                        // physical address by 4 (e.g. HEAD_W_BASE=0x04200000
                        // became avm_address=0x10800000, which the word-
                        // addressed port then turned into byte address
                        // 0x42000000 - nowhere near the 0x10800000 the HPS
                        // actually writes to) - found 2026-09-04 chasing a
                        // real-hardware "every layer reads back as exactly
                        // 0" report: a sentinel value the HPS wrote directly
                        // to buffer_A/B survived completely untouched
                        // through a full inference pass, proving the FPGA
                        // fabric was never actually reaching those bytes.
                        avm_address    <= BASE_ADDR + req_addr;
                        avm_writedata  <= {16'd0, req_wdata};
                        avm_byteenable <= 4'b0011;
                        avm_write      <= 1'b1;
                        state <= S_WR;
                    end else if (req_rd) begin
                        avm_address <= BASE_ADDR + req_addr;
                        avm_read    <= 1'b1;
                        state <= S_RD_CMD;
                    end
                end
                S_WR: begin
                    if (!avm_waitrequest) begin
                        avm_write <= 1'b0;
                        req_ack   <= 1'b1;
                        state <= S_IDLE;
                    end
                end
                S_RD_CMD: begin
                    if (!avm_waitrequest) begin
                        avm_read <= 1'b0;
                        req_ack  <= 1'b1;
                        state <= S_RD_WAIT;
                    end
                end
                S_RD_WAIT: begin
                    if (avm_readdatavalid) begin
                        resp_rdata <= avm_readdata[15:0];
                        resp_valid <= 1'b1;
                        state <= S_IDLE;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
