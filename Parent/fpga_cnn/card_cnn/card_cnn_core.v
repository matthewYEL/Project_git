// ---------------------------------------------------------------------------
// card_cnn_core.v  --  top-level card recognition CNN
//
//   image 384x384         147,456 words  (written by the camera capture path)
//     -> conv1  C filt 5x5 pad2 -> ReLU -> pool 4x4 ->  C x 96x96
//     -> conv2 16 filt 5x5 pad2 -> ReLU -> pool 2x2 -> 16 x 48x48   36,864
//     -> conv3 16 filt 5x5 pad2 -> ReLU -> pool 2x2 -> 16 x 24x24    9,216
//     -> conv4 16 filt 5x5 pad2 -> ReLU -> pool 2x2 -> 16 x 12x12    2,304
//     -> fc_shared 2305->64 ReLU  (2304 features + colour flag)
//     -> fc_rank 13 / fc_suit 4 / fc_joker 2
//     -> argmax, with the suit argmax CONSTRAINED BY COLOUR
//
// Four conv stages, not three, so that pools of (4,2,2,2) bring 384x384 down to
// the same 12x12 the 96x96 model reached in three -- fc_shared therefore stays
// 2305->64.
//
// DDR3. fc_shared's 147,520 weights are NOT on chip: they were 289 M10K blocks
// (145 even at 10 bits), the largest memory in the design, and they are read
// once per inference in strict order -- the one big array that suits the
// professor's one-read-at-a-time sdram_master on the HPS f2h_sdram0 port.
// Moving them is what buys back 8 conv1 filters and full 8-bit input. The
// conv activations stay on chip: each is re-read 25*C times per output, and
// at DDR3 latency that would cost seconds, not milliseconds.
//
// WHY conv1 POOLS BY 4. It is a device limit, not a modelling preference. An
// M10K holds 512 words at x16, so conv1's pooled output costs:
//     pool 2 -> C x 192 x 192 = 36,864*C words = 72*C blocks  (576 at C=8)
//     pool 4 -> C x  96 x  96 =  9,216*C words = 18*C blocks  (144 at C=8)
// The 5CSEBA6U23I7DK has 553 blocks in total, so pool 2 does not fit even with
// the rest of the design deleted. 96x96 feature maps are this chip's ceiling.
//
// Each conv writes POOLED output directly (conv_layer's POOL parameter). The
// separate maxpool_layer is not instantiated; maxpool_layer.v stays in the
// project regardless because ram_dp and ram_dp_dc are defined at its bottom.
//
// MEMORY ALIASING. p3 and p4 share u_act1 with p1 rather than getting RAMs of
// their own. p1 is dead the moment conv2 finishes, and conv3/conv4 run strictly
// after that, so the space is free. conv4 reads p3 and writes p4 in the same
// RAM at once, which a simple-dual-port memory does fine -- the two regions do
// not overlap. This saves 23 M10K blocks, which the 384x384 image RAM needs.
//
// The layers run in sequence, each triggered by the previous one's done pulse
// -- the same chaining LAB 5 uses in cnn_core.v.
//
// THE SUIT CONSTRAINT IS NOT COSMETIC. The suit head is 4-way over
// {S,C,H,D} and the colour flag restricts the argmax to {S,C} when black or
// {H,D} when red. An unconstrained argmax reproduces exactly the failure the
// Python model showed before this constraint was added: a strictly
// one-directional Heart->Diamond, Spade->Club error pattern. The RTL must
// decode the same way the trained model is evaluated, or hardware and
// software will disagree.
// ---------------------------------------------------------------------------

module card_cnn_core #(
    // conv1 output channels. THE MAIN M10K LEVER: each channel costs 18 blocks
    // in u_act1. Must match the checkpoint -- weights_manifest.json records it.
    parameter CONV1_CH = 8,
    // Image RAM word width. 10 stores the camera's 8-bit luminance shifted up
    // into Q6.10 losslessly; 5 halves the RAM's block count (an M10K holds
    // 1024 words at x10 but 2048 at x5) at the cost of 32 grey levels. The
    // model must have been TRAINED at this depth -- see the trainer's
    // --quant_bits.
    parameter IMG_DW   = 10,
    // Narrowed weight ROM widths; see conv_layer.v's W_BITS note. These must
    // agree with export_weights_rtl.py's NARROW_ROM_BITS or weights are read
    // truncated, silently.
    parameter CONV3_W_BITS = 10,
    parameter CONV4_W_BITS = 10,
    // fc_shared's weights live in HPS DDR3, one Q6.10 value per 32-bit word,
    // starting at this WORD address (byte 0x10000000 >> 2). atlas_main.c copies
    // them there at startup -- card_pipeline.h's FCS_W_DDR_BASE must be this
    // times 4.
    parameter [29:0] FCS_W_WORD_BASE = 30'h0400_0000,
    // fc_shared's weights are Q1.14, not Q6.10 -- export_weights_rtl.py
    // FCS_W_FRAC, recorded as "fcs_w_frac" in weights_manifest.json.
    parameter FCS_W_FRAC = 14
)(
    input  wire         clk,
    input  wire         rst,
    input  wire         start,

    // image write port -- driven by capture_384.v in the camera's clock domain
    input  wire         img_wr_clk,
    input  wire         img_wr_en,
    input  wire [17:0]  img_wr_addr,
    input  wire [IMG_DW-1:0] img_wr_data,

    // host read-back of the captured image, for PGM dumps and the sim
    // round-trip. Shares conv1's read port: the two never overlap, because the
    // host only reads while the core is idle.
    input  wire         img_dbg_en,
    input  wire [17:0]  img_dbg_addr,
    output wire [IMG_DW-1:0] img_dbg_data,

    // colour flag: 1 = red, 0 = black
    input  wire         colour_flag,

    // HPS f2h_sdram0 port (Avalon-MM, 32-bit, word-addressed), used only to
    // fetch fc_shared's weights. Same clock as the core (clk_0, 50 MHz).
    output wire [29:0]  avm_address,
    output wire [7:0]   avm_burstcount,
    output wire         avm_read,
    output wire         avm_write,
    output wire [31:0]  avm_writedata,
    output wire [3:0]   avm_byteenable,
    input  wire         avm_waitrequest,
    input  wire [31:0]  avm_readdata,
    input  wire         avm_readdatavalid,
    output wire         ddr_err,        // weight fetch timed out this inference

    output reg  [3:0]   rank_idx,       // 0..12
    output reg  [1:0]   suit_idx,       // 0=S 1=C 2=H 3=D
    output reg          is_joker,
    output reg  signed [15:0] rank_score,   // winning logit, for confidence
    output reg          done
);
    // ---- geometry ----------------------------------------------------------
    localparam IMG_DIM   = 384;
    localparam IMG_WORDS = IMG_DIM * IMG_DIM;          // 147,456
    localparam P1_WORDS  = CONV1_CH * 96 * 96;
    localparam P2_WORDS  = 16 * 48 * 48;               // 36,864
    localparam P3_WORDS  = 16 * 24 * 24;               //  9,216
    localparam P4_WORDS  = 16 * 12 * 12;               //  2,304

    // p3 and p4 live inside u_act1, after p1 has been consumed
    localparam P3_BASE = 0;
    localparam P4_BASE = P3_WORDS;
    localparam ACT1_WORDS = (P1_WORDS > (P4_BASE + P4_WORDS))
                            ? P1_WORDS : (P4_BASE + P4_WORDS);
    localparam ACT1_AW = $clog2(ACT1_WORDS);

    // ---- image RAM ---------------------------------------------------------
    // Dual-clock: written at the camera pixel rate, read by conv1 at 50 MHz.
    // The 96x96 build had the host read a fabric snapshot RAM, rescale it and
    // write it back into a second RAM; at 384x384 that is two 144-block RAMs
    // and about a million bridge accesses, so the fabric writes this directly.
    wire [17:0] img_rd_addr;
    wire [IMG_DW-1:0] img_rd_data;
    wire [17:0] img_rd_addr_mux = img_dbg_en ? img_dbg_addr : img_rd_addr;

    ram_dp_dc #(.AW(18), .DW(IMG_DW), .DEPTH(IMG_WORDS)) u_img (
        .wr_clk(img_wr_clk), .wr_en(img_wr_en),
        .wr_addr(img_wr_addr), .wr_data(img_wr_data),
        .rd_clk(clk), .rd_addr(img_rd_addr_mux), .rd_data(img_rd_data));

    assign img_dbg_data = img_rd_data;

    // Expand the stored pixel to Q6.10. The capture path stores gray*4 with
    // (10 - IMG_DW) low bits dropped, so shifting back up is exact.
    wire signed [15:0] img_rd_q = $signed({{(16-10){1'b0}},
                                           img_rd_data, {(10-IMG_DW){1'b0}}});

    // ---- activation buffers ------------------------------------------------
    // DEPTH = exact activation size; the default 2^AW depth would waste blocks.
    wire        p1_wr_en;  wire [ACT1_AW-1:0] p1_wr_addr; wire signed [15:0] p1_wr_data;
    wire [ACT1_AW-1:0] p1_rd_addr;
    wire        p3_wr_en;  wire [13:0] p3_wr_addr; wire signed [15:0] p3_wr_data;
    wire [13:0] p3_rd_addr;
    wire        p4_wr_en;  wire [11:0] p4_wr_addr; wire signed [15:0] p4_wr_data;
    wire [11:0] p4_rd_addr;

    wire signed [15:0] act1_rd_data;
    wire        act1_wr_en   = p1_wr_en | p3_wr_en | p4_wr_en;
    wire [ACT1_AW-1:0] act1_wr_addr =
           p1_wr_en ? p1_wr_addr :
           p3_wr_en ? (P3_BASE[ACT1_AW-1:0] + p3_wr_addr)
                    : (P4_BASE[ACT1_AW-1:0] + p4_wr_addr);
    wire signed [15:0] act1_wr_data =
           p1_wr_en ? p1_wr_data : p3_wr_en ? p3_wr_data : p4_wr_data;

    // Which layer owns the read port. conv2 reads p1, conv4 reads p3, fc_shared
    // reads p4; conv3 reads u_p2 instead, so the mux is don't-care then. These
    // are LATCHED flags, not the done pulses: a pulse drops the cycle after it
    // fires and would swing the mux back mid-layer. Same trap the head
    // sequencing below documents.
    reg c3_fin, c4_fin;
    wire [ACT1_AW-1:0] act1_rd_addr =
           c4_fin ? (P4_BASE[ACT1_AW-1:0] + p4_rd_addr) :
           c3_fin ? (P3_BASE[ACT1_AW-1:0] + p3_rd_addr)
                  : p1_rd_addr;

    ram_dp #(.AW(ACT1_AW), .DEPTH(ACT1_WORDS)) u_act1 (
        .clk(clk), .wr_en(act1_wr_en), .wr_addr(act1_wr_addr), .wr_data(act1_wr_data),
        .rd_addr(act1_rd_addr), .rd_data(act1_rd_data));

    wire        p2_wr_en;  wire [15:0] p2_wr_addr; wire signed [15:0] p2_wr_data;
    wire [15:0] p2_rd_addr; wire signed [15:0] p2_rd_data;
    ram_dp #(.AW(16), .DEPTH(P2_WORDS)) u_p2 (
        .clk(clk), .wr_en(p2_wr_en), .wr_addr(p2_wr_addr), .wr_data(p2_wr_data),
        .rd_addr(p2_rd_addr), .rd_data(p2_rd_data));

    wire        sh_wr_en;  wire [5:0]  sh_wr_addr; wire signed [15:0] sh_wr_data;
    wire [5:0]  sh_rd_addr; wire signed [15:0] sh_rd_data;
    ram_dp #(.AW(6)) u_sh (
        .clk(clk), .wr_en(sh_wr_en), .wr_addr(sh_wr_addr), .wr_data(sh_wr_data),
        .rd_addr(sh_rd_addr), .rd_data(sh_rd_data));

    // head outputs: rank 0..12, suit 13..16, joker 17..18
    wire        hd_wr_en;  wire [4:0] hd_wr_addr; wire signed [15:0] hd_wr_data;
    reg  [4:0]  hd_rd_addr; wire signed [15:0] hd_rd_data;
    ram_dp #(.AW(5)) u_hd (
        .clk(clk), .wr_en(hd_wr_en), .wr_addr(hd_wr_addr), .wr_data(hd_wr_data),
        .rd_addr(hd_rd_addr), .rd_data(hd_rd_data));

    // ---- layer sequencing --------------------------------------------------
    // Each conv includes its pool, so the chain is four stages, not eight.
    wire p1_done, p2_done, p3_done, p4_done, sh_done;
    wire rk_done, st_done, jk_done;
    reg  start_c1;

    conv_layer #(.IN_CH(1), .OUT_CH(CONV1_CH), .DIM(384), .POOL(4),
                 .IN_AW(18), .OUT_AW(ACT1_AW),
                 .WFILE("conv1_w.hex"), .BFILE("conv1_b.hex"))
    u_conv1 (.clk(clk), .rst(rst), .start(start_c1), .done(p1_done),
             .in_rd_addr(img_rd_addr), .in_rd_data(img_rd_q),
             .out_wr_addr(p1_wr_addr), .out_wr_data(p1_wr_data), .out_wr_en(p1_wr_en));

    conv_layer #(.IN_CH(CONV1_CH), .OUT_CH(16), .DIM(96), .POOL(2),
                 .IN_AW(ACT1_AW), .OUT_AW(16),
                 .WFILE("conv2_w.hex"), .BFILE("conv2_b.hex"))
    u_conv2 (.clk(clk), .rst(rst), .start(p1_done), .done(p2_done),
             .in_rd_addr(p1_rd_addr), .in_rd_data(act1_rd_data),
             .out_wr_addr(p2_wr_addr), .out_wr_data(p2_wr_data), .out_wr_en(p2_wr_en));

    conv_layer #(.IN_CH(16), .OUT_CH(16), .DIM(48), .POOL(2),
                 .IN_AW(16), .OUT_AW(14), .W_BITS(CONV3_W_BITS),
                 .WFILE("conv3_w.hex"), .BFILE("conv3_b.hex"))
    u_conv3 (.clk(clk), .rst(rst), .start(p2_done), .done(p3_done),
             .in_rd_addr(p2_rd_addr), .in_rd_data(p2_rd_data),
             .out_wr_addr(p3_wr_addr), .out_wr_data(p3_wr_data), .out_wr_en(p3_wr_en));

    conv_layer #(.IN_CH(16), .OUT_CH(16), .DIM(24), .POOL(2),
                 .IN_AW(14), .OUT_AW(12), .W_BITS(CONV4_W_BITS),
                 .WFILE("conv4_w.hex"), .BFILE("conv4_b.hex"))
    u_conv4 (.clk(clk), .rst(rst), .start(p3_done), .done(p4_done),
             .in_rd_addr(p3_rd_addr), .in_rd_data(act1_rd_data),
             .out_wr_addr(p4_wr_addr), .out_wr_data(p4_wr_data), .out_wr_en(p4_wr_en));

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            c3_fin <= 1'b0;
            c4_fin <= 1'b0;
        end else begin
            if (start) begin
                c3_fin <= 1'b0;
                c4_fin <= 1'b0;
            end
            if (p3_done) c3_fin <= 1'b1;
            if (p4_done) c4_fin <= 1'b1;
        end
    end

    // 1.0 in Q6.10. Must track FRAC_BITS in the layer modules.
    wire signed [15:0] colour_val = colour_flag ? 16'sd1024 : 16'sd0;

    // fc_shared reads its weights from DDR3 (W_FROM_DDR) through the
    // professor's sdram_master: one read in flight, one weight per 32-bit word.
    wire        fcs_w_req_rd, fcs_w_resp_valid;
    wire [29:0] fcs_w_req_addr;
    wire signed [15:0] fcs_w_resp_data;

    fc_layer #(.N_IN(2305), .N_OUT(64), .USE_RELU(1), .APPEND_COLOUR(1),
               .IN_AW(12), .OUT_AW(6), .W_FROM_DDR(1), .W_FRAC(FCS_W_FRAC),
               .WFILE("fcs_w.hex"), .BFILE("fcs_b.hex"))
    u_fcs (.clk(clk), .rst(rst), .start(p4_done), .done(sh_done),
           .colour_val(colour_val),
           .in_rd_addr(p4_rd_addr), .in_rd_data(act1_rd_data),
           .out_wr_addr(sh_wr_addr), .out_wr_data(sh_wr_data), .out_wr_en(sh_wr_en),
           .w_req_rd(fcs_w_req_rd), .w_req_addr(fcs_w_req_addr),
           .w_resp_valid(fcs_w_resp_valid), .w_resp_data(fcs_w_resp_data),
           .ddr_err(ddr_err));

    sdram_master #(.BASE_ADDR(FCS_W_WORD_BASE)) u_ddr (
        .clk(clk), .rst(rst),
        .req_addr(fcs_w_req_addr), .req_wdata(16'sd0),
        .req_wr(1'b0), .req_rd(fcs_w_req_rd), .req_ack(),
        .resp_rdata(fcs_w_resp_data), .resp_valid(fcs_w_resp_valid),
        .avm_address(avm_address), .avm_burstcount(avm_burstcount),
        .avm_read(avm_read), .avm_write(avm_write),
        .avm_writedata(avm_writedata), .avm_byteenable(avm_byteenable),
        .avm_waitrequest(avm_waitrequest), .avm_readdata(avm_readdata),
        .avm_readdatavalid(avm_readdatavalid));

    // The three heads share the 64-wide hidden layer. They run sequentially
    // so they can share one read port into u_sh; running them in parallel
    // would need three read ports or three copies of the RAM, and they cost
    // only 64*19 = 1,216 MACs between them -- not worth the area.
    wire [5:0] rk_rd, st_rd, jk_rd;
    wire [4:0] rk_wa, st_wa, jk_wa;
    wire signed [15:0] rk_wd, st_wd, jk_wd;
    wire rk_we, st_we, jk_we;

    fc_layer #(.N_IN(64), .N_OUT(13), .USE_RELU(0), .APPEND_COLOUR(0),
               .IN_AW(6), .OUT_AW(5), .WFILE("fcrank_w.hex"), .BFILE("fcrank_b.hex"))
    u_rank (.clk(clk), .rst(rst), .start(sh_done), .done(rk_done), .colour_val(16'sd0),
            .in_rd_addr(rk_rd), .in_rd_data(sh_rd_data),
            .out_wr_addr(rk_wa), .out_wr_data(rk_wd), .out_wr_en(rk_we),
            .w_req_rd(), .w_req_addr(), .w_resp_valid(1'b0), .w_resp_data(16'sd0), .ddr_err());

    fc_layer #(.N_IN(64), .N_OUT(4), .USE_RELU(0), .APPEND_COLOUR(0),
               .IN_AW(6), .OUT_AW(5), .WFILE("fcsuit_w.hex"), .BFILE("fcsuit_b.hex"))
    u_suit (.clk(clk), .rst(rst), .start(rk_done), .done(st_done), .colour_val(16'sd0),
            .in_rd_addr(st_rd), .in_rd_data(sh_rd_data),
            .out_wr_addr(st_wa), .out_wr_data(st_wd), .out_wr_en(st_we),
            .w_req_rd(), .w_req_addr(), .w_resp_valid(1'b0), .w_resp_data(16'sd0), .ddr_err());

    fc_layer #(.N_IN(64), .N_OUT(2), .USE_RELU(0), .APPEND_COLOUR(0),
               .IN_AW(6), .OUT_AW(5), .WFILE("fcjoker_w.hex"), .BFILE("fcjoker_b.hex"))
    u_joker (.clk(clk), .rst(rst), .start(st_done), .done(jk_done), .colour_val(16'sd0),
             .in_rd_addr(jk_rd), .in_rd_data(sh_rd_data),
             .out_wr_addr(jk_wa), .out_wr_data(jk_wd), .out_wr_en(jk_we),
            .w_req_rd(), .w_req_addr(), .w_resp_valid(1'b0), .w_resp_data(16'sd0), .ddr_err());

    // The three heads share one read port into u_sh, so the mux must know
    // which head is currently running. rk_done / st_done are one-cycle
    // PULSES, not held flags -- using them directly swings the mux back to
    // the rank head the moment the pulse drops, so the suit and joker layers
    // read whatever the rank layer is addressing and produce garbage. Latch
    // them instead.
    reg rank_finished, suit_finished;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            rank_finished <= 1'b0;
            suit_finished <= 1'b0;
        end else begin
            if (start) begin
                rank_finished <= 1'b0;
                suit_finished <= 1'b0;
            end
            if (rk_done) rank_finished <= 1'b1;
            if (st_done) suit_finished <= 1'b1;
        end
    end

    assign sh_rd_addr = suit_finished ? jk_rd :
                        rank_finished ? st_rd : rk_rd;
    assign hd_wr_en   = rk_we | st_we | jk_we;
    assign hd_wr_addr = rk_we ? {1'b0, rk_wa[3:0]} :
                        st_we ? (5'd13 + st_wa) : (5'd17 + jk_wa);
    assign hd_wr_data = rk_we ? rk_wd : st_we ? st_wd : jk_wd;

    // ---- argmax with colour-constrained suit ------------------------------
    localparam A_IDLE=0, A_ADDR=1, A_WAIT=2, A_CMP=3, A_FIN=4;
    reg [2:0] astate;
    reg [4:0] scan;
    reg signed [15:0] best_rank, best_suit, best_joker;
    reg [3:0] best_rank_i;
    reg [1:0] best_suit_i;

    // suit index 0=S 1=C are black, 2=H 3=D are red
    wire suit_allowed = colour_flag ? (scan >= 5'd15) : (scan <= 5'd14);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            astate <= A_IDLE; done <= 0; start_c1 <= 0;
        end else begin
            start_c1 <= 1'b0;
            case (astate)
                A_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        start_c1 <= 1'b1;
                        best_rank  <= -16'sd32768; best_rank_i <= 0;
                        best_suit  <= -16'sd32768; best_suit_i <= 0;
                        best_joker <= -16'sd32768;
                        scan <= 0;
                        astate <= A_ADDR;
                    end
                end
                // wait for the whole chain, then scan the 19 head outputs
                A_ADDR: if (jk_done) begin
                    hd_rd_addr <= scan;
                    astate <= A_WAIT;
                end
                A_WAIT: astate <= A_CMP;
                A_CMP: begin
                    if (scan <= 5'd12) begin
                        if ($signed(hd_rd_data) > best_rank) begin
                            best_rank <= $signed(hd_rd_data);
                            best_rank_i <= scan[3:0];
                        end
                    end else if (scan <= 5'd16) begin
                        // colour constraint applied here
                        if (suit_allowed && $signed(hd_rd_data) > best_suit) begin
                            best_suit   <= $signed(hd_rd_data);
                            best_suit_i <= (scan - 5'd13);   // 13->S 14->C 15->H 16->D
                        end
                    end else if (scan == 5'd17) begin
                        best_joker <= $signed(hd_rd_data);   // "not joker" logit
                    end else begin
                        // index 18 is the "joker" logit; compare against 17
                        is_joker <= ($signed(hd_rd_data) > best_joker);
                    end

                    if (scan == 5'd18) astate <= A_FIN;
                    else begin
                        scan <= scan + 1'b1;
                        hd_rd_addr <= scan + 1'b1;
                        astate <= A_WAIT;
                    end
                end
                A_FIN: begin
                    rank_idx   <= best_rank_i;
                    suit_idx   <= best_suit_i;
                    rank_score <= best_rank;
                    done       <= 1'b1;
                    astate     <= A_IDLE;
                end
                default: astate <= A_IDLE;
            endcase
        end
    end
endmodule
