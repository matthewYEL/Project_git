// ---------------------------------------------------------------------------
// card_cnn_core.v  --  top-level card recognition CNN
//
//   image 96x96            9,216 words   (written by HPS or camera snapshot)
//     -> conv1  8 filt 5x5 pad2  -> ReLU -> pool 2x2 ->  8 x 48x48  18,432
//     -> conv2 16 filt 5x5 pad2  -> ReLU -> pool 2x2 -> 16 x 24x24   9,216
//     -> conv3 16 filt 5x5 pad2  -> ReLU -> pool 2x2 -> 16 x 12x12   2,304
//     -> fc_shared 2305->64 ReLU  (2304 features + colour flag)
//     -> fc_rank 13 / fc_suit 4 / fc_joker 2
//     -> argmax, with the suit argmax CONSTRAINED BY COLOUR
//
// Three conv stages, not two, so that three pools bring 96x96 down to the same
// 12x12 the 48x48 model reached in two -- fc_shared therefore stays 2305->64.
// With two stages the flattened vector would be 16x24x24 = 9,216 and fc_shared
// would need 590K weights, which does not fit.
//
// Each conv writes POOLED output directly (conv_layer's FUSE_POOL). The
// separate maxpool_layer is no longer instantiated: materialising conv1's
// unpooled 8 x 96x96 map would cost 144 M10K blocks on its own. maxpool_layer.v
// stays in the project regardless -- ram_dp is defined at the bottom of it.
//
// The layers run in sequence, each triggered by the previous one's done
// pulse -- the same chaining LAB 5 uses in cnn_core.v.
//
// THE SUIT CONSTRAINT IS NOT COSMETIC. The suit head is 4-way over
// {S,C,H,D} and the colour flag restricts the argmax to {S,C} when black or
// {H,D} when red. An unconstrained argmax reproduces exactly the failure the
// Python model showed before this constraint was added: a strictly
// one-directional Heart->Diamond, Spade->Club error pattern. The RTL must
// decode the same way the trained model is evaluated, or hardware and
// software will disagree.
// ---------------------------------------------------------------------------

module card_cnn_core (
    input  wire         clk,
    input  wire         rst,
    input  wire         start,

    // image write port (HPS or camera snapshot)
    input  wire         img_wr_en,
    input  wire [13:0]  img_wr_addr,
    input  wire signed [15:0] img_wr_data,

    // colour flag: 1 = red, 0 = black
    input  wire         colour_flag,

    output reg  [3:0]   rank_idx,       // 0..12
    output reg  [1:0]   suit_idx,       // 0=S 1=C 2=H 3=D
    output reg          is_joker,
    output reg  signed [15:0] rank_score,   // winning logit, for confidence
    output reg          done
);
    // ---- activation buffers ----------------------------------------------
    wire [13:0] img_rd_addr;   wire signed [15:0] img_rd_data;
    // DEPTH = exact activation size (96*96, 8*48*48, 16*24*24, 16*12*12); the
    // default 2^AW depth would cost ~55 extra M10K blocks. There is no separate
    // conv-output buffer any more -- each conv pools as it writes, so only the
    // pooled maps are stored.
    ram_dp #(.AW(14), .DEPTH(9216)) u_img (
        .clk(clk), .wr_en(img_wr_en), .wr_addr(img_wr_addr), .wr_data(img_wr_data),
        .rd_addr(img_rd_addr), .rd_data(img_rd_data));

    wire        p1_wr_en;  wire [14:0] p1_wr_addr; wire signed [15:0] p1_wr_data;
    wire [14:0] p1_rd_addr; wire signed [15:0] p1_rd_data;
    ram_dp #(.AW(15), .DEPTH(18432)) u_p1 (
        .clk(clk), .wr_en(p1_wr_en), .wr_addr(p1_wr_addr), .wr_data(p1_wr_data),
        .rd_addr(p1_rd_addr), .rd_data(p1_rd_data));

    wire        p2_wr_en;  wire [13:0] p2_wr_addr; wire signed [15:0] p2_wr_data;
    wire [13:0] p2_rd_addr; wire signed [15:0] p2_rd_data;
    ram_dp #(.AW(14), .DEPTH(9216)) u_p2 (
        .clk(clk), .wr_en(p2_wr_en), .wr_addr(p2_wr_addr), .wr_data(p2_wr_data),
        .rd_addr(p2_rd_addr), .rd_data(p2_rd_data));

    wire        p3_wr_en;  wire [11:0] p3_wr_addr; wire signed [15:0] p3_wr_data;
    wire [11:0] p3_rd_addr; wire signed [15:0] p3_rd_data;
    ram_dp #(.AW(12), .DEPTH(2304)) u_p3 (
        .clk(clk), .wr_en(p3_wr_en), .wr_addr(p3_wr_addr), .wr_data(p3_wr_data),
        .rd_addr(p3_rd_addr), .rd_data(p3_rd_data));

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

    // ---- layer sequencing -------------------------------------------------
    // Each conv includes its pool, so the chain is three stages, not six.
    wire p1_done, p2_done, p3_done, sh_done;
    wire rk_done, st_done, jk_done;
    reg  start_c1;

    conv_layer #(.IN_CH(1), .OUT_CH(8), .DIM(96), .IN_AW(14), .OUT_AW(15),
                 .WFILE("conv1_w.hex"), .BFILE("conv1_b.hex"))
    u_conv1 (.clk(clk), .rst(rst), .start(start_c1), .done(p1_done),
             .in_rd_addr(img_rd_addr), .in_rd_data(img_rd_data),
             .out_wr_addr(p1_wr_addr), .out_wr_data(p1_wr_data), .out_wr_en(p1_wr_en));

    conv_layer #(.IN_CH(8), .OUT_CH(16), .DIM(48), .IN_AW(15), .OUT_AW(14),
                 .WFILE("conv2_w.hex"), .BFILE("conv2_b.hex"))
    u_conv2 (.clk(clk), .rst(rst), .start(p1_done), .done(p2_done),
             .in_rd_addr(p1_rd_addr), .in_rd_data(p1_rd_data),
             .out_wr_addr(p2_wr_addr), .out_wr_data(p2_wr_data), .out_wr_en(p2_wr_en));

    conv_layer #(.IN_CH(16), .OUT_CH(16), .DIM(24), .IN_AW(14), .OUT_AW(12),
                 .WFILE("conv3_w.hex"), .BFILE("conv3_b.hex"))
    u_conv3 (.clk(clk), .rst(rst), .start(p2_done), .done(p3_done),
             .in_rd_addr(p2_rd_addr), .in_rd_data(p2_rd_data),
             .out_wr_addr(p3_wr_addr), .out_wr_data(p3_wr_data), .out_wr_en(p3_wr_en));

    // 1.0 in Q6.10. Must track FRAC_BITS in the layer modules.
    wire signed [15:0] colour_val = colour_flag ? 16'sd1024 : 16'sd0;

    fc_layer #(.N_IN(2305), .N_OUT(64), .USE_RELU(1), .APPEND_COLOUR(1),
               .IN_AW(12), .OUT_AW(6), .WFILE("fcs_w.hex"), .BFILE("fcs_b.hex"))
    u_fcs (.clk(clk), .rst(rst), .start(p3_done), .done(sh_done),
           .colour_val(colour_val),
           .in_rd_addr(p3_rd_addr), .in_rd_data(p3_rd_data),
           .out_wr_addr(sh_wr_addr), .out_wr_data(sh_wr_data), .out_wr_en(sh_wr_en));

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
            .out_wr_addr(rk_wa), .out_wr_data(rk_wd), .out_wr_en(rk_we));

    fc_layer #(.N_IN(64), .N_OUT(4), .USE_RELU(0), .APPEND_COLOUR(0),
               .IN_AW(6), .OUT_AW(5), .WFILE("fcsuit_w.hex"), .BFILE("fcsuit_b.hex"))
    u_suit (.clk(clk), .rst(rst), .start(rk_done), .done(st_done), .colour_val(16'sd0),
            .in_rd_addr(st_rd), .in_rd_data(sh_rd_data),
            .out_wr_addr(st_wa), .out_wr_data(st_wd), .out_wr_en(st_we));

    fc_layer #(.N_IN(64), .N_OUT(2), .USE_RELU(0), .APPEND_COLOUR(0),
               .IN_AW(6), .OUT_AW(5), .WFILE("fcjoker_w.hex"), .BFILE("fcjoker_b.hex"))
    u_joker (.clk(clk), .rst(rst), .start(st_done), .done(jk_done), .colour_val(16'sd0),
             .in_rd_addr(jk_rd), .in_rd_data(sh_rd_data),
             .out_wr_addr(jk_wa), .out_wr_data(jk_wd), .out_wr_en(jk_we));

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
