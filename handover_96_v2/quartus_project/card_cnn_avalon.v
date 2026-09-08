// ---------------------------------------------------------------------------
// card_cnn_avalon.v  --  Avalon-MM slave wrapper for the card CNN
//
// LAB 5 drives the CNN from a bare-metal HPS application. This project runs
// embedded Linux, so the accelerator is exposed as a memory-mapped peripheral
// on the lightweight HPS-to-FPGA bridge and driven from userspace through
// /dev/mem. Same idea, standard interface.
//
// REGISTER MAP (byte offsets from the peripheral base)
//   0x0000  CONTROL   W   bit0 = start (self-clearing pulse)
//                         bit1 = colour flag override enable
//                         bit2 = colour flag value (1 = red)
//   0x0004  STATUS    R   bit0 = done, bit1 = busy
//   0x0008  RANK      R   0..12  -> "2".."A"
//   0x000C  SUIT      R   0=S 1=C 2=H 3=D
//   0x0010  JOKER     R   bit0 = is_joker
//   0x0014  SCORE     R   winning rank logit, Q4.12 signed
//   0x2000  IMAGE     W   2304 words, one 48x48 pixel each (Q4.12 signed)
//                         word index = y*48 + x
//
// The image window sits at 0x2000 so a single address bit (addr[11] in word
// terms) selects registers vs image memory. Writing 2304 individual words
// over the lightweight bridge takes roughly 2304 * ~100 ns = 0.23 ms, which
// is negligible beside the ~46 ms inference. If that ever matters, the image
// would move to a DMA over the full HPS-to-FPGA bridge instead.
// ---------------------------------------------------------------------------

module card_cnn_avalon (
    input  wire         clk,
    input  wire         reset_n,

    // Avalon-MM slave (word addressed)
    input  wire [12:0]  avs_address,
    input  wire         avs_write,
    input  wire [31:0]  avs_writedata,
    input  wire         avs_read,
    output reg  [31:0]  avs_readdata
);

    // colour_flag_hw and irq were removed. The HPS supplies the colour flag
    // through CONTROL bit 1, and card_cnn.c polls STATUS rather than taking an
    // interrupt, so neither port was doing anything -- and dropping them means
    // the component has no conduit to export and nothing to connect in
    // ghrd_top.v.
    localparam REG_CONTROL = 13'h000,
               REG_STATUS  = 13'h001,
               REG_RANK    = 13'h002,
               REG_SUIT    = 13'h003,
               REG_JOKER   = 13'h004,
               REG_SCORE   = 13'h005,
               REG_ID      = 13'h006;   // constant signature, read-path test

    // A register that always returns the same known value, whatever the core
    // is doing. If a read of +0x18 does not return 0xCA5D0001 then the READ
    // PATH itself is broken -- avs_read is not reaching the component, or
    // avs_readdata is not wired back -- and every other register value seen so
    // far is meaningless.
    localparam [31:0] ID_MAGIC = 32'hCA5D0001;

    wire rst = ~reset_n;
    wire is_image = avs_address[12];               // 0x2000 word window
    wire [11:0] img_idx = avs_address[11:0];

    reg        start_pulse;
    reg        colour_override_en;
    reg        colour_override_val;
    reg        busy;

    wire [3:0] rank_idx;
    wire [1:0] suit_idx;
    wire       is_joker;
    wire signed [15:0] rank_score;
    wire       core_done;

    // The colour flag is forced by the HPS through CONTROL bit 1. When a
    // red/black comparator is eventually built in fabric, it connects here in
    // place of the 1'b0.
    wire colour_flag = colour_override_en ? colour_override_val : 1'b0;

    card_cnn_core u_core (
        .clk(clk), .rst(rst), .start(start_pulse),
        .img_wr_en   (avs_write & is_image),
        .img_wr_addr (img_idx),
        .img_wr_data (avs_writedata[15:0]),
        .colour_flag (colour_flag),
        .rank_idx(rank_idx), .suit_idx(suit_idx), .is_joker(is_joker),
        .rank_score(rank_score), .done(core_done)
    );

    // core_done is a ONE-CYCLE PULSE out of card_cnn_core: A_FIN raises it and
    // A_IDLE clears it on the next cycle. At 50 MHz that is 20 ns.
    //
    // Software polls STATUS through /dev/mem roughly every millisecond, so it
    // has essentially no chance of observing a 20 ns pulse -- the accelerator
    // completes correctly and the HPS times out anyway.
    //
    // A testbench cannot expose this: it polls every clock cycle and so always
    // catches the pulse. The bug only appears once a slow observer is watching.
    //
    // done_latched holds the flag until the next start, which is what a
    // polling driver needs.
    reg done_latched;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            start_pulse <= 0; busy <= 0; done_latched <= 0;
            colour_override_en <= 0; colour_override_val <= 0;
        end else begin
            start_pulse <= 1'b0;                   // one-cycle pulse

            if (avs_write && !is_image && avs_address == REG_CONTROL) begin
                if (avs_writedata[0]) begin
                    start_pulse  <= 1'b1;
                    busy         <= 1'b1;
                    done_latched <= 1'b0;          // clear on a new inference
                end
                colour_override_en  <= avs_writedata[1];
                colour_override_val <= avs_writedata[2];
            end

            if (core_done) begin
                busy         <= 1'b0;
                done_latched <= 1'b1;
            end
        end
    end

    always @(posedge clk) begin
        if (avs_read) begin
            case (avs_address)
                REG_STATUS: avs_readdata <= {30'd0, busy, done_latched};
                REG_RANK:   avs_readdata <= {28'd0, rank_idx};
                REG_SUIT:   avs_readdata <= {30'd0, suit_idx};
                REG_JOKER:  avs_readdata <= {31'd0, is_joker};
                REG_SCORE:  avs_readdata <= {{16{rank_score[15]}}, rank_score};
                REG_ID:     avs_readdata <= ID_MAGIC;
                default:    avs_readdata <= 32'd0;
            endcase
        end
    end

endmodule
