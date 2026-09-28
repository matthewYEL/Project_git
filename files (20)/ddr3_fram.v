// ddr3_fram.v -- DDR3-backed camera frame buffer.
//
// DROP-IN REPLACEMENT for ON_CHIP_FRAM.v. Same port list plus an Avalon-MM
// master group, so camera_capture.v changes only by (a) renaming the
// instance's module and (b) plumbing the avm_* ports up to ghrd_top.
//
// WHY THIS EXISTS
//   ON_CHIP_FRAM's FRAM_BUFF holds the whole 320x240 Bayer frame in M10K --
//   the single largest block consumer in the design, and the reason the
//   camera and the CNN accelerator could not fit together. Moving it to DDR3
//   through hps_0.f2h_sdram0_data frees those blocks and satisfies the
//   project spec's "External DRAM interface" requirement directly.
//
// WHY BURSTING IS NOT OPTIONAL HERE
//   The existing sdram_master.v issues one transaction at a time and waits.
//   At ~10-20 cycles per 32-bit word that tops out around 2.5-5 M words/s,
//   against the 4.61 M words/s this buffer needs sustained (2.30 M write +
//   2.30 M display read-back, both continuous at 60 fps). Too close to the
//   ceiling to be safe. Bursting 16 words amortises the per-transaction
//   overhead and drops the requirement to ~288K bursts/s -- about 9% of the
//   port's 200 MB/s peak. That headroom is what makes this design work.
//
// ADDRESSING
//   f2h_sdram0_data is WORD-addressed (addressUnits=WORDS, 32-bit data) --
//   the same fact sdram_master.v's header records having learned the hard
//   way. avm_address is therefore a word index: BASE_WORD + word offset,
//   with NO <<2 anywhere.
//
// PIXEL PACKING
//   Two 10-bit Bayer pixels per 32-bit word: first pixel in [15:0], second in
//   [31:16], each zero-extended. 320x240 = 76,800 pixels = 38,400 words per
//   frame, and the pixel count is even so pairs never straddle a frame.
//
// !!! LATENCY -- READ THIS BEFORE DEBUGGING A SHIFTED IMAGE !!!
//   R_DATA is pipelined through two registers so the read latency matches
//   FRAM_BUFF's altsyncram (2 cycles). That is what keeps camera_capture.v's
//   win_sr[3] and downsample_96x96.v's LAT=4 correct. If the image comes out
//   shifted by a pixel or two horizontally -- a diagonal tear, or the green
//   crop box not lining up with what the CNN actually sees -- this is the
//   first thing to re-check, NOT the crop coordinates. Verify in simulation
//   that R_DATA appears exactly 2 R_CLK edges after R_DE, as it did before.
//
// WHAT IS NOT PROVEN
//   This has not been simulated or run on hardware. The clock-domain
//   crossings (three: W_CLK->avm_clk, avm_clk->R_CLK, and the frame-start
//   resync) and the read prefetch are the parts most likely to be wrong.
//   Simulate against a golden vector set before programming anything.
module ddr3_fram #(
    // Word index of the frame buffer's first word in DDR3. MUST point at a
    // region the Linux kernel is not using -- see the notes accompanying this
    // file. Default is deliberately NOT 0: address 0 is where the kernel
    // lives, and a collision there corrupts silently and looks like an RTL
    // bug for days.
    parameter [29:0] BASE_WORD  = 30'h0F00_0000,
    parameter integer BURST     = 16,
    parameter integer WORDS_PER_FRAME = 38400      // 320*240/2
)(
    // ---- original ON_CHIP_FRAM port list ----
    input  wire        W_CLK,
    input  wire        R_CLK,
    input  wire        W_DE,
    input  wire [9:0]  W_DATA,
    output wire [9:0]  R_DATA,
    input  wire        W_CLR,
    input  wire        R_CLR,
    input  wire        R_DE,
    output wire [19:0] WR_ADDR,
    output wire [19:0] RD_ADDR,

    // ---- Avalon-MM master to hps_0.f2h_sdram0_data ----
    input  wire        avm_clk,
    input  wire        avm_rst,
    output reg  [29:0] avm_address,
    output reg  [7:0]  avm_burstcount,
    output reg         avm_read,
    output reg         avm_write,
    output wire [31:0] avm_writedata,
    output wire [3:0]  avm_byteenable,
    input  wire        avm_waitrequest,
    input  wire [31:0] avm_readdata,
    input  wire        avm_readdatavalid,

    // ---- debug: non-zero means the display starved or the camera overran.
    // Wire these to spare PIO bits. If rd_underrun climbs, the read FIFO is
    // not being refilled fast enough (raise RFIFO_DEPTH or the arbiter's read
    // priority); if wr_overflow climbs, camera pixels are being dropped.
    output reg  [15:0] dbg_rd_underrun,
    output reg  [15:0] dbg_wr_overflow
);

    localparam RFIFO_DEPTH = 1024;   // ~3 display lines of prefetch
    localparam WFIFO_DEPTH = 512;

    assign avm_byteenable = 4'b1111;

    // =====================================================================
    // WRITE SIDE -- W_CLK domain (MIPI_PIXEL_CLK)
    // =====================================================================
    // The 2x2-block skip is preserved exactly as ON_CHIP_FRAM did it: keep
    // pixels with x%4 in {0,1} and y%4 in {0,1}, gating BOTH the position
    // counter and the write enable so the Bayer colour phase stays valid.
    reg [9:0] wx;
    reg [8:0] wy;
    reg       rDE, rVS;
    always @(posedge W_CLK) begin
        rDE <= W_DE;
        rVS <= W_CLR;
        if (!W_DE)            wx <= 10'd0;
        else                  wx <= wx + 1'b1;
        if (!rVS & W_CLR)     wy <= 9'd0;
        else if (rDE & !W_DE) wy <= wy + 1'b1;
    end

    wire keep   = ~wx[1] & ~wy[1];
    wire W_DE_K = W_DE & keep;

    // Pack pixel pairs into 32-bit words.
    reg        wpair_phase;
    reg [15:0] wpair_lo;
    reg        wfifo_wrreq;
    reg [31:0] wfifo_data;
    always @(posedge W_CLK) begin
        wfifo_wrreq <= 1'b0;
        if (W_CLR) begin
            wpair_phase <= 1'b0;
        end else if (W_DE_K) begin
            if (!wpair_phase) begin
                wpair_lo    <= {6'b0, W_DATA};
                wpair_phase <= 1'b1;
            end else begin
                wfifo_data  <= {{6'b0, W_DATA}, wpair_lo};
                wfifo_wrreq <= 1'b1;
                wpair_phase <= 1'b0;
            end
        end
    end

    wire        wfifo_full, wfifo_empty;
    wire [31:0] wfifo_q;
    wire [8:0]  wfifo_rdusedw;
    reg         wfifo_rdreq;

    always @(posedge W_CLK)
        if (W_CLR) dbg_wr_overflow <= 16'd0;
        else if (wfifo_wrreq && wfifo_full) dbg_wr_overflow <= dbg_wr_overflow + 1'b1;

    dcfifo #(
        .lpm_width(32), .lpm_numwords(WFIFO_DEPTH), .lpm_widthu(9),
        .lpm_showahead("ON"), .clocks_are_synchronized("FALSE"),
        .use_eab("ON"), .overflow_checking("ON"), .underflow_checking("ON")
    ) wfifo (
        .wrclk(W_CLK), .wrreq(wfifo_wrreq && !wfifo_full), .data(wfifo_data),
        .wrfull(wfifo_full), .wrempty(), .wrusedw(),
        .rdclk(avm_clk), .rdreq(wfifo_rdreq), .q(wfifo_q),
        .rdempty(wfifo_empty), .rdfull(), .rdusedw(wfifo_rdusedw),
        .aclr(1'b0)
    );
    assign avm_writedata = wfifo_q;

    // =====================================================================
    // FRAME-START RESYNC
    // =====================================================================
    // Both address counters restart at their frame boundary. W_CLR (camera
    // VS) and R_CLR (VGA VS) are in their own domains, so each is edge
    // detected and passed into avm_clk as a single-cycle pulse. Without this
    // the buffer drifts by a frame and the display shows a torn mix of two.
    reg [2:0] wclr_sync, rclr_sync;
    always @(posedge avm_clk) begin
        wclr_sync <= {wclr_sync[1:0], W_CLR};
        rclr_sync <= {rclr_sync[1:0], R_CLR};
    end
    wire wframe_start = wclr_sync[1] & ~wclr_sync[2];
    wire rframe_start = rclr_sync[1] & ~rclr_sync[2];

    // =====================================================================
    // READ SIDE -- prefetch into a FIFO, R_CLK domain pops pixels
    // =====================================================================
    // The display read pattern is perfectly sequential (linear 0..WORDS-1,
    // restarted by VS), which is exactly what makes prefetching safe: the
    // next address is always known, so bursts can run ahead of the raster.
    wire        rfifo_full, rfifo_empty;
    wire [31:0] rfifo_q;
    wire [9:0]  rfifo_wrusedw;
    reg         rfifo_wrreq;
    reg         rfifo_aclr;

    dcfifo #(
        .lpm_width(32), .lpm_numwords(RFIFO_DEPTH), .lpm_widthu(10),
        .lpm_showahead("ON"), .clocks_are_synchronized("FALSE"),
        .use_eab("ON"), .overflow_checking("ON"), .underflow_checking("ON")
    ) rfifo (
        .wrclk(avm_clk), .wrreq(rfifo_wrreq && !rfifo_full), .data(avm_readdata),
        .wrfull(rfifo_full), .wrempty(), .wrusedw(rfifo_wrusedw),
        .rdclk(R_CLK), .rdreq(rfifo_pop), .q(rfifo_q),
        .rdempty(rfifo_empty), .rdfull(), .rdusedw(),
        .aclr(rfifo_aclr)
    );

    // Unpack: two pixels per word, low half first. Pop only after the second.
    reg  rd_phase;
    wire rfifo_pop = R_DE & rd_phase & ~rfifo_empty;
    always @(posedge R_CLK)
        if (R_CLR)      rd_phase <= 1'b0;
        else if (R_DE)  rd_phase <= ~rd_phase;

    wire [9:0] pix_sel = rd_phase ? rfifo_q[25:16] : rfifo_q[9:0];

    // Two pipeline registers to match FRAM_BUFF's 2-cycle altsyncram latency.
    // See the latency warning in the header before changing this.
    reg [9:0] rdata_r1, rdata_r2;
    always @(posedge R_CLK) begin
        rdata_r1 <= rfifo_empty ? 10'd0 : pix_sel;
        rdata_r2 <= rdata_r1;
    end
    assign R_DATA = rdata_r2;

    always @(posedge R_CLK)
        if (R_CLR) dbg_rd_underrun <= 16'd0;
        else if (R_DE && rfifo_empty) dbg_rd_underrun <= dbg_rd_underrun + 1'b1;

    // =====================================================================
    // BURST MASTER -- avm_clk domain, shared read/write port
    // =====================================================================
    // One Avalon master port serves both directions, so read and write bursts
    // are mutually exclusive. The arbiter alternates with a slight bias to
    // reads: a starved read FIFO tears the visible display, whereas the write
    // FIFO has enough depth to ride out a few bursts. At 9% utilisation
    // neither should ever actually starve -- the debug counters above are
    // there to prove it rather than to assume it.
    localparam S_IDLE = 3'd0, S_WR = 3'd1, S_RD_CMD = 3'd2, S_RD_DATA = 3'd3;
    reg [2:0]  state;
    reg [7:0]  beat;
    reg [29:0] wr_word, rd_word;
    reg        last_was_read;

    // Free space in the read FIFO, computed at 11 bits so the RFIFO_DEPTH
    // subtraction cannot wrap when the FIFO is completely full.
    wire [10:0] rfifo_space = RFIFO_DEPTH[10:0] - {1'b0, rfifo_wrusedw};
    wire wr_ready = (wfifo_rdusedw >= BURST[8:0]);
    wire rd_ready = (rfifo_space   >= BURST[10:0]);

    always @(posedge avm_clk or posedge avm_rst) begin
        if (avm_rst) begin
            state <= S_IDLE; avm_read <= 1'b0; avm_write <= 1'b0;
            wr_word <= 30'd0; rd_word <= 30'd0; beat <= 8'd0;
            wfifo_rdreq <= 1'b0; rfifo_wrreq <= 1'b0;
            rfifo_aclr <= 1'b0; last_was_read <= 1'b0;
        end else begin
            wfifo_rdreq <= 1'b0;
            rfifo_wrreq <= 1'b0;
            rfifo_aclr  <= 1'b0;

            if (wframe_start) wr_word <= 30'd0;
            if (rframe_start) begin
                rd_word    <= 30'd0;
                rfifo_aclr <= 1'b1;     // drop stale prefetch from last frame
            end

            case (state)
                S_IDLE: begin
                    if (rd_ready && (!wr_ready || !last_was_read)) begin
                        avm_address    <= BASE_WORD + rd_word;
                        avm_burstcount <= BURST[7:0];
                        avm_read       <= 1'b1;
                        beat           <= 8'd0;
                        last_was_read  <= 1'b1;
                        state          <= S_RD_CMD;
                    end else if (wr_ready) begin
                        avm_address    <= BASE_WORD + wr_word;
                        avm_burstcount <= BURST[7:0];
                        avm_write      <= 1'b1;
                        beat           <= 8'd0;
                        last_was_read  <= 1'b0;
                        state          <= S_WR;
                    end
                end

                // Burst write: address is presented once, then BURST beats of
                // writedata, each accepted when waitrequest is low. The FIFO
                // is in showahead mode so q already holds the next word.
                S_WR: begin
                    if (!avm_waitrequest) begin
                        wfifo_rdreq <= 1'b1;
                        if (beat == BURST[7:0] - 8'd1) begin
                            avm_write <= 1'b0;
                            wr_word   <= (wr_word + BURST >= WORDS_PER_FRAME)
                                       ? 30'd0 : wr_word + BURST;
                            state     <= S_IDLE;
                        end else begin
                            beat <= beat + 1'b1;
                        end
                    end
                end

                S_RD_CMD: begin
                    if (!avm_waitrequest) begin
                        avm_read <= 1'b0;
                        rd_word  <= (rd_word + BURST >= WORDS_PER_FRAME)
                                  ? 30'd0 : rd_word + BURST;
                        state    <= S_RD_DATA;
                    end
                end

                // readdatavalid arrives independently of waitrequest and may
                // have gaps; count beats rather than assuming they are
                // contiguous.
                S_RD_DATA: begin
                    if (avm_readdatavalid) begin
                        rfifo_wrreq <= 1'b1;
                        if (beat == BURST[7:0] - 8'd1) state <= S_IDLE;
                        else                           beat  <= beat + 1'b1;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // Kept for port compatibility with ON_CHIP_FRAM; camera_capture.v declares
    // these wires but does not use them.
    assign WR_ADDR = {wr_word[18:0], 1'b0};
    assign RD_ADDR = {rd_word[18:0], 1'b0};

endmodule
