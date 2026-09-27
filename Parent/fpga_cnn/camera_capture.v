module camera_capture #(
    // image RAM word width; set once in ghrd_top.v and passed to both this
    // (the writer) and card_cnn_core (the reader)
    parameter IMG_DW = 10
)(
    input  wire        clk50,
    input  wire        clk2_50,
    input  wire        rst_n,             // 来自ghrd_top的hps_fpga_reset_n
    input  wire        capture_trigger_in, // 来自HPS的PIO触发（替代物理按钮）

    // HDMI
    inout  wire        HDMI_I2C_SCL, inout wire HDMI_I2C_SDA,
    inout  wire        HDMI_I2S, inout wire HDMI_LRCLK,
    inout  wire        HDMI_MCLK, inout wire HDMI_SCLK,
    output wire        HDMI_TX_CLK, output wire HDMI_TX_DE,
    output wire [23:0] HDMI_TX_D, output wire HDMI_TX_HS,
    input  wire        HDMI_TX_INT, output wire HDMI_TX_VS,

    // Camera/MIPI
    inout  wire        CAMERA_I2C_SCL, inout wire CAMERA_I2C_SDA,
    output wire        CAMERA_PWDN_n, output wire MIPI_CS_n,
    inout  wire        MIPI_I2C_SCL, inout wire MIPI_I2C_SDA,
    output wire        MIPI_MCLK, input wire MIPI_PIXEL_CLK,
    input  wire [9:0]  MIPI_PIXEL_D,
    input  wire        MIPI_PIXEL_HS, input wire MIPI_PIXEL_VS,
    output wire        MIPI_REFCLK, output wire MIPI_RESET_n,

    // 给HPS查看配置状态用（可选，接到LED或忽略）
    output wire        dbg_mipi_release, dbg_camera_release, dbg_hdmi_ready, dbg_pll_ok,
    output wire [4:0]  dbg_lut_index,
    output wire        dbg_ack, dbg_ready_latched, dbg_hdmi_int,

    // image write port -> card_cnn_core.u_img (dual-clock; card_cnn_core reads
    // it on clk50). capture_384.v owns this now -- the old design's HPS-side
    // snapshot readback + rescale + re-upload is gone; the fabric writes the
    // image RAM directly.
    output wire        img_wr_en,
    output wire [17:0] img_wr_addr,
    output wire [IMG_DW-1:0] img_wr_data,

    // 给HPS读取诊断数据 (addr 0 = red_count (saturated to 16 bit), 1 = colour flag,
    // 2 = vs_count, 3 = pixclk_ticks, 4 = retry_count, 5 = config STEP,
    // 6 = {hdmi_ready, audio_pll_ok, camera_release, mipi_release},
    // 7 = wde_ticks (frame-buffer write-enable activity))
    //        NB pll_ok is the AUDIO PLL (it feeds HDMI_TX_AD7513), not the video
    //        one -- VIDEO_PLL has no locked port wired, but VGA_CLK comes from it
    //        so a completing snapshot already proves it is locked.
    // ghrd_top.v maps these onto snapshot_addr IMG_PIXELS + n, the same order
    // card_pipeline.h's SNAP_* constants have always used -- only IMG_PIXELS
    // moved. Combinational; ghrd_top registers the result to line up with the
    // image RAM's one-cycle read.
    input  wire [2:0]  hps_diag_addr,
    output wire [15:0] hps_diag_data,
    output wire        snapshot_done,
    output wire        snapshot_colour,   // 1 = red card (hardware detector), latched with the snapshot

    // HDMI text layer write port (clk50): the HPS GUI's cells, see text_overlay.v
    input  wire        txt_wr_en,
    input  wire [11:0] txt_wr_addr,
    input  wire [15:0] txt_wr_data
);

    wire RESET_N, RESET_N_DELAY;
    wire cfg_reset_n;
    wire [9:0]  cfg_step;
    wire MIPI_BRIDGE_RELEASE, CAMERA_MIPI_RELAESE;
    wire AUD_CTRL_CLK, PLL_TEST_OK, VGA_CLK;
    wire [9:0] RD_DATA;
    wire [19:0] WR_ADDR, RD_ADDR;
    wire [7:0] RED, GREEN, BLUE;
    wire VGA_HS, VGA_VS, READ_Request, HDMI_READY;
    wire [10:0] cur_x, cur_y;
    wire HDMI_I2S_int;

    // 320x240 display window. The frame buffer holds a Bayer-skipped 320x240
    // image, shown at the top left of the 640x480 frame; the HPS's text layer
    // (text_overlay.v) fills the rest. PV_X/PV_Y sit on the 8x16 text grid, so
    // the preview is exactly text cols 1..40, rows 2..16 and the GUI can frame
    // it. Moving the window is safe: RAW2RGB_J's X/Y counters and the FRAM read
    // address are both relative to disp_win, so the Bayer phase does not move.
    // RGB out of RAW2RGB_J lags READ_Request by 4 clocks (FRAM 2 + line
    // buffer 1 + RAW_RGB_BIN 1), so the HDMI mux uses the delayed window.
    localparam [10:0] PV_X = 11'd8;
    localparam [10:0] PV_Y = 11'd32;
    wire disp_win = READ_Request && (cur_x >= PV_X) && (cur_x < PV_X + 11'd320)
                                 && (cur_y >= PV_Y) && (cur_y < PV_Y + 11'd240);
    reg  [3:0] win_sr;
    always @(posedge VGA_CLK) win_sr <= {win_sr[2:0], disp_win};
    wire disp_win_d = win_sr[3];

    // 1-px green frame around the CNN crop (screen x 72..263, y 56..247).
    // capture_384.v now crops 384x384 RAW camera pixels directly (not via
    // the decimated display buffer), so the box is derived from the display
    // buffer's own 2x downsample of the raw stream: buffer_x = raw_x/2,
    // screen_x = PV_X + buffer_x (disp_win's origin). Raw crop x128..511,
    // y48..431 (see capture_384.v) maps to buffer x 64..255, y 24..215 --
    // centred in the preview and square. The model classifies the card's
    // rank/suit index corner, so aim that corner into the box -- it should
    // fill the box top to bottom. Delayed by the same 4 clocks so it lines up
    // with the image.
    wire crop_edge = disp_win && (
        ((cur_x == PV_X + 11'd64 || cur_x == PV_X + 11'd255) &&
         (cur_y >= PV_Y + 11'd24) && (cur_y < PV_Y + 11'd216)) ||
        ((cur_y == PV_Y + 11'd24 || cur_y == PV_Y + 11'd215) &&
         (cur_x >= PV_X + 11'd64) && (cur_x < PV_X + 11'd256)));
    reg  [3:0] edge_sr;
    always @(posedge VGA_CLK) edge_sr <= {edge_sr[2:0], crop_edge};
    wire crop_edge_d = edge_sr[3];

    assign CAMERA_PWDN_n = 1'b1;
    assign MIPI_CS_n     = 1'b0;
    assign MIPI_RESET_n  = RESET_N;

    RESET_DELAY_HDMI dl (
        .RESET_N (rst_n), .CLK (clk50),
        .READY0  (RESET_N), .READY1 (RESET_N_DELAY)
    );

    MIPI_BRIDGE_CAMERA_Config cfin (
        .RESET_N (cfg_reset_n), .CLK_50 (clk50),
        .MIPI_I2C_SCL (MIPI_I2C_SCL), .MIPI_I2C_SDA (MIPI_I2C_SDA),
        .MIPI_I2C_RELEASE (MIPI_BRIDGE_RELEASE),
        .CAMERA_I2C_SCL (CAMERA_I2C_SCL), .CAMERA_I2C_SDA (CAMERA_I2C_SDA),
        .CAMERA_I2C_RELAESE (CAMERA_MIPI_RELAESE),
        .STEP (cfg_step)
    );

    AUDIO_PLL pll1 (.refclk(clk50), .rst(1'b0), .outclk_0(AUD_CTRL_CLK), .locked(PLL_TEST_OK));
    VIDEO_PLL pll2 (.refclk(clk2_50), .rst(1'b0), .outclk_0(MIPI_REFCLK), .outclk_1(VGA_CLK));

    ON_CHIP_FRAM fra (
        .W_CLK(MIPI_PIXEL_CLK), .W_DE(MIPI_PIXEL_HS & MIPI_PIXEL_VS),
        .W_DATA(MIPI_PIXEL_D[9:0]), .W_CLR(MIPI_PIXEL_VS),
        .R_CLK(VGA_CLK), .R_DATA(RD_DATA), .R_CLR(VGA_VS), .R_DE(disp_win),
        .WR_ADDR(WR_ADDR), .RD_ADDR(RD_ADDR)
    );

    RAW2RGB_J u4 (
        .RST(VGA_VS), .CCD_PIXCLK(VGA_CLK), .mCCD_DATA(RD_DATA[9:0]),
        .VGA_CLK(VGA_CLK), .READ_Request(disp_win),
        .VGA_VS(VGA_VS), .VGA_HS(VGA_HS),
        .oRed(RED), .oGreen(GREEN), .oBlue(BLUE)
    );

    VGA_Controller u1 (
        .iRed(10'd0), .iGreen(10'd0), .iBlue(10'd0),
        .oCurrent_X(cur_x), .oCurrent_Y(cur_y), .oAddress(),
        .oRequest(READ_Request),
        .oVGA_R(), .oVGA_G(), .oVGA_B(),
        .oVGA_HS(VGA_HS), .oVGA_VS(VGA_VS),
        .oVGA_SYNC(), .oVGA_BLANK(), .oVGA_CLOCK(),
        .iCLK(VGA_CLK), .iRST_N(1'b1)
    );

    HDMI_TX_AD7513 hdmi (
        .RESET_N(RESET_N), .CLK_50(clk50),
        .AUD_CTRL_CLK(AUD_CTRL_CLK), .PLL_TEST_OK(PLL_TEST_OK),
        .HDMI_I2C_SCL(HDMI_I2C_SCL), .HDMI_I2C_SDA(HDMI_I2C_SDA),
        .HDMI_I2S(HDMI_I2S_int), .HDMI_LRCLK(HDMI_LRCLK),
        .HDMI_MCLK(HDMI_MCLK), .HDMI_SCLK(HDMI_SCLK),
        .HDMI_TX_INT(HDMI_TX_INT), .READY(HDMI_READY),
        .dbg_lut_index(dbg_lut_index), .dbg_ack(dbg_ack),
        .dbg_ready_latched(dbg_ready_latched)
    );

    assign dbg_hdmi_int = HDMI_TX_INT;

    // HPS text layer: same 4-clock lag as disp_win_d / crop_edge_d / RGB.
    // Glyph pixels may overlay the video; cell backgrounds never cover it.
    wire        txt_fg;
    wire [23:0] txt_fg_rgb, txt_bg_rgb;
    text_overlay u_text (
        .wr_clk(clk50), .wr_en(txt_wr_en), .wr_addr(txt_wr_addr), .wr_data(txt_wr_data),
        .clk(VGA_CLK), .x(cur_x), .y(cur_y),
        .fg_on(txt_fg), .fg_rgb(txt_fg_rgb), .bg_rgb(txt_bg_rgb)
    );

    assign HDMI_TX_CLK = VGA_CLK;
    assign HDMI_TX_D   = crop_edge_d ? 24'h00FF00 :                    // green crop box
                         txt_fg      ? txt_fg_rgb :                    // text glyph pixel
                         disp_win_d  ? {RED, GREEN, BLUE} : txt_bg_rgb;
    assign HDMI_TX_DE  = READ_Request;      // full 640x480 active area for the monitor
    assign HDMI_TX_HS  = VGA_HS;
    assign HDMI_TX_VS  = VGA_VS;
    assign HDMI_I2S    = 1'b0;

    assign dbg_mipi_release   = MIPI_BRIDGE_RELEASE;
    assign dbg_camera_release = CAMERA_MIPI_RELAESE;
    assign dbg_hdmi_ready     = HDMI_READY;
    assign dbg_pll_ok         = PLL_TEST_OK;

    // ---------- 拍照捕获：触发信号来自HPS ----------
    // capture_384 syncs capture_trigger_in itself (MIPI_PIXEL_CLK domain --
    // downsample_96x96's VGA_CLK sync is gone along with the VGA_CLK tap).
    //
    // 384x384 1:1 capture + red/black detector, tapping the RAW MIPI stream
    // directly (not the decimated display buffer -- see capture_384.v's
    // header for why). snapshot_done/snapshot_colour are still exported on
    // clk50 for the HPS PIO path; the values themselves are set on
    // MIPI_PIXEL_CLK and held stable (sticky) well past any metastability
    // window, same as the old downsample_96x96 path relied on.
    wire [17:0] ds_red_count;
    capture_384 #(.IMG_DW(IMG_DW)) u_ds (
        .mipi_clk(MIPI_PIXEL_CLK), .mipi_data(MIPI_PIXEL_D),
        .mipi_hs(MIPI_PIXEL_HS), .mipi_vs(MIPI_PIXEL_VS),
        .capture_trigger(capture_trigger_in),
        .img_wr_en(img_wr_en), .img_wr_addr(img_wr_addr), .img_wr_data(img_wr_data),
        .snapshot_done(snapshot_done), .colour_hw(snapshot_colour),
        .red_count(ds_red_count)
    );
    // Saturate to 16 bits for the diagnostic register -- red_count is a count
    // of red pixels in a 147,456-pixel crop (18 bits), but RED_THRESH (1536)
    // means a real reading rarely approaches even 16 bits; saturating rather
    // than wrapping keeps a debug read from showing a confusingly small
    // wrapped number if it ever does.
    wire [15:0] ds_red_count_sat = (ds_red_count > 18'd65535) ? 16'hFFFF
                                                               : ds_red_count[15:0];

    // ---------- camera link diagnostics + config auto-retry ----------
    // The I2C config sequence runs exactly once off RESET_N_DELAY and nothing
    // retries it, so one NACK leaves the camera dark forever with no visible
    // symptom -- the snapshot just reads all zeros and the CNN returns its bias
    // class. Two additions: counters the HPS can read, and a watchdog that
    // re-runs the sequence when no frames arrive.

    // MIPI_PIXEL_VS is ~60 Hz. Two-flop it into clk50 and count edges THERE, so
    // the counter lives entirely in the read clock domain and the HPS never
    // samples a multi-bit value mid-change.
    reg [2:0] vs_sync;
    always @(posedge clk50) vs_sync <= {vs_sync[1:0], MIPI_PIXEL_VS};
    wire vs_rise_50 = vs_sync[1] & ~vs_sync[2];

    reg [15:0] vs_count;
    always @(posedge clk50) if (vs_rise_50) vs_count <= vs_count + 1'b1;

    // Pixel-clock liveness. Divide MIPI_PIXEL_CLK down to one slowly toggling
    // bit, synchronise that single bit, count its edges in clk50 -- same trick,
    // no wide CDC. If MIPI_PIXEL_CLK is dead this simply never advances, which
    // is exactly the thing we need to be able to tell apart from a config fault.
    reg [15:0] pixclk_div;
    always @(posedge MIPI_PIXEL_CLK) pixclk_div <= pixclk_div + 1'b1;
    reg [2:0] pix_sync;
    always @(posedge clk50) pix_sync <= {pix_sync[1:0], pixclk_div[15]};
    reg [15:0] pixclk_ticks;
    always @(posedge clk50)
        if (pix_sync[1] ^ pix_sync[2]) pixclk_ticks <= pixclk_ticks + 1'b1;

    // Sticky "the frame-buffer write enable has fired at least once". W_DE is
    // MIPI_PIXEL_HS & MIPI_PIXEL_VS, so this is the single most direct evidence
    // that pixels ever reached ON_CHIP_FRAM. Never cleared, so it also tells
    // "never worked since config" apart from "worked, then stopped".
    // A sticky flag does NOT work here: with 1'b1 as its only assignment and no
    // reset, synthesis folds it to a constant and strips it (Quartus did exactly
    // that -- "stuck at VCC", removed during synthesis), so it would read 1 even
    // on a dead link. A counter has real state and survives.
    reg [15:0] wde_div;
    always @(posedge MIPI_PIXEL_CLK)
        if (MIPI_PIXEL_HS & MIPI_PIXEL_VS) wde_div <= wde_div + 1'b1;
    reg [2:0] wde_sync;
    always @(posedge clk50) wde_sync <= {wde_sync[1:0], wde_div[15]};
    reg [15:0] wde_ticks;
    always @(posedge clk50)
        if (wde_sync[1] ^ wde_sync[2]) wde_ticks <= wde_ticks + 1'b1;

    // Watchdog: ~250 ms with no frame -> pulse the config module's reset.
    // Dropping RESET_N restarts MIPI_BRIDGE_CONFIG, and MIPI_CAMERA_CONFIG is
    // gated behind its completion, so the whole I2C sequence re-runs.
    // The deadline depends on whether the config has finished. The camera LUT is
    // several hundred I2C writes and takes well over 250 ms to play out, so an
    // unconditional 250 ms timeout would keep restarting the sequence before it
    // could ever complete -- a livelock that looks exactly like the fault it is
    // meant to fix. Only hold the tight deadline once CAMERA_I2C_RELAESE says
    // the sequence is done; while it is still running, allow 2 s before forcing
    // a restart (which then covers a genuinely wedged I2C bus).
    localparam [26:0] WD_STREAM = 27'd12_500_000;    // 250 ms at 50 MHz
    localparam [26:0] WD_CONFIG = 27'd100_000_000;   // 2 s at 50 MHz
    localparam [26:0] WD_PULSE  = 27'd512;
    wire [26:0] wd_limit = CAMERA_MIPI_RELAESE ? WD_STREAM : WD_CONFIG;

    reg [26:0] wd_cnt;
    reg        retry_pulse;
    reg [15:0] retry_count;
    always @(posedge clk50) begin
        if (retry_pulse) begin
            if (wd_cnt >= WD_PULSE) begin wd_cnt <= 27'd0; retry_pulse <= 1'b0; end
            else wd_cnt <= wd_cnt + 1'b1;
        end else if (vs_rise_50) begin
            wd_cnt <= 27'd0;                                  // frames arriving
        end else if (wd_cnt >= wd_limit) begin
            wd_cnt <= 27'd0; retry_pulse <= 1'b1; retry_count <= retry_count + 1'b1;
        end else begin
            wd_cnt <= wd_cnt + 1'b1;
        end
    end

    assign cfg_reset_n = RESET_N_DELAY & ~retry_pulse;

    // A small fixed-size diagnostic register file, not the image-address-space
    // mux the 96x96 build used (there is no snapshot RAM here to share a read
    // port with -- capture_384 writes the image RAM directly). Combinational:
    // nothing here is registered RAM, so no read-address delay is needed.
    assign hps_diag_data =
        (hps_diag_addr == 3'd0) ? ds_red_count_sat :
        (hps_diag_addr == 3'd1) ? {15'b0, snapshot_colour} :
        (hps_diag_addr == 3'd2) ? vs_count :
        (hps_diag_addr == 3'd3) ? pixclk_ticks :
        (hps_diag_addr == 3'd4) ? retry_count :
        (hps_diag_addr == 3'd5) ? {6'b0, cfg_step} :
        (hps_diag_addr == 3'd6) ? {12'b0, HDMI_READY, PLL_TEST_OK,
                                   CAMERA_MIPI_RELAESE, MIPI_BRIDGE_RELEASE} :
                                  wde_ticks;

endmodule
