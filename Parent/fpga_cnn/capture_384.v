// 384x384 1:1 snapshot capture + red/black detector.
// Replaces downsample_96x96.v.
//
// WHY THIS TAPS A DIFFERENT POINT IN THE PIPELINE. downsample_96x96.v read
// from the DISPLAY stream (RAW2RGB_J's output, driven off FRAM_BUFF's replay
// of the 320x240 buffer -- itself already a 2x2-block-decimated copy of the
// camera's 640x480 frame; see ON_CHIP_FRAM.v). A 384-wide crop cannot come out
// of a 320-wide buffer. This module taps MIPI_PIXEL_D/HS/VS/CLK directly --
// the same raw signals ON_CHIP_FRAM.v's write side uses -- so every CNN input
// pixel is a real, distinct camera sample, not an interpolated stretch.
//
// DEMOSAIC. The raw MIPI stream is single-channel Bayer data, not RGB. Rather
// than write a new Bayer-to-RGB scheme (getting the sensor's CFA phase wrong
// produces a plausible-looking but colour-shifted image, silently), this
// re-instantiates RAW2RGB_J -- the same module the live HDMI preview already
// uses, proven correct on this hardware -- fed directly by the raw stream
// instead of by a buffer replay. RAW2RGB_J is a single-clock-domain module
// (its own vestigial CCD_PIXCLK port is dead; everything runs off what it
// calls VGA_CLK), so MIPI_PIXEL_CLK stands in for that clock here.
//
// PIPELINE LATENCY. downsample_96x96.v's LAT=4 came from three sources: the
// FRAM read (address register + registered RAM output, 2 cycles),
// Line_Buffer_J's registered read (1), RAW_RGB_BIN's registered output (1).
// This module has no FRAM stage in its path -- RAW2RGB_J is fed live -- so
// only the latter two apply: LAT=2. Column/row counters are tracked here
// (RAW2RGB_J does not export its own) and delayed by LAT_RGB to line up with
// the RGB it emits, the same shift-register idiom camera_capture.v's own
// disp_win/crop_edge already use for the display path's LAT=4.
//
// THIS IS THE PART OF THE DESIGN LEAST PROVEN BY MEASUREMENT: nobody has
// captured a real frame through this exact tap point yet. Verify by dumping a
// capture as PGM and eyeballing it (or diffing against a display-path capture
// of the same scene) before trusting it for training-data collection.
//
// What that check needs to rule out is a COLOUR swap (R<->B would break the
// red/black detector) or a grossly wrong window -- not pixel-exact alignment.
// RAW2RGB_J builds each output from the two previously buffered rows, so the
// image sits ~1-2 rows above the counters here, and its address counter lags
// the data bus by a column; both are consistent frame to frame. The model is
// trained with +/-5% translation (RandomAffine, ~19 px at 384), so a 1-2 px
// offset is invisible to it.
//
// Cell sums are single camera pixels now, max 255 (was 2 px / 510 at 96x96,
// 25 px / 6375 at 48x48). The image RAM stores gray<<2 (approximately Q6.10,
// max 1020 of 1024) at IMG_DW bits -- IMG_DW<10 keeps only the top bits,
// which is the M10K-budget trim the trainer's --quant_bits matches.
//
// HPS side: no read-back port here. capture_384 writes card_cnn_core's image
// RAM directly; PGM dumps and the sim round-trip read it back through that
// RAM's own debug port (img_dbg_*), not through this module.
module capture_384 #(
    parameter IMG_DW     = 10,
    parameter RED_MIN    = 8'd64,
    // 1.04% of the crop, as the 96x96 build's 192 was of its 18,432-pixel
    // window (which itself carried the 48x48 build's 600/57,600 ratio
    // forward). The crop is now 384*384 = 147,456 pixels; SAME RATIO, NOT
    // MEASURED against a real 384x384 capture. Recalibrate against the
    // printed red_count for known red and black cards.
    parameter RED_THRESH = 17'd1536
)(
    input  wire        mipi_clk,          // = MIPI_PIXEL_CLK
    input  wire [9:0]  mipi_data,         // = MIPI_PIXEL_D
    input  wire        mipi_hs,           // = MIPI_PIXEL_HS
    input  wire        mipi_vs,           // = MIPI_PIXEL_VS
    input  wire        capture_trigger,   // 1-cycle pulse, HPS clock domain (async)

    // image write port -> card_cnn_core.u_img (dual-clock)
    output wire        img_wr_en,
    output wire [17:0] img_wr_addr,
    output wire [IMG_DW-1:0] img_wr_data,

    // status, read out through the same clk50 diagnostic port camera_capture.v
    // already exposes (SNAP_RED_COUNT / SNAP_COLOUR_HW)
    output reg         snapshot_done,
    output reg         colour_hw,
    output reg [17:0]  red_count
);

    // crop geometry: 384x384, centred in the 640x480 raw frame
    localparam CROP_X0 = 10'd128;            // 384 columns: 128..511
    localparam CROP_X1 = 10'd512;
    localparam CROP_Y0 = 9'd48;               // 384 rows: 48..431
    localparam CROP_Y1 = 9'd432;

    localparam LAT_RGB = 2;                   // Line_Buffer_J (1) + RAW_RGB_BIN (1)

    // ---------------- raw-stream column/row counters ----------------
    // Same shape as RAW2RGB_J's own internal mX_Cont/mY_Cont (which it does
    // not export): mX resets on the row-start edge of mipi_hs and counts
    // while it is high; mY counts completed rows since mipi_vs's rising edge.
    reg        hs_d, vs_d;
    reg [9:0]  cx;
    reg [8:0]  cy;
    wire vs_rise = mipi_vs & ~vs_d;
    always @(posedge mipi_clk) begin
        hs_d <= mipi_hs;
        vs_d <= mipi_vs;
        if (!mipi_hs)         cx <= 10'd0;
        else                  cx <= cx + 10'd1;
        if (vs_rise)          cy <= 9'd0;              // frame start
        else if (hs_d & ~mipi_hs) cy <= cy + 9'd1;     // end of a row
    end

    // delay to match RAW2RGB_J's LAT_RGB-cycle pipeline
    reg [9:0] cx_d [0:LAT_RGB-1];
    reg [8:0] cy_d [0:LAT_RGB-1];
    integer li;
    always @(posedge mipi_clk) begin
        cx_d[0] <= cx; cy_d[0] <= cy;
        for (li = 1; li < LAT_RGB; li = li + 1) begin
            cx_d[li] <= cx_d[li-1];
            cy_d[li] <= cy_d[li-1];
        end
    end
    wire [9:0] cx_rgb = cx_d[LAT_RGB-1];
    wire [8:0] cy_rgb = cy_d[LAT_RGB-1];

    // ---------------- demosaic: reuse the proven RAW2RGB_J ----------------
    wire [7:0] red, green, blue;
    // .RST(mipi_vs) mirrors camera_capture.v's display instance, which resets
    // RAW_RGB_BIN on its vertical sync; here that is the MIPI frame-valid.
    RAW2RGB_J u_raw2rgb (
        .mCCD_DATA(mipi_data), .CCD_PIXCLK(mipi_clk), .RST(mipi_vs),
        .VGA_CLK(mipi_clk), .READ_Request(mipi_hs),
        .VGA_VS(mipi_vs), .VGA_HS(mipi_hs),
        .oRed(red), .oGreen(green), .oBlue(blue)
    );

    wire in_crop = (cx_rgb >= CROP_X0) && (cx_rgb < CROP_X1)
                && (cy_rgb >= CROP_Y0) && (cy_rgb < CROP_Y1);

    // Y = 0.299 R + 0.587 G + 0.114 B -> (77R + 150G + 29B) >> 8, weights sum
    // to 256. Same formula as downsample_96x96.v.
    wire [15:0] lum16 = {8'b0, red} * 16'd77 + {8'b0, green} * 16'd150
                       + {8'b0, blue} * 16'd29;
    wire [7:0]  gray  = lum16[15:8];

    // Quantise to IMG_DW bits of an approximate Q6.10 magnitude (gray*4, which
    // peaks at 1020 of 1024): keep the top IMG_DW bits. At IMG_DW=10 this is
    // gray<<2 unchanged; at IMG_DW=5 it is gray>>3, matching the trainer's
    // QuantizeInput(bits=5). card_cnn_core.v expands it back by shifting the
    // dropped bits in as zero, which is exact.
    wire [IMG_DW-1:0] img_code = ({gray, 2'b00}) >> (10 - IMG_DW);

    wire [17:0] wr_idx = (cy_rgb - CROP_Y0) * 18'd384 + (cx_rgb - CROP_X0);

    // Ratio test (R > 1.5G and R > 1.5B), same as downsample_96x96.v -- robust
    // to the D8M's warm white balance.
    wire is_red = in_crop && (red > RED_MIN)
                          && ({1'b0, red} > {1'b0, green} + {2'b0, green[7:1]})
                          && ({1'b0, red} > {1'b0, blue}  + {2'b0, blue[7:1]});
    reg [17:0] red_cnt;
    always @(posedge mipi_clk)
        if (vs_rise) red_cnt <= 18'd0; else if (is_red) red_cnt <= red_cnt + 18'd1;

    // ---------------- one-shot capture FSM ----------------
    // Same IDLE/ARMED/CAPTURING shape as downsample_96x96.v: arm on trigger,
    // start writing at the next clean frame boundary, run until the crop's
    // last pixel, then hold (sticky snapshot_done) until the next trigger.
    reg [2:0] trig_sync;
    always @(posedge mipi_clk) trig_sync <= {trig_sync[1:0], capture_trigger};
    wire trigger_pulse = trig_sync[1] & ~trig_sync[2];

    localparam IDLE = 2'd0, ARMED = 2'd1, CAPTURING = 2'd2;
    reg [1:0] state;
    reg       capturing;
    wire [17:0] red_cnt_final = red_cnt + (is_red ? 18'd1 : 18'd0);
    wire frame_done = in_crop && (cy_rgb == CROP_Y1 - 1) && (cx_rgb == CROP_X1 - 1);

    always @(posedge mipi_clk) begin
        case (state)
            IDLE: if (trigger_pulse) begin
                snapshot_done <= 1'b0;
                state <= ARMED;
            end
            ARMED: if (vs_rise) begin
                capturing <= 1'b1;
                state <= CAPTURING;
            end
            CAPTURING: if (frame_done) begin
                capturing     <= 1'b0;
                red_count     <= red_cnt_final;
                colour_hw     <= (red_cnt_final > RED_THRESH);
                snapshot_done <= 1'b1;
                state <= IDLE;
            end
            default: state <= IDLE;
        endcase
    end

    assign img_wr_en   = in_crop && capturing;
    assign img_wr_addr = wr_idx;
    assign img_wr_data = img_code;

endmodule
