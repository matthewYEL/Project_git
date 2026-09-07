// 96x96 box-average downsampler + snapshot buffer + red/black detector.
// Replaces downsample_48x48.v (which used 5x5 cells over a square 240x240 crop
// and left the host to trim 29 of the 48 columns back to the model's aspect).
//
// Video side (vga_clk): `win` is the 320x240 display window (READ_Request
// qualified, undelayed). RGB out of RAW2RGB_J lags READ_Request by LAT cycles
// (FRAM 2 + line buffer 1 + RAW_RGB_BIN 1), so a delayed copy `win_d` frames
// the pixels.
//
// CELLS ARE 1 PIXEL WIDE AND 2 TALL, over a 96 x 192 crop.
//
// The 48x48 build cropped a square 240x240 and the host then kept only columns
// 9..37 (ZOOM_W 29), nearest-neighbour stretching 29 real columns back to 48 --
// so 40% of the network's horizontal samples carried no new information. Here
// the crop itself carries the model's aspect (96/192 = 0.5), every column is a
// real buffer pixel, and the host upload is 1:1. Horizontal detail per network
// column goes from ~0.55 mm to ~0.167 mm of card.
//
// Cell sums are therefore 2 pixels, max 2*255 = 510 (was 25 px / 6375). The
// host scales to Q6.10 as sum*1024/510; SUM_MAX must move with this file.
//
// A cell is one column wide, so there is no horizontal accumulation and the
// 48-entry circular shift register the old version needed is gone. One 96-entry
// array holds the first row of each vertical pair; the second row reads it back
// at the same index and writes the sum. Never read and written in the same
// cycle -- phase 0 writes, phase 1 reads.
//
// HPS side (rd_clk): rd_addr 0..9215 = cell sums (raw, software scales);
// 9216 = red pixel count of the captured frame; 9217 = hardware colour flag
// (1 = red card). Stable once snapshot_done is 1.
module downsample_96x96 #(
    parameter LAT        = 4,
    parameter RED_MIN    = 8'd64,
    // 1.04% of the crop, as the 48x48 build's 600 was of its 57,600-pixel
    // square crop. The crop is now 96*192 = 18,432 pixels. Recalibrate against
    // the printed red_count for known red and black cards.
    parameter RED_THRESH = 16'd192
)(
    input  wire        vga_clk,
    input  wire        vga_vs,
    input  wire        win,
    input  wire [7:0]  red, green, blue,
    input  wire        capture_trigger,     // 1-cycle pulse, vga_clk domain

    input  wire        rd_clk,
    input  wire [13:0] rd_addr,
    output wire [15:0] rd_data,

    output reg         snapshot_done,
    output reg         colour_hw
);

    // crop geometry, centred in the 320x240 window
    localparam CROP_X0 = 9'd112;             // 96 columns: 112..207
    localparam CROP_X1 = 9'd208;
    localparam CROP_Y0 = 8'd24;              // 192 rows: 24..215
    localparam CROP_Y1 = 8'd216;

    // ---------------- window alignment and timing ----------------
    reg [LAT-1:0] win_sr;
    always @(posedge vga_clk) win_sr <= {win_sr[LAT-2:0], win};
    wire win_d = win_sr[LAT-1];              // aligned with RED/GREEN/BLUE

    reg vs_d, win_dd;
    always @(posedge vga_clk) begin
        vs_d   <= vga_vs;
        win_dd <= win_d;
    end
    wire vs_rise = vga_vs & ~vs_d;           // start of a new frame
    wire row_end = win_dd & ~win_d;          // end of a window row

    reg [8:0] px;                            // 0..319 inside the window row
    always @(posedge vga_clk)
        if (!win_d) px <= 9'd0; else px <= px + 1'b1;

    // Window row index. The 48x48 version derived the cell row by counting
    // row_end events straight into a cell counter; the vertical crop needs the
    // absolute row as well, so count that and derive the cell row from it.
    reg [7:0] win_row;                       // 0..239
    always @(posedge vga_clk)
        if (vs_rise)      win_row <= 8'd0;
        else if (row_end) win_row <= win_row + 1'b1;

    wire in_cols = win_d && (px >= CROP_X0) && (px < CROP_X1);
    wire in_rows = (win_row >= CROP_Y0) && (win_row < CROP_Y1);
    wire in_crop = in_cols && in_rows;

    wire [6:0] cell_x   = px - CROP_X0;                // 0..95, cells are 1 px wide
    wire [7:0] row_off  = win_row - CROP_Y0;           // 0..191
    wire [6:0] cell_y   = row_off[7:1];                // 0..95
    wire       row_phase = row_off[0];                 // 0 = first row of the pair

    // ---------------- luminance + accumulation ----------------
    // Y = 0.299 R + 0.587 G + 0.114 B  ->  (77 R + 150 G + 29 B) >> 8, weights sum to 256.
    wire [15:0] lum16 = {8'b0, red} * 16'd77 + {8'b0, green} * 16'd150 + {8'b0, blue} * 16'd29;
    wire [7:0]  gray  = lum16[15:8];                     // 0..255

    reg  [7:0] acc [0:95];                   // first row of the current cell row
    wire [8:0] new_sum = {1'b0, acc[cell_x]} + {1'b0, gray};   // max 510

    always @(posedge vga_clk)
        if (in_crop && !row_phase) acc[cell_x] <= gray;

    wire cell_done  = in_crop && row_phase;  // second row completes the cell
    wire frame_done = cell_done && (cell_y == 7'd95) && (cell_x == 7'd95);

    // ---------------- red/black detector ----------------
    // Ratio test (R > 1.5G and R > 1.5B) is robust to the D8M's warm white balance.
    wire is_red = in_crop && (red > RED_MIN)
                          && ({1'b0, red} > {1'b0, green} + {2'b0, green[7:1]})
                          && ({1'b0, red} > {1'b0, blue}  + {2'b0, blue[7:1]});
    reg [15:0] red_cnt;
    always @(posedge vga_clk)
        if (vs_rise) red_cnt <= 16'd0; else if (is_red) red_cnt <= red_cnt + 1'b1;

    // ---------------- capture FSM + frame RAM ----------------
    localparam IDLE = 2'd0, ARMED = 2'd1, CAPTURING = 2'd2;
    reg [1:0]  state;
    reg        capturing;
    reg [15:0] red_count_lat;
    wire [15:0] red_cnt_final = red_cnt + (is_red ? 16'd1 : 16'd0);

    always @(posedge vga_clk) begin
        case (state)
            IDLE: if (capture_trigger) begin
                snapshot_done <= 1'b0;
                state <= ARMED;
            end
            ARMED: if (vs_rise) begin           // start on a clean frame boundary
                capturing <= 1'b1;
                state <= CAPTURING;
            end
            CAPTURING: if (frame_done) begin
                capturing     <= 1'b0;
                red_count_lat <= red_cnt_final;
                colour_hw     <= (red_cnt_final > RED_THRESH);
                snapshot_done <= 1'b1;
                state <= IDLE;
            end
            default: state <= IDLE;
        endcase
    end

    reg  [15:0] frame [0:9215];
    // cell_y*96 = cell_y*64 + cell_y*32. Both terms are padded to 14 bits
    // deliberately: the expression width is the widest operand, and at 13 bits
    // the 9,120 maximum would wrap.
    wire [13:0] wr_idx = {1'b0, cell_y, 6'b0} + {2'b0, cell_y, 5'b0} + {7'b0, cell_x};
    always @(posedge vga_clk)
        if (cell_done && capturing) frame[wr_idx] <= {7'b0, new_sum};

    // HPS read port (rd_clk). Mux the two status words AFTER the RAM output
    // register so Quartus keeps inferring the dual-clock RAM.
    reg [15:0] ram_q;
    reg [13:0] addr_d;
    always @(posedge rd_clk) begin
        ram_q  <= frame[rd_addr];
        addr_d <= rd_addr;
    end
    assign rd_data = (addr_d == 14'd9216) ? red_count_lat :
                     (addr_d == 14'd9217) ? {15'b0, colour_hw} : ram_q;

endmodule
