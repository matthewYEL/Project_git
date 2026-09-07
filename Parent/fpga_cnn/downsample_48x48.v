// 48x48 box-average downsampler + snapshot buffer + red/black detector.
// Replaces downsample_28x28.v and capture_snapshot.v.
//
// Video side (vga_clk): `win` is the 320x240 display window (READ_Request
// qualified, undelayed). RGB out of RAW2RGB_J lags READ_Request by LAT cycles
// (FRAM 2 + line buffer 1 + RAW_RGB_BIN 1), so a delayed copy `win_d` frames
// the pixels. Inside the window the centre 240x240 (px 40..279) is split into
// 48x48 cells of 5x5. Each cell's luminance sum (ITU-R 601, 0..255 per pixel,
// max 25*255 = 6375) is written into a 2304x16 frame RAM while a capture is
// in progress. Luminance, not R+G+B: the model was trained on PIL convert("L").
//
// The running sums of the current cell row live in a 48-entry circular shift
// register: every completed cell pushes its sum at the tail, so after the 48
// cells of a row the register is back in alignment and acc[0] is always the
// cell being completed. No accumulator RAM, no read-modify-write hazard.
//
// HPS side (rd_clk): rd_addr 0..2303 = cell sums (raw, software scales to
// Q6.10 as sum*1024/6375); 2304 = red pixel count of the captured frame;
// 2305 = hardware colour flag (1 = red card). Stable once snapshot_done is 1.
module downsample_48x48 #(
    parameter LAT        = 4,
    parameter RED_MIN    = 8'd64,
    parameter RED_THRESH = 16'd600
)(
    input  wire        vga_clk,
    input  wire        vga_vs,
    input  wire        win,
    input  wire [7:0]  red, green, blue,
    input  wire        capture_trigger,     // 1-cycle pulse, vga_clk domain

    input  wire        rd_clk,
    input  wire [11:0] rd_addr,
    output wire [15:0] rd_data,

    output reg         snapshot_done,
    output reg         colour_hw
);

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

    wire in_crop = win_d && (px >= 9'd40) && (px < 9'd280);   // centre 240 px

    // 5x5 cell counters. Columns count inside the crop; rows advance at row_end.
    reg [2:0] col5;
    reg [5:0] cell_x;
    always @(posedge vga_clk) begin
        if (!in_crop) begin
            col5   <= 3'd0;
            cell_x <= 6'd0;
        end else if (col5 == 3'd4) begin
            col5   <= 3'd0;
            cell_x <= cell_x + 1'b1;
        end else begin
            col5   <= col5 + 1'b1;
        end
    end

    reg [2:0] row5;
    reg [5:0] cell_y;
    always @(posedge vga_clk) begin
        if (vs_rise) begin
            row5   <= 3'd0;
            cell_y <= 6'd0;
        end else if (row_end) begin
            if (row5 == 3'd4) begin
                row5   <= 3'd0;
                cell_y <= cell_y + 1'b1;
            end else begin
                row5   <= row5 + 1'b1;
            end
        end
    end

    // ---------------- luminance + accumulation ----------------
    // Y = 0.299 R + 0.587 G + 0.114 B  ->  (77 R + 150 G + 29 B) >> 8, weights sum to 256.
    wire [15:0] lum16 = {8'b0, red} * 16'd77 + {8'b0, green} * 16'd150 + {8'b0, blue} * 16'd29;
    wire [7:0]  gray  = lum16[15:8];                     // 0..255

    wire        cell_done = in_crop && (col5 == 3'd4);
    wire        last_row  = (row5 == 3'd4);

    reg  [12:0] csum;                                    // pixels 0..3 of the current cell
    always @(posedge vga_clk)
        if (col5 == 3'd0) csum <= {5'b0, gray}; else csum <= csum + gray;

    reg  [15:0] acc [0:47];
    wire [15:0] new_sum = acc[0] + csum + gray;          // 25-pixel sum, max 6375

    integer i;
    always @(posedge vga_clk) begin
        if (vs_rise) begin
            for (i = 0; i < 48; i = i + 1) acc[i] <= 16'd0;
        end else if (cell_done) begin
            for (i = 0; i < 47; i = i + 1) acc[i] <= acc[i+1];
            acc[47] <= last_row ? 16'd0 : new_sum;
        end
    end

    wire frame_done = cell_done && last_row && (cell_y == 6'd47) && (cell_x == 6'd47);

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

    reg  [15:0] frame [0:2303];
    wire [11:0] wr_idx = {cell_y, 5'b0} + {cell_y, 4'b0} + cell_x;   // cell_y*48 + cell_x
    always @(posedge vga_clk)
        if (cell_done && last_row && capturing) frame[wr_idx] <= new_sum;

    // HPS read port (rd_clk). Mux the two status words AFTER the RAM output
    // register so Quartus keeps inferring the dual-clock RAM.
    reg [15:0] ram_q;
    reg [11:0] addr_d;
    always @(posedge rd_clk) begin
        ram_q  <= frame[rd_addr];
        addr_d <= rd_addr;
    end
    assign rd_data = (addr_d == 12'd2304) ? red_count_lat :
                     (addr_d == 12'd2305) ? {15'b0, colour_hw} : ram_q;

endmodule
