// 80x30 colour text layer over the 640x480 HDMI raster: 8x16-pixel cells, a
// CP437 font ROM and the 16-colour CGA palette. It exists for the HPS's GUI
// (M2/hdmi_gui.c): the layout is software, so the dashboard can change without
// a recompile.
//
// Write side (clk50, the PIO clock domain -- no crossing needed there): one
// word per cell, {bg[3:0], fg[3:0], cp437[7:0]}, at row*80 + col. wr_en is a
// level (ghrd_top.v: img_wr_addr[13]); the host sets data, then address with
// the enable bit, then clears the bit, so the cell is written while both are
// stable.
//
// Read side (VGA_CLK): fg_on/fg_rgb/bg_rgb describe the pixel at (x, y) four
// clocks after x/y are presented -- the same lag as camera_capture.v's
// disp_win_d / crop_edge_d and RAW2RGB_J's RGB, so the layer lines up with the
// video without any other delay. text_layer.py selftest replays these four
// stages against a direct render.
module text_overlay (
    input  wire        wr_clk,
    input  wire        wr_en,
    input  wire [11:0] wr_addr,            // row*80 + col, 0..2399
    input  wire [15:0] wr_data,            // {bg, fg, cp437}

    input  wire        clk,                // VGA_CLK
    input  wire [10:0] x,                  // VGA_Controller cur_x / cur_y
    input  wire [10:0] y,
    output reg         fg_on,              // glyph pixel set
    output reg  [23:0] fg_rgb,
    output reg  [23:0] bg_rgb
);

    // s1: cell address. x = 640 appears for one clock in blanking (the
    // controller's 801-clock line); it reads past DEPTH, which is harmless.
    reg [11:0] a1;
    reg [3:0]  yr1;
    reg [2:0]  xb1;
    always @(posedge clk) begin
        a1  <= {y[8:4], 6'b0} + {y[8:4], 4'b0} + x[9:3];
        yr1 <= y[3:0];
        xb1 <= x[2:0];
    end

    // s2: the cell word. An explicit M10K altsyncram (same style as FRAM_BUFF.v),
    // not an inferred array: with the device's M10K nearly full, synthesis ran
    // out of RAM-inference budget (Warning 276002) and built the inferred
    // version from 38k flip-flops. Address registered on clk, output not, so
    // the read still takes exactly one clock -- the pipeline is unchanged.
    wire [15:0] cell_q;
    altsyncram u_txt (
        .clock0 (wr_clk), .wren_a (wr_en), .address_a (wr_addr), .data_a (wr_data),
        .clock1 (clk), .address_b (a1), .q_b (cell_q),
        .aclr0 (1'b0), .aclr1 (1'b0), .addressstall_a (1'b0), .addressstall_b (1'b0),
        .byteena_a (1'b1), .byteena_b (1'b1),
        .clocken0 (1'b1), .clocken1 (1'b1), .clocken2 (1'b1), .clocken3 (1'b1),
        .data_b ({16{1'b1}}), .eccstatus (), .q_a (),
        .rden_a (1'b1), .rden_b (1'b1), .wren_b (1'b0)
    );
    defparam
        u_txt.operation_mode                     = "DUAL_PORT",
        u_txt.width_a                            = 16,
        u_txt.widthad_a                          = 12,
        u_txt.numwords_a                         = 2400,
        u_txt.width_b                            = 16,
        u_txt.widthad_b                          = 12,
        u_txt.numwords_b                         = 2400,
        u_txt.width_byteena_a                    = 1,
        u_txt.address_reg_b                      = "CLOCK1",
        u_txt.outdata_reg_b                      = "UNREGISTERED",
        u_txt.address_aclr_b                     = "NONE",
        u_txt.outdata_aclr_b                     = "NONE",
        u_txt.clock_enable_input_a               = "BYPASS",
        u_txt.clock_enable_input_b               = "BYPASS",
        u_txt.clock_enable_output_b              = "BYPASS",
        u_txt.read_during_write_mode_mixed_ports = "DONT_CARE",
        u_txt.ram_block_type                     = "M10K",
        u_txt.maximum_depth                      = 512,
        u_txt.power_up_uninitialized             = "FALSE",
        u_txt.intended_device_family             = "Cyclone V",
        u_txt.lpm_type                           = "altsyncram";
    reg [3:0] yr2;
    reg [2:0] xb2;
    always @(posedge clk) begin
        yr2 <= yr1;
        xb2 <= xb1;
    end

    // s3: the glyph row. 256 glyphs x 16 rows, bit 7 = leftmost pixel;
    // font8x16.hex comes from text_layer.py font.
    (* romstyle = "M10K" *) reg [7:0] font [0:4095];
    initial $readmemh("font8x16.hex", font);
    reg [7:0] bits3;
    reg [7:0] attr3;
    reg [2:0] xb3;
    always @(posedge clk) begin
        bits3 <= font[{cell_q[7:0], yr2}];
        attr3 <= cell_q[15:8];
        xb3   <= xb2;
    end

    // s4: the pixel and its colours
    function [23:0] cga;
        input [3:0] c;
        case (c)
            4'd0:  cga = 24'h000000;  4'd1:  cga = 24'h0000AA;
            4'd2:  cga = 24'h00AA00;  4'd3:  cga = 24'h00AAAA;
            4'd4:  cga = 24'hAA0000;  4'd5:  cga = 24'hAA00AA;
            4'd6:  cga = 24'hAA5500;  4'd7:  cga = 24'hAAAAAA;
            4'd8:  cga = 24'h555555;  4'd9:  cga = 24'h5555FF;
            4'd10: cga = 24'h55FF55;  4'd11: cga = 24'h55FFFF;
            4'd12: cga = 24'hFF5555;  4'd13: cga = 24'hFF55FF;
            4'd14: cga = 24'hFFFF55;  default: cga = 24'hFFFFFF;
        endcase
    endfunction

    always @(posedge clk) begin
        fg_on  <= bits3[3'd7 - xb3];
        fg_rgb <= cga(attr3[3:0]);
        bg_rgb <= cga(attr3[7:4]);
    end

endmodule
