module camera_hdmi_test (
    input  wire        FPGA_CLK1_50,
    input  wire        FPGA_CLK2_50,
    input  wire        FPGA_RESET_N,
	 input  wire [1:0]  KEY,
    output wire [7:0]  FPGA_LED,

    // HDMI
    inout  wire        HDMI_I2C_SCL,
    inout  wire        HDMI_I2C_SDA,
    inout  wire 		  HDMI_I2S,
    inout  wire        HDMI_LRCLK,
    inout  wire        HDMI_MCLK,
    inout  wire        HDMI_SCLK,
    output wire        HDMI_TX_CLK,
    output wire        HDMI_TX_DE,
    output wire [23:0] HDMI_TX_D,
    output wire        HDMI_TX_HS,
    input  wire        HDMI_TX_INT,
    output wire        HDMI_TX_VS,

    // Camera/MIPI
    inout  wire        CAMERA_I2C_SCL,
    inout  wire        CAMERA_I2C_SDA,
    output wire        CAMERA_PWDN_n,
    output wire        MIPI_CS_n,
    inout  wire        MIPI_I2C_SCL,
    inout  wire        MIPI_I2C_SDA,
    output wire        MIPI_MCLK,
    input  wire        MIPI_PIXEL_CLK,
    input  wire [9:0]  MIPI_PIXEL_D,
    input  wire        MIPI_PIXEL_HS,
    input  wire        MIPI_PIXEL_VS,
    output wire        MIPI_REFCLK,
    output wire        MIPI_RESET_n
);

    wire RESET_N, RESET_N_DELAY;
    wire MIPI_BRIDGE_RELEASE, CAMERA_MIPI_RELAESE;
    wire AUD_CTRL_CLK, PLL_TEST_OK, VGA_CLK;
    wire [9:0] RD_DATA;
    wire [19:0] WR_ADDR, RD_ADDR;
    wire [7:0] RED, GREEN, BLUE;
    wire VGA_HS, VGA_VS, READ_Request, HDMI_READY;
	 wire [10:0] cur_x, cur_y;
    wire HDMI_I2S_int;

    assign CAMERA_PWDN_n = 1'b1;
    assign MIPI_CS_n     = 1'b0;
    assign MIPI_RESET_n  = RESET_N;

    RESET_DELAY_HDMI dl (
        .RESET_N (FPGA_RESET_N), .CLK (FPGA_CLK1_50),
        .READY0  (RESET_N),      .READY1 (RESET_N_DELAY)
    );

    MIPI_BRIDGE_CAMERA_Config cfin (
        .RESET_N            (RESET_N_DELAY),
        .CLK_50             (FPGA_CLK1_50),
        .MIPI_I2C_SCL       (MIPI_I2C_SCL),
        .MIPI_I2C_SDA       (MIPI_I2C_SDA),
        .MIPI_I2C_RELEASE   (MIPI_BRIDGE_RELEASE),
        .CAMERA_I2C_SCL     (CAMERA_I2C_SCL),
        .CAMERA_I2C_SDA     (CAMERA_I2C_SDA),
        .CAMERA_I2C_RELAESE (CAMERA_MIPI_RELAESE)
    );

    // ---------- 两个PLL：音频参考时钟 + 摄像头20MHz参考时钟/VGA 25MHz像素时钟 ----------
    AUDIO_PLL pll1 (
        .refclk(FPGA_CLK1_50), .rst(1'b0),
        .outclk_0(AUD_CTRL_CLK), .locked(PLL_TEST_OK)
    );

    VIDEO_PLL pll2 (
        .refclk(FPGA_CLK2_50), .rst(1'b0),
        .outclk_0(MIPI_REFCLK),  // 20MHz，摄像头参考时钟
        .outclk_1(VGA_CLK)       // 25MHz，VGA/HDMI像素时钟
    );

    // ---------- 帧缓冲：摄像头像素时钟写入，VGA时钟读出（跨时钟域） ----------
    ON_CHIP_FRAM fra (
        .W_CLK  (MIPI_PIXEL_CLK),
        .W_DE   (MIPI_PIXEL_HS & MIPI_PIXEL_VS),
        .W_DATA (MIPI_PIXEL_D[9:0]),
        .W_CLR  (MIPI_PIXEL_VS),
        .R_CLK  (VGA_CLK),
        .R_DATA (RD_DATA),
        .R_CLR  (VGA_VS),
        .R_DE   (READ_Request),
        .WR_ADDR(WR_ADDR), .RD_ADDR(RD_ADDR)
    );

    // ---------- RAW Bayer转RGB ----------
    RAW2RGB_J u4 (
        .RST(VGA_VS), .CCD_PIXCLK(VGA_CLK), .mCCD_DATA(RD_DATA[9:0]),
        .VGA_CLK(VGA_CLK), .READ_Request(READ_Request),
        .VGA_VS(VGA_VS), .VGA_HS(VGA_HS),
        .oRed(RED), .oGreen(GREEN), .oBlue(BLUE), .oDVAL()
    );

    // ---------- VGA时序发生器（640x480@60，用作HDMI的HS/VS/DE基础） ----------
    VGA_Controller u1 (
        .iRed(10'd0), .iGreen(10'd0), .iBlue(10'd0),
        .oCurrent_X(cur_x), .oCurrent_Y(cur_y), .oAddress(),
        .oRequest(READ_Request),
        .oVGA_R(), .oVGA_G(), .oVGA_B(),
        .oVGA_HS(VGA_HS), .oVGA_VS(VGA_VS),
        .oVGA_SYNC(), .oVGA_BLANK(), .oVGA_CLOCK(),
        .iCLK(VGA_CLK), .iRST_N(1'b1)
    );

    // ---------- HDMI芯片(ADV7513) I2C配置 ----------
    HDMI_TX_AD7513 hdmi (
        .RESET_N(RESET_N), .CLK_50(FPGA_CLK1_50),
        .AUD_CTRL_CLK(AUD_CTRL_CLK), .PLL_TEST_OK(PLL_TEST_OK),
        .HDMI_I2C_SCL(HDMI_I2C_SCL), .HDMI_I2C_SDA(HDMI_I2C_SDA),
        .HDMI_I2S(HDMI_I2S_int), .HDMI_LRCLK(HDMI_LRCLK),
        .HDMI_MCLK(HDMI_MCLK), .HDMI_SCLK(HDMI_SCLK),
        .HDMI_TX_INT(HDMI_TX_INT), .READY(HDMI_READY)
    );

    assign HDMI_TX_CLK = VGA_CLK;
    assign HDMI_TX_D   = READ_Request ? {RED, GREEN, BLUE} : 24'd0;
    assign HDMI_TX_DE  = READ_Request;
    assign HDMI_TX_HS  = VGA_HS;
    assign HDMI_TX_VS  = VGA_VS;
    assign HDMI_I2S    = 4'b0;   // 不需要音频

    // ---------- 调试用LED：I2C配置状态 ----------
    assign FPGA_LED[0] = MIPI_BRIDGE_RELEASE;
    assign FPGA_LED[1] = CAMERA_MIPI_RELAESE;
    assign FPGA_LED[2] = HDMI_READY;
    assign FPGA_LED[3] = PLL_TEST_OK;
	 assign FPGA_LED[4]   = 1'b0;
    // ---------- 新增：拍照捕获相关信号 ----------
    wire capture_trigger;
    wire frame_ready;
    wire snapshot_done;

    button_edge_detect u_btn (
        .clk(VGA_CLK), .key_n(KEY[1]), .trigger_pulse(capture_trigger)
    );

    wire [9:0] snap_src_addr;
    wire signed [15:0] snap_src_data;

    downsample_28x28 u_downsample (
        .vga_clk(VGA_CLK), .vga_vs(VGA_VS),
        .cur_x(cur_x), .cur_y(cur_y),
        .pixel_valid(READ_Request),
        .red(RED), .green(GREEN), .blue(BLUE),
        .frame_ready(frame_ready),
        .rd_addr(snap_src_addr), .rd_data(snap_src_data)
    );

    capture_snapshot u_snapshot (
        .clk(VGA_CLK), .capture_trigger(capture_trigger),
        .frame_ready(frame_ready),
        .src_rd_addr(snap_src_addr), .src_rd_data(snap_src_data),
        .rd_addr(10'd0), .rd_data(),   // 下游读端口先空着，下一步接HPS/CNN时再用
        .snapshot_done(snapshot_done)
    );

    assign FPGA_LED[5] = snapshot_done;   // 临时调试用：拍照完成后这颗灯会亮
	 assign FPGA_LED[6] = frame_ready;
	 assign FPGA_LED[7] = VGA_VS; 
	 
endmodule