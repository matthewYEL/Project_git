module camera_test (
    input  wire        FPGA_CLK1_50,
    input  wire        FPGA_RESET_N,
    output wire [7:0]  FPGA_LED,

    // ---------- D8M摄像头相关引脚（接到GPIO_1排针）----------
    output wire        CAMERA_I2C_SCL,
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

    wire rst_n = FPGA_RESET_N;

    // ---------- 摄像头需要的参考时钟：20MHz ----------
    wire mipi_refclk_internal;
    pll_test u_pll_ref (
        .inclk0 (FPGA_CLK1_50),
        .areset (~rst_n),
        .c0     (mipi_refclk_internal)
    );
    assign MIPI_REFCLK = mipi_refclk_internal;

    // ---------- 复位延迟：确保摄像头/解码器上电后有足够稳定时间 ----------
    wire dly_rst_0, dly_rst_1, dly_rst_2;
    wire reset_n_delayed;
    RESET_DELAY u_reset_delay (
        .iRST   (rst_n),
        .iCLK   (FPGA_CLK1_50),
        .oRST_0 (dly_rst_0),
        .oRST_1 (dly_rst_1),
        .oRST_2 (dly_rst_2),
        .oREADY (reset_n_delayed)
    );

    // ---------- 摄像头+解码器 上电/复位控制 ----------
    assign CAMERA_PWDN_n = 1'b1;
    assign MIPI_CS_n     = 1'b0;
    assign MIPI_RESET_n  = reset_n_delayed;

    // ---------- I2C配置：摄像头 + MIPI解码器 ----------
    wire mipi_bridge_release, camera_i2c_release;
    wire [9:0] step_unused;
    wire vcm_release_unused;

    MIPI_BRIDGE_CAMERA_Config u_cam_config (
        .RESET_N            (reset_n_delayed),
        .CLK_50             (FPGA_CLK1_50),
        .MIPI_I2C_SCL       (MIPI_I2C_SCL),
        .MIPI_I2C_SDA       (MIPI_I2C_SDA),
        .MIPI_I2C_RELEASE   (mipi_bridge_release),
        .CAMERA_I2C_SCL     (CAMERA_I2C_SCL),
        .CAMERA_I2C_SDA     (CAMERA_I2C_SDA),
        .CAMERA_I2C_RELAESE (camera_i2c_release)
    );

    // ---------- 验证逻辑：数1秒内 MIPI_PIXEL_VS 跳变了几次 ----------
    reg [5:0] vs_count;
    reg [5:0] vs_count_latched;
    reg [25:0] sec_counter;   // 50MHz计数到5千万，刚好1秒
    reg        vs_prev;

    always @(posedge FPGA_CLK1_50 or negedge reset_n_delayed) begin
        if (!reset_n_delayed) begin
            vs_count         <= 0;
            vs_count_latched <= 0;
            sec_counter       <= 0;
            vs_prev           <= 0;
        end else begin
            vs_prev <= MIPI_PIXEL_VS;

            // 检测VS的上升沿，代表新的一帧开始
            if (MIPI_PIXEL_VS & ~vs_prev) begin
                vs_count <= vs_count + 1;
            end

            if (sec_counter == 26'd49_999_999) begin
                sec_counter       <= 0;
                vs_count_latched  <= vs_count;   // 每秒更新一次显示值
                vs_count          <= 0;
            end else begin
                sec_counter <= sec_counter + 1;
            end
        end
    end

    // ---------- LED显示：低6位显示帧率数值，第6位=MIPI配置完成，第7位=摄像头配置完成 ----------
    assign FPGA_LED[5:0] = vs_count_latched;
    assign FPGA_LED[6]   = mipi_bridge_release;
    assign FPGA_LED[7]   = camera_i2c_release;

endmodule