module ghrd_top (
	output wire [14:0] hps_memory_mem_a,
	output wire [2:0]  hps_memory_mem_ba,
	output wire        hps_memory_mem_ck,
	output wire        hps_memory_mem_ck_n,
	output wire        hps_memory_mem_cke,
	output wire        hps_memory_mem_cs_n,
	output wire        hps_memory_mem_ras_n,
	output wire        hps_memory_mem_cas_n,
	output wire        hps_memory_mem_we_n,
	output wire        hps_memory_mem_reset_n,
	inout  wire [31:0] hps_memory_mem_dq,
	inout  wire [4:0]  hps_memory_mem_dqs,
	inout  wire [4:0]  hps_memory_mem_dqs_n,
	output wire        hps_memory_mem_odt,
	output wire [4:0]  hps_memory_mem_dm,
	input  wire        hps_memory_oct_rzqin,
	output wire        hps_emac1_TX_CLK,
	output wire        hps_emac1_TXD0,
	output wire        hps_emac1_TXD1,
	output wire        hps_emac1_TXD2,
	output wire        hps_emac1_TXD3,
	input  wire        hps_emac1_RXD0,
	inout  wire        hps_emac1_MDIO,
	output wire        hps_emac1_MDC,
	input  wire        hps_emac1_RX_CTL,
	output wire        hps_emac1_TX_CTL,
	input  wire        hps_emac1_RX_CLK,
	input  wire        hps_emac1_RXD1,
	input  wire        hps_emac1_RXD2,
	input  wire        hps_emac1_RXD3,
	inout  wire        hps_sdio_CMD,
	inout  wire        hps_sdio_D0,
	inout  wire        hps_sdio_D1,
	output wire        hps_sdio_CLK,
	inout  wire        hps_sdio_D2,
	inout  wire        hps_sdio_D3,
	inout  wire        hps_usb1_D0,
	inout  wire        hps_usb1_D1,
	inout  wire        hps_usb1_D2,
	inout  wire        hps_usb1_D3,
	inout  wire        hps_usb1_D4,
	inout  wire        hps_usb1_D5,
	inout  wire        hps_usb1_D6,
	inout  wire        hps_usb1_D7,
	input  wire        hps_usb1_CLK,
	output wire        hps_usb1_STP,
	input  wire        hps_usb1_DIR,
	input  wire        hps_usb1_NXT,
	output wire        hps_spim1_CLK,
	output wire        hps_spim1_MOSI,
	input  wire        hps_spim1_MISO,
	output wire        hps_spim1_SS0,
	input  wire        hps_uart0_RX,
	output wire        hps_uart0_TX,
	inout  wire        hps_i2c0_SDA,
	inout  wire        hps_i2c0_SCL,
	inout  wire        hps_i2c1_SDA,
	inout  wire        hps_i2c1_SCL,
	inout  wire        hps_gpio_GPIO09,
	inout  wire        hps_gpio_GPIO35,
	inout  wire        hps_gpio_GPIO40,
	inout  wire        hps_gpio_GPIO53,
	inout  wire        hps_gpio_GPIO54,
	inout  wire        hps_gpio_GPIO61,
    input  wire     [1:0]  KEY,
    output wire     [7:0]  LED,
    input  wire     [3:0]  SW,
    output wire            ADC_CONVST,
    output wire            ADC_SCK,
    output wire            ADC_SDI,
    input  wire            ADC_SDO,
    inout       [15:0] ARDUINO_IO,
    inout              ARDUINO_RESET_N,
    input              FPGA_CLK1_50,
    input              FPGA_CLK2_50,
    input              FPGA_CLK3_50,

    // ---------- 新增：HDMI引脚 ----------
    inout  wire        HDMI_I2C_SCL,
    inout  wire        HDMI_I2C_SDA,
    inout  wire        HDMI_I2S,
    inout  wire        HDMI_LRCLK,
    inout  wire        HDMI_MCLK,
    inout  wire        HDMI_SCLK,
    output wire        HDMI_TX_CLK,
    output wire        HDMI_TX_DE,
    output wire [23:0] HDMI_TX_D,
    output wire        HDMI_TX_HS,
    input  wire        HDMI_TX_INT,
    output wire        HDMI_TX_VS,

    // ---------- 新增：Camera/MIPI引脚 ----------
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

  wire fpga_clk_50;
  wire [3:0]  fpga_dipsw_pio;
  wire [7:0]  fpga_led_pio;
  wire [1:0]  fpga_button_pio;

  assign fpga_clk_50 = FPGA_CLK1_50;
  wire dbg_mipi_rel, dbg_cam_rel, dbg_hdmi_rdy, dbg_pll_ok;
  wire [4:0] dbg_lut_index;
  wire dbg_ack, dbg_ready_latched, dbg_hdmi_int;
  // LED7 = config completed at least once (sticky), LED6 = HDMI_TX_INT,
  // LED5..1 = LUT_INDEX in binary (0-31), LED0 = live ACK (1 = NACK).
  assign LED = {dbg_ready_latched, dbg_hdmi_int, dbg_lut_index, dbg_ack};
  assign fpga_dipsw_pio = SW;
  assign fpga_button_pio = KEY;

  wire [1:0] fpga_debounced_buttons;
  wire [7:0]  fpga_led_internal;
  wire        hps_fpga_reset_n;
  wire [2:0]  hps_reset_req;

  assign fpga_led_pio = fpga_led_internal;

  // ---------- HPS <-> CNN / camera PIOs (see atlas_main.c for the bit layout) ----------
  wire [31:0] cnn_result_wire;     // [3:0] rank [5:4] suit [6] joker [7] done [8] colour [9] snapshot_done [31:16] rank_score
  wire        cnn_start_wire;
  wire [13:0] img_wr_addr_wire;    // 0..9215 = y*96 + x
  wire [15:0] img_wr_data_wire;    // Q6.10 pixel
  wire [3:0]  img_wr_ctrl_wire;    // [0] wr_en, [1] unused, [2] colour override en, [3] colour override val

  // ---------- 新增：摄像头相关连线 ----------
  wire [13:0] snapshot_addr_wire;  // 0..9215 cells, 9216 = red_count, 9217 = colour flag
  wire [15:0] snapshot_data_wire;
  wire        camera_trigger_wire;
  wire        snapshot_done_w, snapshot_colour_w;

soc_system soc_inst (
  .memory_mem_a                         (hps_memory_mem_a),
  .memory_mem_ba                        (hps_memory_mem_ba),
  .memory_mem_ck                        (hps_memory_mem_ck),
  .memory_mem_ck_n                      (hps_memory_mem_ck_n),
  .memory_mem_cke                       (hps_memory_mem_cke),
  .memory_mem_cs_n                      (hps_memory_mem_cs_n),
  .memory_mem_ras_n                     (hps_memory_mem_ras_n),
  .memory_mem_cas_n                     (hps_memory_mem_cas_n),
  .memory_mem_we_n                      (hps_memory_mem_we_n),
  .memory_mem_reset_n                   (hps_memory_mem_reset_n),
  .memory_mem_dq                        (hps_memory_mem_dq),
  .memory_mem_dqs                       (hps_memory_mem_dqs),
  .memory_mem_dqs_n                     (hps_memory_mem_dqs_n),
  .memory_mem_odt                       (hps_memory_mem_odt),
  .memory_mem_dm                        (hps_memory_mem_dm),
  .memory_oct_rzqin                     (hps_memory_oct_rzqin),
  .dipsw_pio_external_connection_export (fpga_dipsw_pio),
  .led_pio_external_connection_in_port  (fpga_led_internal),
  .led_pio_external_connection_out_port (fpga_led_internal),
  .button_pio_external_connection_export(fpga_debounced_buttons),
  .hps_io_hps_io_emac1_inst_TX_CLK(hps_emac1_TX_CLK),
  .hps_io_hps_io_emac1_inst_TXD0  (hps_emac1_TXD0),
  .hps_io_hps_io_emac1_inst_TXD1  (hps_emac1_TXD1),
  .hps_io_hps_io_emac1_inst_TXD2  (hps_emac1_TXD2),
  .hps_io_hps_io_emac1_inst_TXD3  (hps_emac1_TXD3),
  .hps_io_hps_io_emac1_inst_RXD0  (hps_emac1_RXD0),
  .hps_io_hps_io_emac1_inst_MDIO  (hps_emac1_MDIO),
  .hps_io_hps_io_emac1_inst_MDC   (hps_emac1_MDC),
  .hps_io_hps_io_emac1_inst_RX_CTL(hps_emac1_RX_CTL),
  .hps_io_hps_io_emac1_inst_TX_CTL(hps_emac1_TX_CTL),
  .hps_io_hps_io_emac1_inst_RX_CLK(hps_emac1_RX_CLK),
  .hps_io_hps_io_emac1_inst_RXD1  (hps_emac1_RXD1),
  .hps_io_hps_io_emac1_inst_RXD2  (hps_emac1_RXD2),
  .hps_io_hps_io_emac1_inst_RXD3  (hps_emac1_RXD3),
  .hps_io_hps_io_sdio_inst_CMD    (hps_sdio_CMD),
  .hps_io_hps_io_sdio_inst_D0     (hps_sdio_D0),
  .hps_io_hps_io_sdio_inst_D1     (hps_sdio_D1),
  .hps_io_hps_io_sdio_inst_CLK    (hps_sdio_CLK),
  .hps_io_hps_io_sdio_inst_D2     (hps_sdio_D2),
  .hps_io_hps_io_sdio_inst_D3     (hps_sdio_D3),
  .hps_io_hps_io_usb1_inst_D0     (hps_usb1_D0),
  .hps_io_hps_io_usb1_inst_D1     (hps_usb1_D1),
  .hps_io_hps_io_usb1_inst_D2     (hps_usb1_D2),
  .hps_io_hps_io_usb1_inst_D3     (hps_usb1_D3),
  .hps_io_hps_io_usb1_inst_D4     (hps_usb1_D4),
  .hps_io_hps_io_usb1_inst_D5     (hps_usb1_D5),
  .hps_io_hps_io_usb1_inst_D6     (hps_usb1_D6),
  .hps_io_hps_io_usb1_inst_D7     (hps_usb1_D7),
  .hps_io_hps_io_usb1_inst_CLK    (hps_usb1_CLK),
  .hps_io_hps_io_usb1_inst_STP    (hps_usb1_STP),
  .hps_io_hps_io_usb1_inst_DIR    (hps_usb1_DIR),
  .hps_io_hps_io_usb1_inst_NXT    (hps_usb1_NXT),
  .hps_io_hps_io_spim1_inst_CLK   (hps_spim1_CLK),
  .hps_io_hps_io_spim1_inst_MOSI  (hps_spim1_MOSI),
  .hps_io_hps_io_spim1_inst_MISO  (hps_spim1_MISO),
  .hps_io_hps_io_spim1_inst_SS0   (hps_spim1_SS0),
  .hps_io_hps_io_uart0_inst_RX    (hps_uart0_RX),
  .hps_io_hps_io_uart0_inst_TX    (hps_uart0_TX),
  .hps_io_hps_io_i2c0_inst_SDA    (hps_i2c0_SDA),
  .hps_io_hps_io_i2c0_inst_SCL    (hps_i2c0_SCL),
  .hps_io_hps_io_i2c1_inst_SDA    (hps_i2c1_SDA),
  .hps_io_hps_io_i2c1_inst_SCL    (hps_i2c1_SCL),
  .hps_io_hps_io_gpio_inst_GPIO09 (hps_gpio_GPIO09),
  .hps_io_hps_io_gpio_inst_GPIO35 (hps_gpio_GPIO35),
  .hps_io_hps_io_gpio_inst_GPIO40 (hps_gpio_GPIO40),
  .hps_io_hps_io_gpio_inst_GPIO53 (hps_gpio_GPIO53),
  .hps_io_hps_io_gpio_inst_GPIO54 (hps_gpio_GPIO54),
  .hps_io_hps_io_gpio_inst_GPIO61 (hps_gpio_GPIO61),
  .clk_clk                              (fpga_clk_50),
  .h2f_reset_reset_n              (hps_fpga_reset_n),
  .reset_reset_n                        (hps_fpga_reset_n),
  .cnn_result_pio_external_connection_export (cnn_result_wire),
  .cnn_start_pio_external_connection_export  (cnn_start_wire),
  // ---------- 新增：接三个摄像头相关PIO ----------
  .snapshot_addr_pio_external_connection_export          (snapshot_addr_wire),
  .snapshot_data_pio_external_connection_export          (snapshot_data_wire),
  .camera_capture_trigger_pio_external_connection_export (camera_trigger_wire),
  .img_wr_addr_pio_external_connection_export (img_wr_addr_wire),
  .img_wr_data_pio_external_connection_export (img_wr_data_wire),
  .img_wr_ctrl_pio_external_connection_export (img_wr_ctrl_wire)
);

debounce debounce_inst (
  .clk                                  (fpga_clk_50),
  .reset_n                              (hps_fpga_reset_n),
  .data_in                              (fpga_button_pio),
  .data_out                             (fpga_debounced_buttons)
);
  defparam debounce_inst.WIDTH = 2;
  defparam debounce_inst.POLARITY = "LOW";
  defparam debounce_inst.TIMEOUT = 50000;
  defparam debounce_inst.TIMEOUT_WIDTH = 16;

  // ---------- friend's 3-head card CNN (96x96 in, Q6.10) ----------
  wire        core_done;
  wire [3:0]  rank_idx;
  wire [1:0]  suit_idx;
  wire        is_joker;
  wire signed [15:0] rank_score;

  // core reset: hps_fpga_reset_n is asynchronous to fpga_clk_50 -> async assert, sync deassert
  reg [1:0] rst_sync;
  always @(posedge fpga_clk_50 or negedge hps_fpga_reset_n)
    if (!hps_fpga_reset_n) rst_sync <= 2'b11;
    else                   rst_sync <= {rst_sync[0], 1'b0};
  wire core_rst = rst_sync[1];

  // start: PIO level -> one-cycle pulse (the core samples start in its idle state)
  reg cnn_start_d;
  always @(posedge fpga_clk_50) cnn_start_d <= cnn_start_wire;
  wire start_pulse = cnn_start_wire & ~cnn_start_d;

  // colour flag: hardware detector (2-flop synced from the VGA domain) unless software overrides.
  // Latched on start so it holds through the whole ~52 ms inference.
  reg [1:0] col_sync, snap_done_sync;
  always @(posedge fpga_clk_50) begin
    col_sync       <= {col_sync[0],       snapshot_colour_w};
    snap_done_sync <= {snap_done_sync[0], snapshot_done_w};
  end
  wire colour_sel = img_wr_ctrl_wire[2] ? img_wr_ctrl_wire[3] : col_sync[1];
  reg  colour_lat;
  always @(posedge fpga_clk_50) if (start_pulse) colour_lat <= colour_sel;

  // the core's done is a 1-cycle pulse; software polls a sticky copy
  reg done_sticky;
  always @(posedge fpga_clk_50)
    if (core_rst | start_pulse) done_sticky <= 1'b0;
    else if (core_done)         done_sticky <= 1'b1;

  card_cnn_core u_card_cnn (
      .clk          (fpga_clk_50),
      .rst          (core_rst),
      .start        (start_pulse),
      .img_wr_en    (img_wr_ctrl_wire[0]),
      .img_wr_addr  (img_wr_addr_wire),
      .img_wr_data  (img_wr_data_wire),
      .colour_flag  (colour_lat),
      .rank_idx     (rank_idx),
      .suit_idx     (suit_idx),
      .is_joker     (is_joker),
      .rank_score   (rank_score),
      .done         (core_done)
  );

  assign cnn_result_wire = {rank_score, 6'b0, snap_done_sync[1], colour_lat, done_sticky, is_joker, suit_idx, rank_idx};

  // ---------- 新增：例化摄像头模块 ----------
  camera_capture u_camera (
      .clk50               (fpga_clk_50),
      .clk2_50             (FPGA_CLK2_50),
      .rst_n               (hps_fpga_reset_n),
      .capture_trigger_in  (camera_trigger_wire),

      .HDMI_I2C_SCL(HDMI_I2C_SCL), .HDMI_I2C_SDA(HDMI_I2C_SDA),
      .HDMI_I2S(HDMI_I2S), .HDMI_LRCLK(HDMI_LRCLK),
      .HDMI_MCLK(HDMI_MCLK), .HDMI_SCLK(HDMI_SCLK),
      .HDMI_TX_CLK(HDMI_TX_CLK), .HDMI_TX_DE(HDMI_TX_DE),
      .HDMI_TX_D(HDMI_TX_D), .HDMI_TX_HS(HDMI_TX_HS),
      .HDMI_TX_INT(HDMI_TX_INT), .HDMI_TX_VS(HDMI_TX_VS),

      .CAMERA_I2C_SCL(CAMERA_I2C_SCL), .CAMERA_I2C_SDA(CAMERA_I2C_SDA),
      .CAMERA_PWDN_n(CAMERA_PWDN_n), .MIPI_CS_n(MIPI_CS_n),
      .MIPI_I2C_SCL(MIPI_I2C_SCL), .MIPI_I2C_SDA(MIPI_I2C_SDA),
      .MIPI_MCLK(MIPI_MCLK), .MIPI_PIXEL_CLK(MIPI_PIXEL_CLK),
      .MIPI_PIXEL_D(MIPI_PIXEL_D),
      .MIPI_PIXEL_HS(MIPI_PIXEL_HS), .MIPI_PIXEL_VS(MIPI_PIXEL_VS),
      .MIPI_REFCLK(MIPI_REFCLK), .MIPI_RESET_n(MIPI_RESET_n),

      .dbg_mipi_release(dbg_mipi_rel), .dbg_camera_release(dbg_cam_rel),
      .dbg_hdmi_ready(dbg_hdmi_rdy),   .dbg_pll_ok(dbg_pll_ok),
      .dbg_lut_index(dbg_lut_index),   .dbg_ack(dbg_ack),
      .dbg_ready_latched(dbg_ready_latched), .dbg_hdmi_int(dbg_hdmi_int),

      .hps_rd_addr(snapshot_addr_wire),
      .hps_rd_data(snapshot_data_wire),
      .snapshot_done(snapshot_done_w),
      .snapshot_colour(snapshot_colour_w)
  );

endmodule
