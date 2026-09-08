// ghrd_top.v  --  camera removed so the accelerator fits
//
// WHY THE CAMERA IS GONE
//   camera_capture's ON_CHIP_FRAM frame buffer consumes roughly 410 of the
//   device's 553 M10K blocks -- more than the entire CNN. With the
//   accelerator's 364 blocks and the lab's 64-block on-chip RAM, the design
//   needed 838 and the fitter gave up.
//
//   Without the camera: 364 + 64 = 428 blocks. Fits with room to spare.
//
//   Milestone 1 does not need the camera. Its card criteria are 10 marks for
//   detecting one numeric and one alphabetic rank "under controlled
//   conditions", which the PGM route satisfies -- images on the SD card, fed
//   to the accelerator by card_cnn.c. The 90 marks for the SoC build,
//   dual-core Linux, and a CNN producing a verifiable inference output are
//   all unaffected.
//
// GETTING THE CAMERA BACK LATER
//   It will not fit alongside the accelerator on this device as-is, even with
//   8-bit weights. The frame buffer has to move to DDR3 (which is exactly what
//   the spec's "External DRAM interface" and the viva topic on "frame
//   buffering, external DRAM access and bandwidth considerations" are pointing
//   at), or shrink to just the crop window instead of a full VGA frame.
//
// WHAT ELSE TO DO
//   1. Remove the camera sources from the Quartus project file list:
//      camera_capture.v, RAW2RGB_J, ON_CHIP_FRAM/FRAM_BUFF, the MIPI/HDMI
//      config modules, downsample_28x28.v, capture_snapshot.v.
//      Deleting the instantiation is not enough on its own -- unused modules
//      still get compiled if they are in the project.
//   2. soc_system.qsys needs NO changes. The snapshot and camera-trigger PIOs
//      stay; they simply do nothing now. Leaving them avoids regenerating and
//      costs a handful of logic elements.
//   3. The .qsf will have pin assignments for the removed HDMI/MIPI ports.
//      Quartus warns about orphaned assignments; it does not error. Ignore
//      them, or delete those lines if the warnings bother you.

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
    input              FPGA_CLK3_50

    // HDMI and camera/MIPI ports removed along with camera_capture.
    // Restore them when the frame buffer moves to DDR3.
);

  wire fpga_clk_50;
  wire [3:0]  fpga_dipsw_pio;
  wire [7:0]  fpga_led_pio;
  wire [1:0]  fpga_button_pio;

  assign fpga_clk_50 = FPGA_CLK1_50;
  assign LED = fpga_led_pio;
  assign fpga_dipsw_pio = SW;
  assign fpga_button_pio = KEY;

  wire [1:0] fpga_debounced_buttons;
  wire [7:0] fpga_led_internal;
  wire       hps_fpga_reset_n;

  assign fpga_led_pio = fpga_led_internal;

  // ---- orphaned PIO nets --------------------------------------------------
  // The lab's CNN and the camera are both gone, but their PIOs remain in
  // soc_system. Only the INPUT pios need driving -- an undriven input floats
  // and Quartus warns. Output pios may simply go unread.
  wire [4:0]  cnn_result_wire;
  wire        cnn_start_wire;
  wire [9:0]  img_wr_addr_wire;
  wire [15:0] img_wr_data_wire;
  wire [1:0]  img_wr_ctrl_wire;
  wire [9:0]  snapshot_addr_wire;
  wire [15:0] snapshot_data_wire;
  wire        camera_trigger_wire;

  assign cnn_result_wire   = 5'b0;    // input pio, lab CNN removed
  assign snapshot_data_wire = 16'b0;  // input pio, camera removed

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
  .h2f_reset_reset_n                    (hps_fpga_reset_n),
  .reset_reset_n                        (hps_fpga_reset_n),

  // colour_flag_hw conduit. Tied low: the HPS supplies the colour flag via
  // CONTROL bit 1. CHECK THE EXACT PORT NAME against the generated
  // soc_system.v -- search it for "card_cnn" -- and delete this line if the
  // conduit was not exported.
  .card_cnn_avalon_0_conduit_end_export (1'b0),

  .cnn_result_pio_external_connection_export (cnn_result_wire),
  .cnn_start_pio_external_connection_export  (cnn_start_wire),
  .snapshot_addr_pio_external_connection_export          (snapshot_addr_wire),
  .snapshot_data_pio_external_connection_export          (snapshot_data_wire),
  .camera_capture_trigger_pio_external_connection_export (camera_trigger_wire),
  .img_wr_addr_pio_external_connection_export (img_wr_addr_wire),
  .img_wr_data_pio_external_connection_export (img_wr_data_wire),
  .img_wr_ctrl_pio_external_connection_export (img_wr_ctrl_wire)
);

debounce debounce_inst (
  .clk      (fpga_clk_50),
  .reset_n  (hps_fpga_reset_n),
  .data_in  (fpga_button_pio),
  .data_out (fpga_debounced_buttons)
);
  defparam debounce_inst.WIDTH = 2;
  defparam debounce_inst.POLARITY = "LOW";
  defparam debounce_inst.TIMEOUT = 50000;
  defparam debounce_inst.TIMEOUT_WIDTH = 16;

  // The card CNN accelerator needs no instantiation here -- it lives inside
  // soc_system as an Avalon-MM slave on the lightweight bridge.

endmodule
