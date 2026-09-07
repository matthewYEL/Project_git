module cnn_core (
    input  wire        clk50,
    input  wire        start,
    input  wire        img_load_done,        // 新增
    input  wire [9:0]  img_wr_addr,          // 新增
    input  wire signed [15:0] img_wr_data,   // 新增
    input  wire        img_wr_en,            // 新增
    output wire [3:0]  predicted_digit,
    output wire        done
);

    wire pll_locked;
    wire clk;

    my_pll u_pll (
        .refclk   (clk50),
        .rst      (1'b0),
        .outclk_0 (clk),
        .locked   (pll_locked)
    );

    wire rst = start | ~pll_locked;

    wire conv_done, pool_done, fc_done, argmax_done;
    wire [11:0] conv_rd_addr;
    wire signed [15:0] conv_rd_data;
    wire [9:0] pool_rd_addr;
    wire signed [15:0] pool_rd_data;
    wire signed [31:0] fc_out0, fc_out1, fc_out2, fc_out3, fc_out4;
    wire signed [31:0] fc_out5, fc_out6, fc_out7, fc_out8, fc_out9;
    wire signed [31:0] fc_out10, fc_out11, fc_out12, fc_out13;   // 14 card classes

    conv_full u_conv (
        .clk(clk), .rst(rst), .done(conv_done),
        .img_load_done(img_load_done),      // 新增：透传
        .img_wr_addr(img_wr_addr),          // 新增：透传
        .img_wr_data(img_wr_data),          // 新增：透传
        .img_wr_en(img_wr_en),              // 新增：透传
        .rd_addr(conv_rd_addr),
        .rd_data(conv_rd_data)
    );

    maxpool u_pool (
        .clk(clk), .rst(rst), .conv_done(conv_done), .done(pool_done),
        .conv_rd_addr(conv_rd_addr),
        .conv_rd_data(conv_rd_data),
        .rd_addr(pool_rd_addr), .rd_data(pool_rd_data)
    );

    fc_layer u_fc (
        .clk(clk), .rst(rst), .pool_done(pool_done), .done(fc_done),
        .pool_rd_addr(pool_rd_addr), .pool_rd_data(pool_rd_data),
        .fc_out0(fc_out0), .fc_out1(fc_out1), .fc_out2(fc_out2), .fc_out3(fc_out3), .fc_out4(fc_out4),
        .fc_out5(fc_out5), .fc_out6(fc_out6), .fc_out7(fc_out7), .fc_out8(fc_out8), .fc_out9(fc_out9),
        .fc_out10(fc_out10), .fc_out11(fc_out11), .fc_out12(fc_out12), .fc_out13(fc_out13)
    );

    argmax u_argmax (
        .clk(clk), .rst(rst), .fc_done(fc_done),
        .fc_out0(fc_out0), .fc_out1(fc_out1), .fc_out2(fc_out2), .fc_out3(fc_out3), .fc_out4(fc_out4),
        .fc_out5(fc_out5), .fc_out6(fc_out6), .fc_out7(fc_out7), .fc_out8(fc_out8), .fc_out9(fc_out9),
        .fc_out10(fc_out10), .fc_out11(fc_out11), .fc_out12(fc_out12), .fc_out13(fc_out13),
        .done(argmax_done), .predicted_digit(predicted_digit)
    );

    assign done = argmax_done;

endmodule