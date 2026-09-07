module conv_full (
    input  wire clk,
    input  wire rst,
    input  wire img_load_done,        // 新增：HPS写完图片后触发这个
    input  wire [9:0] img_wr_addr,    // 新增：写地址
    input  wire signed [15:0] img_wr_data,  // 新增：写数据
    input  wire img_wr_en,            // 新增：写使能脉冲
    output reg  done,
    input  wire [11:0] rd_addr,
    output reg  signed [15:0] rd_data
);
    reg signed [15:0] image_mem  [0:783];
    reg signed [15:0] weight_mem [0:35];
    reg signed [15:0] bias_mem   [0:3];

    initial begin
        $readmemh("conv_weight.hex", weight_mem);
        $readmemh("conv_bias.hex",   bias_mem);
    end

    // ---------- 新增：外部写入image_mem的逻辑 ----------
    always @(posedge clk) begin
        if (img_wr_en) begin
            image_mem[img_wr_addr] <= img_wr_data;
        end
    end

    reg [1:0] f; reg [4:0] i; reg [4:0] j;
    reg signed [15:0] conv_out [0:2703];

    reg signed [15:0] px [0:8]; reg signed [15:0] wt [0:8]; reg signed [15:0] bs;
    reg signed [31:0] prod [0:8];
    reg signed [31:0] s01, s23, s45, s67, s0123, s4567, sum_reg, result_reg;
    reg [9:0] px_addr [0:8];
    reg [1:0] f_reg;
    reg [11:0] out_addr_ctr;

    // ---------- 新增：等待图片加载完成的状态 ----------
    localparam WAIT_LOAD=0, CALC_ADDR=1, FETCH=2, MULT=3, SUMM1=4, SUMM2=5, SUMM3=6, WRITE=7;
    reg [2:0] state;
    integer n;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            f<=0; i<=0; j<=0; done<=0; state<=WAIT_LOAD; out_addr_ctr<=0;
        end else begin
            case (state)
                // ---------- 新增：卡在这里，直到HPS说"图片写完了" ----------
                WAIT_LOAD: begin
                    if (img_load_done) state <= CALC_ADDR;
                end

                CALC_ADDR: begin
                    px_addr[0]<=(i+0)*28+j+0; px_addr[1]<=(i+0)*28+j+1; px_addr[2]<=(i+0)*28+j+2;
                    px_addr[3]<=(i+1)*28+j+0; px_addr[4]<=(i+1)*28+j+1; px_addr[5]<=(i+1)*28+j+2;
                    px_addr[6]<=(i+2)*28+j+0; px_addr[7]<=(i+2)*28+j+1; px_addr[8]<=(i+2)*28+j+2;
                    f_reg <= f;
                    state <= FETCH;
                end
                FETCH: begin
                    for (n=0;n<9;n=n+1) px[n] <= image_mem[px_addr[n]];
                    for (n=0;n<9;n=n+1) wt[n] <= weight_mem[f_reg*9+n];
                    bs <= bias_mem[f_reg];
                    state <= MULT;
                end
                MULT: begin
                    for (n=0;n<9;n=n+1) prod[n] <= px[n]*wt[n];
                    state <= SUMM1;
                end
                SUMM1: begin
                    s01<=prod[0]+prod[1]; s23<=prod[2]+prod[3];
                    s45<=prod[4]+prod[5]; s67<=prod[6]+prod[7];
                    state <= SUMM2;
                end
                SUMM2: begin
                    s0123<=s01+s23; s4567<=s45+s67;
                    state <= SUMM3;
                end
                SUMM3: begin
                    sum_reg <= s0123+s4567+prod[8]+bs*16'sd4096;
                    state <= WRITE;
                end
                WRITE: begin
                    result_reg = (sum_reg<0) ? 32'sd0 : sum_reg;
                    conv_out[out_addr_ctr] <= result_reg >>> 12;
                    out_addr_ctr <= out_addr_ctr + 1;
                    if (j==25) begin j<=0;
                        if (i==25) begin i<=0;
                            if (f==3) done<=1; else f<=f+1;
                        end else i<=i+1;
                    end else j<=j+1;
                    state <= CALC_ADDR;
                end
                default: state <= WAIT_LOAD;
            endcase
        end
    end

    always @(posedge clk) rd_data <= conv_out[rd_addr];
endmodule