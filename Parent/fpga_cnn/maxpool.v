module maxpool (
    input  wire clk,
    input  wire rst,
    input  wire conv_done,
    output reg  done,

    output reg  [11:0] conv_rd_addr,
    input  wire signed [15:0] conv_rd_data,

    input  wire [9:0] rd_addr,
    output reg  signed [15:0] rd_data
);

    reg [1:0] f;
    reg [3:0] i;
    reg [3:0] j;

    reg signed [15:0] pool_out [0:675];

    reg [11:0] addr0_reg, addr1_reg, addr2_reg, addr3_reg;
    reg [11:0] base_f;
    reg [9:0]  out_addr_reg;   // 新增：预先算好写地址

    localparam CALC=0,
               A0=1, W0=2, L0=3,
               A1=4, W1=5, L1=6,
               A2=7, W2=8, L2=9,
               A3=10, W3=11, L3=12,
               MAXCALC1=13, MAXCALC2=14, WR=15;
    reg [3:0] state;

    reg signed [15:0] v0, v1, v2, v3;
    reg signed [15:0] max01_reg, max23_reg, max_result_reg;   // 新增：中间寄存器

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            f <= 0; i <= 0; j <= 0; done <= 0; state <= CALC; base_f <= 0;
        end else if (conv_done && !done) begin
            case (state)
                CALC: begin
                    addr0_reg <= base_f + (i*2+0)*26 + (j*2+0);
                    addr1_reg <= base_f + (i*2+0)*26 + (j*2+1);
                    addr2_reg <= base_f + (i*2+1)*26 + (j*2+0);
                    addr3_reg <= base_f + (i*2+1)*26 + (j*2+1);
                    out_addr_reg <= f*169 + i*13 + j;   // 新增：这里一起算好
                    state <= A0;
                end

                A0: begin conv_rd_addr <= addr0_reg; state <= W0; end
                W0: begin state <= L0; end
                L0: begin v0 <= conv_rd_data; state <= A1; end

                A1: begin conv_rd_addr <= addr1_reg; state <= W1; end
                W1: begin state <= L1; end
                L1: begin v1 <= conv_rd_data; state <= A2; end

                A2: begin conv_rd_addr <= addr2_reg; state <= W2; end
                W2: begin state <= L2; end
                L2: begin v2 <= conv_rd_data; state <= A3; end

                A3: begin conv_rd_addr <= addr3_reg; state <= W3; end
                W3: begin state <= L3; end
                L3: begin v3 <= conv_rd_data; state <= MAXCALC1; end

                // ---------- 新增：把3次比较拆成2个周期 ----------
                MAXCALC1: begin
                    max01_reg <= (v0 > v1) ? v0 : v1;
                    max23_reg <= (v2 > v3) ? v2 : v3;
                    state <= MAXCALC2;
                end

                MAXCALC2: begin
                    max_result_reg <= (max01_reg > max23_reg) ? max01_reg : max23_reg;
                    state <= WR;
                end

                // ---------- WR现在只做"写入内存"，不再做比较 ----------
                WR: begin
                    pool_out[out_addr_reg] <= max_result_reg;
                    if (j == 12) begin
                        j <= 0;
                        if (i == 12) begin
                            i <= 0;
                            if (f == 3) done <= 1;
                            else begin f <= f+1; base_f <= base_f + 676; state <= CALC; end
                        end else begin i <= i+1; state <= CALC; end
                    end else begin j <= j+1; state <= CALC; end
                end

                default: state <= CALC;
            endcase
        end
    end

    always @(posedge clk) begin
        rd_data <= pool_out[rd_addr];
    end

endmodule