module fc_layer (
    input  wire clk,
    input  wire rst,
    input  wire pool_done,
    output reg  done,

    output reg  [9:0] pool_rd_addr,
    input  wire signed [15:0] pool_rd_data,

    output reg signed [31:0] fc_out0, fc_out1, fc_out2, fc_out3, fc_out4,
    output reg signed [31:0] fc_out5, fc_out6, fc_out7, fc_out8, fc_out9,
    output reg signed [31:0] fc_out10, fc_out11, fc_out12, fc_out13
);

    // 14 card classes (2,3,4,5,6,7,8,9,10,J,Q,K,A,Joker) instead of 10 digits
    reg signed [15:0] fc_weight_mem [0:9463];   // 14 x 676
    reg signed [15:0] fc_bias_mem   [0:13];

    initial begin
        $readmemh("fc_weight.hex", fc_weight_mem);
        $readmemh("fc_bias.hex",   fc_bias_mem);
    end

    reg [3:0] out_idx;
    reg [9:0] k;
    // 14 bits, not 13: max weight_addr is 13*676+675 = 9463, which overflows [12:0] (max 8191)
    reg [13:0] base_addr;
    reg [13:0] weight_addr;
    reg signed [15:0] weight_val_reg;
    reg signed [31:0] product_reg;     // 新增：单独存乘法结果
    reg signed [47:0] acc;
    reg signed [47:0] final_val;

    localparam IDLE=0, SET_ADDR=1, WAIT_READ=2, READ_MEM=3, MULT=4, ADD=5, ADD_BIAS=6, WRITE=7;
    reg [2:0] state;

    wire signed [15:0] bias_val = fc_bias_mem[out_idx];

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            out_idx <= 0; k <= 0; acc <= 0; done <= 0; state <= IDLE;
            pool_rd_addr <= 0; base_addr <= 0;
        end else begin
            case (state)
                IDLE: if (pool_done) begin
                    acc <= 0; k <= 0; base_addr <= 0; pool_rd_addr <= 0; state <= SET_ADDR;
                end

                SET_ADDR: begin
                    pool_rd_addr <= k;
                    weight_addr  <= base_addr + k;
                    state <= WAIT_READ;
                end

                WAIT_READ: begin
                    state <= READ_MEM;
                end

                READ_MEM: begin
                    weight_val_reg <= fc_weight_mem[weight_addr];
                    state <= MULT;
                end

                // ---------- 新增：单独一个周期，只做乘法 ----------
                MULT: begin
                    product_reg <= pool_rd_data * weight_val_reg;
                    state <= ADD;
                end

                // ---------- ADD：现在只做加法，不再夹带乘法 ----------
                ADD: begin
                    acc <= acc + product_reg;
                    if (k == 675) state <= ADD_BIAS;
                    else begin k <= k + 1; state <= SET_ADDR; end
                end

                ADD_BIAS: begin
                    final_val <= acc + (bias_val * 32'sd4096);
                    state <= WRITE;
                end

                WRITE: begin
                    case (out_idx)
                        0: fc_out0 <= final_val; 1: fc_out1 <= final_val;
                        2: fc_out2 <= final_val; 3: fc_out3 <= final_val;
                        4: fc_out4 <= final_val; 5: fc_out5 <= final_val;
                        6: fc_out6 <= final_val; 7: fc_out7 <= final_val;
                        8: fc_out8 <= final_val; 9: fc_out9 <= final_val;
                        10: fc_out10 <= final_val; 11: fc_out11 <= final_val;
                        12: fc_out12 <= final_val; 13: fc_out13 <= final_val;
                    endcase
                    if (out_idx == 13) done <= 1;
                    else begin
                        out_idx   <= out_idx + 1;
                        base_addr <= base_addr + 676;
                        acc <= 0; k <= 0; pool_rd_addr <= 0;
                        state <= SET_ADDR;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule