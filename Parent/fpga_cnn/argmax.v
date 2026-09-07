module argmax (
    input  wire clk,
    input  wire rst,
    input  wire fc_done,
    input  wire signed [31:0] fc_out0, fc_out1, fc_out2, fc_out3, fc_out4,
    input  wire signed [31:0] fc_out5, fc_out6, fc_out7, fc_out8, fc_out9,
    input  wire signed [31:0] fc_out10, fc_out11, fc_out12, fc_out13,
    output reg  done,
    // now a CLASS INDEX 0..13 into the card label table, not a digit
    output reg [3:0] predicted_digit
);

    reg signed [31:0] vals [0:13];
    reg signed [31:0] max_val;
    reg [3:0] max_idx;
    reg [3:0] idx;
    reg signed [31:0] cur_val;   // 新增：先把当前要比的数读出来

    localparam LOAD=0, READ=1, CMP=2, FINISH=3;
    reg [1:0] state;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            done <= 0;
            state <= LOAD;
        end else begin
            case (state)
                LOAD: if (fc_done) begin
                    vals[0] <= fc_out0; vals[1] <= fc_out1; vals[2] <= fc_out2;
                    vals[3] <= fc_out3; vals[4] <= fc_out4; vals[5] <= fc_out5;
                    vals[6] <= fc_out6; vals[7] <= fc_out7; vals[8] <= fc_out8;
                    vals[9] <= fc_out9;
                    vals[10] <= fc_out10; vals[11] <= fc_out11;
                    vals[12] <= fc_out12; vals[13] <= fc_out13;

                    max_val <= fc_out0;
                    max_idx <= 0;
                    idx     <= 1;
                    state   <= READ;
                end

                // ---------- 新增：单独一个周期，只读出vals[idx] ----------
                READ: begin
                    cur_val <= vals[idx];
                    state <= CMP;
                end

                // ---------- CMP现在只做比较，不再夹带数组读取 ----------
                CMP: begin
                    if (cur_val > max_val) begin
                        max_val <= cur_val;
                        max_idx <= idx;
                    end

                    if (idx == 13) begin
                        state <= FINISH;
                    end else begin
                        idx <= idx + 1;
                        state <= READ;
                    end
                end

                FINISH: begin
                    predicted_digit <= max_idx;
                    done <= 1;
                end

                default: state <= LOAD;
            endcase
        end
    end

endmodule