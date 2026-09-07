module button_edge_detect (
    input  wire clk,
    input  wire key_n,        // 按钮原始信号，未按下时是1，按下是0
    output reg  trigger_pulse // 按下瞬间，只产生1个周期的脉冲
);
    reg key_prev;
    always @(posedge clk) begin
        key_prev <= key_n;
        trigger_pulse <= key_prev & ~key_n;   // 检测下降沿（按下的瞬间）
    end
endmodule