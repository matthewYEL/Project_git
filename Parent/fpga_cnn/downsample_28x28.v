module downsample_28x28 (
    input  wire        vga_clk,
    input  wire        vga_vs,
    input  wire [10:0] cur_x,
    input  wire [10:0] cur_y,
    input  wire        pixel_valid,
    input  wire [7:0]  red, green, blue,

    output reg          frame_ready,
    input  wire [9:0]   rd_addr,
    output reg  signed [15:0] rd_data
);
    reg signed [15:0] frame_out [0:783];
    reg [17:0] accum [0:783];
    reg vs_prev;
    wire frame_start = vga_vs & ~vs_prev;

    wire in_crop = (cur_x >= 82) && (cur_x < 558) && (cur_y >= 2) && (cur_y < 478);
    wire [8:0] crop_x = cur_x - 82;
    wire [8:0] crop_y = cur_y - 2;
    wire [4:0] cell_x = crop_x / 17;   // 注意：这两个除法是"除以17"，也不是2的幂，
    wire [4:0] cell_y = crop_y / 17;   // 但这两处只是"单个"组合逻辑除法(不是784份并行)，
                                        // 資源消耗相对小得多，暂不处理，先解决主要矛盾
    wire [9:0] cell_idx = cell_y * 28 + cell_x;
    wire [9:0] gray = red + green + blue;

    // ---------- 新增：串行提交状态机，一次只处理一个格子 ----------
    reg committing;
    reg [9:0] commit_idx;

    always @(posedge vga_clk) begin
        vs_prev <= vga_vs;

        if (frame_start && !committing) begin
            committing <= 1;
            commit_idx <= 0;
        end else if (committing) begin
            // 用"乘以227再右移12位"近似"除以289"，只有1份硬件，串行复用
            frame_out[commit_idx] <= (accum[commit_idx] * 227) >> 12;
            accum[commit_idx] <= 0;
            if (commit_idx == 783) begin
                committing <= 0;
                frame_ready <= 1;
            end else begin
                commit_idx <= commit_idx + 1;
                frame_ready <= 0;
            end
        end else begin
            frame_ready <= 0;
            if (pixel_valid && in_crop)
                accum[cell_idx] <= accum[cell_idx] + gray;
        end
    end

    always @(posedge vga_clk) rd_data <= frame_out[rd_addr];
endmodule