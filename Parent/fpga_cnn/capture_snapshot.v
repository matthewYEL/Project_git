module capture_snapshot (
    input  wire        clk,
    input  wire        capture_trigger,
    input  wire        frame_ready,

    output reg  [9:0]  src_rd_addr,
    input  wire signed [15:0] src_rd_data,

    input  wire [9:0]  rd_addr,
    output reg  signed [15:0] rd_data,

    output reg          snapshot_done
);
    reg signed [15:0] snapshot [0:783];
    reg [9:0] copy_idx;

    localparam IDLE=0, WAIT_FRAME=1, SET_ADDR=2, WAIT_READ=3, WRITE=4;
    reg [2:0] state;

    always @(posedge clk) begin
        case (state)
            IDLE: begin
                if (capture_trigger) begin
                    snapshot_done <= 0;   // 新一次拍照开始，才清零
                    state <= WAIT_FRAME;
                end
            end
            WAIT_FRAME: if (frame_ready) begin
                copy_idx <= 0;
                state <= SET_ADDR;
            end
            SET_ADDR: begin
                src_rd_addr <= copy_idx;
                state <= WAIT_READ;
            end
            WAIT_READ: state <= WRITE;
            WRITE: begin
                snapshot[copy_idx] <= src_rd_data;
                if (copy_idx == 783) begin
                    snapshot_done <= 1;   // 完成后保持为1，不再自动清零
                    state <= IDLE;
                end else begin
                    copy_idx <= copy_idx + 1;
                    state <= SET_ADDR;
                end
            end
            default: state <= IDLE;
        endcase
    end

    always @(posedge clk) rd_data <= snapshot[rd_addr];
endmodule