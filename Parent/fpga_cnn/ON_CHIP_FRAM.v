// Camera frame buffer, 320x240 Bayer.
//
// The D8M bridge delivers 640x480 raw Bayer. Only pixels with x%4 in {0,1} and
// y%4 in {0,1} are stored (2x2-block skip). Adjacent stored pixels therefore
// still alternate R/G and G/B, so the half-resolution image keeps a valid
// Bayer mosaic and RAW2RGB_J demosaics it unchanged. Both the write address
// counter and the RAM write enable are gated by `keep`; gating only the counter
// would let the skipped pixels overwrite each slot with the wrong colour phase.
module ON_CHIP_FRAM  (
    input         W_CLK ,
    input         R_CLK ,
    input         W_DE   ,
    input  [9:0]  W_DATA ,
    output [9:0]  R_DATA ,
    input         W_CLR ,
    input         R_CLR ,
    input         R_DE,
    output [19:0]WR_ADDR,
    output [19:0]RD_ADDR

 );

//--write-side pixel/line position (same cycle as the data)
reg [9:0] wx;
reg [8:0] wy;
reg       rDE, rVS;
always @(posedge W_CLK) begin
    rDE <= W_DE;
    rVS <= W_CLR;
    if (!W_DE)          wx <= 10'd0;
    else                wx <= wx + 1'b1;
    if (!rVS & W_CLR)   wy <= 9'd0;            // frame start (rising edge of VS)
    else if (rDE & !W_DE) wy <= wy + 1'b1;     // end of line
end

wire keep   = ~wx[1] & ~wy[1];
wire W_DE_K = W_DE & keep;

//--read /write address  counter (linear 0..76799)
FRM_COUNTER wrw(.CLOCK( W_CLK),.CLR( W_CLR ),.DE( W_DE_K ),.ADDR( WR_ADDR)  );
FRM_COUNTER rrr(.CLOCK( R_CLK),.CLR( R_CLR ),.DE( R_DE   ),.ADDR( RD_ADDR)  );

FRAM_BUFF GG(
	.wraddress(WR_ADDR[16:0]),
	.rdaddress(RD_ADDR[16:0]),
	.data     (W_DATA [9:0]),
	.wrclock  (W_CLK),
	.rdclock  (R_CLK),
	.wren     (W_DE_K),
	.q        (R_DATA[9:0] )
	);

endmodule
