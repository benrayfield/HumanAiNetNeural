/** Ben F Rayfield offers this software opensource MIT license */
package immutable.rnn.metarnn;

/** A NodeType defines a nonlinear relation between
a small number of input vars and output vars
which can be computed forward and backproped
and a certain number of them are copied from
output to input like the memory var in LSTM node
and a node can instead be an InputNode which has
just 1 output and is nondeterministic. Between those
nodes are sparse matrix of weightedSums and biases
and those are the only parts that learn.
<br><br>
A nodetype can be entirely defined by a forest of
(float,float)->float and float->float math ops
such as multiply, add, neg, sine, arcsine, exp, etc,
and of course that metadata.
<br><br>
Im coding the first of these in java to make
sure it works and will opencl optimize it later.
<br><br>
OLD...
<br><br>
In each recurrent neuralnet, there is only 1 nodetype,
but it can have similar effect as multiple layers of multiple types.
<br><br>
Heres 4 example nodetypes I'm planning:
<br><br>
GRU:
(indata,ingate,forgetgate)->(out,externalInput),
where exteranlInput is not derived from (indata,ingate,forgetgate)
but is just a math abstraction of where to put the exteranl input.
Only the first n nodes use externalInput and only the
last m nodes have their out interpreted as external output.
<br><br>
(indata,ingate,forgetgate,feedforwardSum)->(gruOut,feedforwardOut,externalInput),
where feedforwardSum sums from the gruOuts and externalInputs
so its like a GRU with a feedforward layer that sigmoids at the end,
like has worked for me in recurrentjava as of 2019-11.
<br><br>
The other 2 node types will be same as those 2 except
will be LSTM instead of GRU so will have an outGate on the left side.
<br><br>
I will keep trying different nodetypes etc until I find one that
is easily opencl optimized. What I want is to opencl optimize
what recurrentjava does as I have it learning chuascircuit
but it only learns it when trained sequentially
instead of multiple dataseq in parallel 1 time step in multiple of
them at once then the next time step in multiple of them at once
and backprop all together. I need a model that makes that optimizable.
*/
public interface NodeType{
	
	public int ins();
	
	public int outs();
	
	/** The first mems() of outs() are copied to the last mems() of ins(),
	such as the memory var in an LSTM node.
	*/
	public int mems();
	
	/** The only nondeterministic node type should be InputNode.
	False means dont call forward(...) or backprop(...) on it.
	*/
	public boolean isDeterministic();
	
	public void forward(float[] ins, float[] getOuts);
	
	/** Uses ins to compute outs, then backprops from outDerivs to getInDerivs */
	public void backprop(float[] getInDerivs, float[] ins, float outDerivs[]);
	
	/** getDerivs[i][o] is change in out[o] per change in in[i],
	at the given ins[] and the outs[] they derive.
	This could be computed using forward(ins,getOuts) using
	epsilon change of each ins[i] and observing change in getOuts[o],
	but in theory it can be done this way with less roundoff
	and in less number of calculations.
	*/
	public void derivAt(float[][] getDerivs, float[] ins);

}














