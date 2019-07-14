package mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.hash;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Text;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Err;

public class Sha256{
	
	public static byte[] sha256(byte in[]){
		return sha256(in, in.length*8L);
	}
	
	/** sizeInBits <= in.length*8 */
	public static byte[] sha256(byte in[], long sizeInBits){
		if(sizeInBits != in.length*8) throw new Err("TODO bit alignment by shifting the 0x80 and bitLenTemp");
		int chunks = (in.length+9+63)/64; //512 bit each
		
		//Copy in[] into b[], then pad bit1, then pad bit0s,
		//then append int64 bit length, finishing the last block of 512 bits.
		byte b[] = new byte[chunks*64];
		System.arraycopy(in, 0, b, 0, in.length);
		b[in.length] = (byte)0x80;
		long bitLenTemp = in.length*8;
		for(int i=7; i>=0; i--){
			b[b.length-8+i] = (byte)bitLenTemp;
			bitLenTemp >>>= 8;
		}
		
		int a[] = new int[136];
		//"first 32 bits of the fractional parts of the cube roots of the first 64 primes 2..311"
		a[0]=0x428a2f98;
		a[1]=0x71374491;
		a[2]=0xb5c0fbcf;
		a[3]=0xe9b5dba5;
		a[4]=0x3956c25b;
		a[5]=0x59f111f1;
		a[6]=0x923f82a4;
		a[7]=0xab1c5ed5;
		a[8]=0xd807aa98;
		a[9]=0x12835b01;
		a[10]=0x243185be;
		a[11]=0x550c7dc3;
		a[12]=0x72be5d74;
		a[13]=0x80deb1fe;
		a[14]=0x9bdc06a7;
		a[15]=0xc19bf174;
		a[16]=0xe49b69c1;
		a[17]=0xefbe4786;
		a[18]=0x0fc19dc6;
		a[19]=0x240ca1cc;
		a[20]=0x2de92c6f;
		a[21]=0x4a7484aa;
		a[22]=0x5cb0a9dc;
		a[23]=0x76f988da;
		a[24]=0x983e5152;
		a[25]=0xa831c66d;
		a[26]=0xb00327c8;
		a[27]=0xbf597fc7;
		a[28]=0xc6e00bf3;
		a[29]=0xd5a79147;
		a[30]=0x06ca6351;
		a[31]=0x14292967;
		a[32]=0x27b70a85;
		a[33]=0x2e1b2138;
		a[34]=0x4d2c6dfc;
		a[35]=0x53380d13;
		a[36]=0x650a7354;
		a[37]=0x766a0abb;
		a[38]=0x81c2c92e;
		a[39]=0x92722c85;
		a[40]=0xa2bfe8a1;
		a[41]=0xa81a664b;
		a[42]=0xc24b8b70;
		a[43]=0xc76c51a3;
		a[44]=0xd192e819;
		a[45]=0xd6990624;
		a[46]=0xf40e3585;
		a[47]=0x106aa070;
		a[48]=0x19a4c116;
		a[49]=0x1e376c08;
		a[50]=0x2748774c;
		a[51]=0x34b0bcb5;
		a[52]=0x391c0cb3;
		a[53]=0x4ed8aa4a;
		a[54]=0x5b9cca4f;
		a[55]=0x682e6ff3;
		a[56]=0x748f82ee;
		a[57]=0x78a5636f;
		a[58]=0x84c87814;
		a[59]=0x8cc70208;
		a[60]=0x90befffa;
		a[61]=0xa4506ceb;
		a[62]=0xbef9a3f7;
		a[63]=0xc67178f2;
		//h0-h7 "first 32 bits of the fractional parts of the square roots of the first 8 primes 2..19"
		a[64]=0x6a09e667;
		a[65]=0xbb67ae85;
		a[66]=0x3c6ef372;
		a[67]=0xa54ff53a;
		a[68]=0x510e527f;
		a[69]=0x9b05688c;
		a[70]=0x1f83d9ab;
		a[71]=0x5be0cd19;
		//a[72..135] are the size 64 w array of ints
		for(int chunk=0; chunk<chunks; chunk++){
			final int bOffset = chunk<<6;
			//copy chunk into first 16 words w[0..15] of the message schedule array
			for(int i=0; i<16; i++){
				//Get 4 bytes from b[]
				final int o = bOffset+(i<<2);
				a[72+i] = ((b[o]&0xff)<<24) | ((b[o+1]&0xff)<<16) | ((b[o+2]&0xff)<<8) | (b[o+3]&0xff);
			}
			//Extend the first 16 words into the remaining 48 words w[16..63] of the message schedule array:
			for(int i=16; i<64; i++){
				//s0 := (w[i-15] rightrotate 7) xor (w[i-15] rightrotate 18) xor (w[i-15] rightshift 3)
				//s1 := (w[i-2] rightrotate 17) xor (w[i-2] rightrotate 19) xor (w[i-2] rightshift 10)
				//w[i] := w[i-16] + s0 + w[i-7] + s1
				final int wim15 = a[72+i-15];
				final int s0 = ((wim15>>>7)|(wim15<<25)) ^ ((wim15>>>18)|(wim15<<14)) ^ (wim15>>>3);
				final int wim2 = a[72+i-2];
				final int s1 = ((wim2>>>17)|(wim2<<15)) ^ ((wim2>>>19)|(wim2<<13)) ^ (wim2>>>10);
				a[72+i] = a[72+i-16] + s0 + a[72+i-7] + s1;
			}
			/*
			//a[136..143] are a-h
			for(int i=0; i<8; i++){
				a[136+i] = a[64+i];
			}
			*/
			int A = a[64];
			int B = a[65];
			int C = a[66];
			int D = a[67];
			int E = a[68];
			int F = a[69];
			int G = a[70];
			int H = a[71];
			for(int i=0; i<64; i++){
				/* S1 := (e rightrotate 6) xor (e rightrotate 11) xor (e rightrotate 25)
				ch := (e and f) xor ((not e) and g)
				temp1 := h + S1 + ch + k[i] + w[i]
				S0 := (a rightrotate 2) xor (a rightrotate 13) xor (a rightrotate 22)
				maj := (a and b) xor (a and c) xor (b and c)
				temp2 := S0 + maj
				h := g
				g := f
				f := e
				e := d + temp1
				d := c
				c := b
				b := a
				a := temp1 + temp2
				*/
				final int s1 = ((E>>>6)|(E<<26)) ^ ((E>>>11)|(E<<21)) ^ ((E>>>25)|(E<<7));
				final int ch = (E&F) ^ ((~E)&G);
				final int temp1 = H + s1 + ch + a[i] + a[72+i];
				final int s0 = ((A>>>2)|(A<<30)) ^ ((A>>>13)|(A<<19)) ^ ((A>>>22)|(A<<10));
				final int maj = (A&B) ^ (A&C) ^ (B&C);
				final int temp2 = s0 + maj;
				H = G;
				G = F;
				F = E;
				E = D + temp1;
				D = C;
				C = B;
				B = A;
				A = temp1 + temp2;
			}
			a[64] += A;
			a[65] += B;
			a[66] += C;
			a[67] += D;
			a[68] += E;
			a[69] += F;
			a[70] += G;
			a[71] += H;
		}
		//RETURN h0..h7 = a[64..71]
		byte ret[] = new byte[32];
		for(int i=0; i<8; i++){
			final int ah = a[64+i];
			ret[i*4] = (byte)(ah>>>24);
			ret[i*4+1] = (byte)(ah>>>16);
			ret[i*4+2] = (byte)(ah>>>8);
			ret[i*4+3] = (byte)ah;
		}
		return ret;
	}
	
	static void test(String in, String correctHashHex){
		byte h[] = sha256(Text.stringToBytes(in));
		String hashHex = toHex(h);
		System.out.print("\r\nHash["+hashHex+"] of in["+in+"]\r\n");
		if(!hashHex.equals(correctHashHex)) throw new Err("\r\n"+hashHex+" observed\r\n"+correctHashHex+" correct");
	}
	
	static String toHex(byte b[]){
		StringBuilder sb = new StringBuilder(b.length*2);
		for(int i=0; i<b.length; i++){
			String s = Integer.toHexString(b[i]);
			if(s.length() == 1) s = "0"+s;
			s = s.substring(s.length()-2); //TODO optimize by not creating whole 8 digits. Also dont need strings in middle steps.
			sb.append(s);
		}
		return sb.toString();
	}
	
	public static void main(String[] args){
		test("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
		test("Powcompete is a system of searching per word for anonymous tweets by how much computing time has been spent in each, in lists of them which contain each word. 431ed5a51dfae8adfe31e079",
			"e20ec021db8faab4752f0e2ebba90bd933b217c4d1471aa91900f6fd63165846");
		test("heres some text first 32 bits of the fractional parts of the square roots of the first 8 primes 2..19 text text text",
			"9c38d764eceeae44905b844d9608120df7d62a1d508ad796d1b0d8faf79817cc");
	}

}
